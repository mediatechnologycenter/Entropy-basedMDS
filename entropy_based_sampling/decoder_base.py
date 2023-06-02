#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

from collections import OrderedDict
import copy
from typing import Tuple

from torch.nn import functional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput

from entropy_based_sampling.decoding_utils import *


class BaseDynamicEnsembleDecoder(ABC):
    """
    Base class which implements the main functionality for dynamic ensemble decoding. For running the decoding with
    specific models(Bart, MBart etc.) this class needs to be inherited from (see
    decoder_implementations.py)
    """

    def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, decoding_hyperparams: Dict,
                 score_reduce_type: str, **kwargs):
        if torch.cuda.is_available():
            device_name = "cuda"
        else:
            device_name = "cpu"
        self.device_name = device_name
        self.model = model
        self.encoder = model.get_encoder()
        self.tokenizer = tokenizer
        self.decoding_hyperparams = decoding_hyperparams
        self.component_states = None
        self.ensemble_state = None
        self.score_reduce_strategy = ScoreReduceStrategy.get_score_reduce_strategy(score_reduce_type)
        self.print_per_step_output = kwargs["print_per_step_output"]
        pmi_score_modifier_key_name = "pmi_score_modifier"
        if pmi_score_modifier_key_name in kwargs:
            self.pmi_score_modifier = kwargs[pmi_score_modifier_key_name]

    def get_ensemble_state_vocab_size(self) -> int:
        assert self.ensemble_state is not None
        return self.ensemble_state['vocab_size']

    def generate_initial_states(self, source_articles: List[str]):
        cleaned_source_articles = [article.strip() for article in source_articles if article.strip() != ""]
        self.component_states = [self.get_start_state(article) for article in cleaned_source_articles]
        assert len(self.component_states) == len(cleaned_source_articles)
        self.ensemble_state = self.get_start_state(cleaned_source_articles[0])

    def get_start_state(self, input_text: str) -> OrderedDict:
        input_ids = self.get_input_ids_from_text(input_text, True)
        decoder_state = self.initialize_generation(input_ids, **self.decoding_hyperparams)
        decoder_state['generated_hyps'] = [
            BeamHypotheses(
                decoder_state['num_beams'],
                decoder_state['max_length'],
                decoder_state['length_penalty'],
                early_stopping=decoder_state['early_stopping'])
            for _ in range(decoder_state['batch_size'])
        ]

        # scores for each sentence in the beam
        decoder_state['beam_scores'] = \
            torch.zeros((decoder_state['batch_size'], decoder_state['num_beams']),
                        dtype=torch.float,
                        device=self.device_name)

        decoder_state['beam_scores'][:, 1:] = -1e9
        decoder_state['beam_scores'] = decoder_state['beam_scores'].view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        decoder_state['past'] = decoder_state[
            'encoder_outputs']  # defined for encoder-decoder models, None for decoder-only models

        # done sentences
        decoder_state['done'] = [False for _ in range(decoder_state['batch_size'])]

        return decoder_state

    def initialize_generation(self,
                              input_ids: Optional[torch.Tensor] = None,
                              max_length: Optional[int] = None,
                              min_length: Optional[int] = None,
                              do_sample: Optional[bool] = None,
                              early_stopping: Optional[bool] = None,
                              num_beams: Optional[int] = None,
                              temperature: Optional[float] = None,
                              top_k: Optional[int] = None,
                              top_p: Optional[float] = None,
                              repetition_penalty: Optional[float] = None,
                              bad_words_ids: Optional[List] = None,
                              bos_token_id: Optional[int] = None,
                              pad_token_id: Optional[int] = None,
                              eos_token_id: Optional[int] = None,
                              length_penalty: Optional[int] = None,
                              no_repeat_ngram_size: Optional[int] = None,
                              num_return_sequences: Optional[int] = None,
                              decoder_start_token_id: Optional[int] = None,
                              **kwargs
                              ) -> OrderedDict:

        model = self.model
        # We cannot generate if the model does not have a LM head
        if model.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not have a LM Head."
                "Please use another model class (e.g. `OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`, "
                "`CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`, `XLMWithLMHeadModel`, "
                "`BartForConditionalGeneration` ) "
            )

        max_length = max_length if max_length is not None else model.config.max_length
        min_length = min_length if min_length is not None else model.config.min_length
        do_sample = do_sample if do_sample is not None else model.config.do_sample
        early_stopping = early_stopping if early_stopping is not None else model.config.early_stopping
        num_beams = num_beams if num_beams is not None else model.config.num_beams
        temperature = temperature if temperature is not None else model.config.temperature
        top_k = top_k if top_k is not None else model.config.top_k
        top_p = top_p if top_p is not None else model.config.top_p
        repetition_penalty = repetition_penalty if repetition_penalty is not None else model.config.repetition_penalty
        bos_token_id = bos_token_id if bos_token_id is not None else model.config.bos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else model.config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else model.config.eos_token_id
        length_penalty = length_penalty if length_penalty is not None else model.config.length_penalty
        no_repeat_ngram_size = (
            no_repeat_ngram_size if no_repeat_ngram_size is not None else model.config.no_repeat_ngram_size
        )
        bad_words_ids = bad_words_ids if bad_words_ids is not None else model.config.bad_words_ids
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else model.config.num_return_sequences
        )
        decoder_start_token_id = (
            decoder_start_token_id if decoder_start_token_id is not None else model.config.decoder_start_token_id
        )

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # override by the input batch_size
        else:
            batch_size = 1

        assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
        assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
        assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
                isinstance(bos_token_id, int) and bos_token_id >= 0
        ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
        assert pad_token_id is None or (
                isinstance(pad_token_id, int) and (pad_token_id >= 0)
        ), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (
                isinstance(eos_token_id, int) and (eos_token_id >= 0)
        ), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, "`length_penalty` should be strictly positive."
        assert (
                isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
        ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
                isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
                bad_words_ids is None or isinstance(bad_words_ids, list) and isinstance(bad_words_ids[0], list)
        ), "`bad_words_ids` is either `None` or a list of lists of tokens that should not be generated"

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids` input "
                "or a `bos_token_id` (integer >= 0) as a first token to start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(model.parameters()).device,
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."

            input_ids = input_ids.to(next(model.parameters()).device)

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams != 1:
                # beam_search greedy generation conditions
                assert (
                        num_beams >= num_return_sequences
                ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams " \
                   ">= " \
                   "num_return_sequences "

            else:
                # no_beam_search greedy generation conditions
                assert (
                        num_return_sequences == 1
                ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > " \
                   "1. Please set num_return_sequences = 1 "

        attention_mask = create_attention_mask(input_ids, pad_token_id)

        # set pad_token_id to eos_token_id if not set. Important that this is done after
        # attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            print("Setting `pad_token_id` to {} (first `eos_token_id`) to generate sequence".format(eos_token_id))
            pad_token_id = eos_token_id

        # current position and vocab size
        vocab_size = model.config.vocab_size

        # set effective batch size and effective batch multiplier according to do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if model.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                    decoder_start_token_id is not None
            ), "decoder_start_token_id or bos_token_id has to be defined for encoder-decoder generation"
            assert hasattr(model, "get_encoder"), "{} should have a 'get_encoder' function defined".format(model)
            assert callable(model.get_encoder), "{} should be a method".format(model.get_encoder)

            # store encoder outputs
            encoder_outputs = self.compute_encoder_outputs(input_ids, attention_mask)

            # create empty decoder_input_ids
            decoder_input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(model.parameters()).device,
            )
            cur_len = 1
            encoder_outputs_batch_size = self.get_encoder_outputs_batch_size(encoder_outputs)
            assert (

                    batch_size == encoder_outputs_batch_size
            ), f"expected encoder_outputs[0] to have 1st dimension bs={batch_size}, got {encoder_outputs_batch_size} "

            # expand batch_idx to assign correct encoder output for expanded input_ids (due to num_beams > 1 and
            # num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = self.expand_encoder_outputs(encoder_outputs, expanded_batch_idxs)
        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = expand_input_dim_to_num_beams(input_ids, input_ids_len, num_beams, batch_size,
                                                      effective_batch_mult,
                                                      effective_batch_size)
            attention_mask = expand_input_dim_to_num_beams(attention_mask, input_ids_len, num_beams, batch_size,
                                                           effective_batch_mult, effective_batch_size)

        if not model.config.is_encoder_decoder:
            decoder_input_ids = input_ids

        return OrderedDict([
            ('model', model),
            ('input_ids', decoder_input_ids),
            ('cur_len', cur_len),
            ('max_length', max_length),
            ('min_length', min_length),
            ('do_sample', do_sample),
            ('early_stopping', early_stopping),
            ('temperature', temperature),
            ('top_k', top_k),
            ('top_p', top_p),
            ('repetition_penalty', repetition_penalty),
            ('no_repeat_ngram_size', no_repeat_ngram_size),
            ('bad_words_ids', bad_words_ids),
            ('bos_token_id', bos_token_id),
            ('pad_token_id', pad_token_id),
            ('decoder_start_token_id', decoder_start_token_id),
            ('eos_token_id', eos_token_id),
            ('batch_size', effective_batch_size),
            ('num_return_sequences', num_return_sequences),
            ('length_penalty', length_penalty),
            ('num_beams', num_beams),
            ('vocab_size', vocab_size),
            ('encoder_outputs', encoder_outputs),
            ('attention_mask', attention_mask)
        ])

    def get_input_ids_from_text(self, input_text: str, pad_to_max_length: bool) -> torch.Tensor:
        tokenized_input = self.tokenizer.batch_encode_plus(
            [input_text],
            max_length=self.decoding_hyperparams['max_length'],
            pad_to_max_length=pad_to_max_length,
            return_tensors='pt'
        )

        tokenized_input_ids = tokenized_input['input_ids'].to(self.device_name)
        return tokenized_input_ids

    @abstractmethod
    def compute_encoder_outputs(self, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor) -> BaseModelOutput:
        pass

    @abstractmethod
    def get_encoder_outputs_batch_size(self, encoder_outputs: torch.Tensor) -> int:
        pass

    def compute_dynamic_ensemble_decoding(self, source_articles: List[str],
                                          timestep_mask: Optional[torch.Tensor] = None) -> Dict:
        """
           Run generation for a number of timesteps
           """
        self.generate_initial_states(source_articles)
        step_mask = None
        for step_idx in range(self.decoding_hyperparams["max_tgt_length"]):
            if timestep_mask is not None:
                if step_idx == timestep_mask.shape[0] - 1:
                    break
                step_mask = timestep_mask[step_idx]

            self.ensembled_beam_search_step(step_mask=step_mask)

        return self.ensemble_state

    def get_summary(self, source_articles: List[str],
                    timestep_mask: Optional[torch.Tensor] = None) -> str:
        self.compute_dynamic_ensemble_decoding(source_articles, timestep_mask)
        ensemble_input_ids = self.ensemble_state['input_ids'][0]
        summary = self.tokenizer.decode(ensemble_input_ids, skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)
        return summary

    def ensembled_beam_search_step(self, step_mask: Optional[torch.Tensor] = None):
        self.initialize_ensemble_state_decoding_stats()

        self.compute_state_scores(step_mask)
        # print("Model allocation:" + str(torch.cuda.memory_allocated()))

        next_tokens, next_scores = self.get_next_token_candidates_from_scores()

        beam_tokens_results = self.compute_next_beam_tokens(next_tokens, next_scores)

        if beam_tokens_results is None:
            return

        beam_tokens, beam_idx = beam_tokens_results
        self.set_state_decoder_input_ids_to_beam_token_ids(beam_tokens, beam_idx)

        self.update_ensemble_state_decoding_stats(beam_tokens, beam_idx)

        self.reorder_internal_states(beam_idx)

        # update current length
        for state in self.component_states:
            state['cur_len'] = state['cur_len'] + 1

        self.ensemble_state['cur_len'] = self.ensemble_state['cur_len'] + 1

        if self.print_per_step_output:
            self.print_output_per_step()

    def initialize_ensemble_state_decoding_stats(self):
        ensemble_state = self.ensemble_state
        if 'decoding_stats' not in ensemble_state:
            # fires on first decoding step
            ensemble_state['decoding_stats'] = []
            for _ in range(len(self.component_states)):
                ensemble_state['decoding_stats'].append([[] for _ in range(ensemble_state['num_beams'])])

    def compute_state_scores(self, step_mask: Optional[torch.Tensor] = None):
        ensemble_state = self.ensemble_state
        for state in self.component_states:

            state['outputs'] = self.outputs_from_state(state)
            state['next_token_logits'] = state['outputs'][0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            state = apply_heuristics_to_logits(state)
            # apply softmax to logits
            state['scores'] = functional.log_softmax(state['next_token_logits'],
                                                     dim=-1)  # (batch_size * num_beams, vocab_size)

            prob_per_beam = functional.softmax(state['next_token_logits'],
                                               dim=-1)
            state['max_prob'] = torch.max(prob_per_beam, dim=-1).values
            state['entropy'] = torch.distributions.Categorical(logits=state['next_token_logits']).entropy()

            if state['model'].config.is_encoder_decoder and ensemble_state['do_sample'] is False:
                state['scores'] = self.prepare_scores_for_generation(
                    state['scores'],
                    cur_len=state['cur_len'],
                    max_length=state['max_length'])

            # set state's eos token prob to zero if min_length is not reached
            if ensemble_state['eos_token_id'] is not None and ensemble_state['cur_len'] < ensemble_state['min_length']:
                state['scores'][:, state['eos_token_id']] = -float("inf")

            if ensemble_state['no_repeat_ngram_size'] > 0:
                # calculate a list of banned tokens to prevent repetitively generating the same ngrams
                num_batch_hypotheses = ensemble_state['batch_size'] * ensemble_state['num_beams']
                # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76
                # /fairseq/sequence_generator.py#L345
                banned_batch_tokens = calc_banned_ngram_tokens(
                    ensemble_state['input_ids'],
                    num_batch_hypotheses,
                    ensemble_state['no_repeat_ngram_size'],
                    ensemble_state['cur_len']
                )
                for i, banned_tokens in enumerate(banned_batch_tokens):
                    state['scores'][i, banned_tokens] = -float("inf")

            if ensemble_state['bad_words_ids'] is not None:
                # calculate a list of banned tokens according to bad words
                banned_tokens = calc_banned_bad_words_ids(
                    ensemble_state['input_ids'],
                    ensemble_state['bad_words_ids']
                )

                for i, banned_tokens in enumerate(banned_tokens):
                    state['scores'][i, banned_tokens] = -float("inf")

            if step_mask is not None:
                state['scores'] = state['scores'] * step_mask

            assert state['scores'].shape == (
                ensemble_state['batch_size'] * ensemble_state['num_beams'],
                ensemble_state['vocab_size']), "Shapes of scores: {} != {}".format(
                state['scores'].shape,
                (ensemble_state['batch_size'] * ensemble_state['num_beams'], ensemble_state['vocab_size'])
            )

            # if model has past, then set the past variable to speed up decoding
            if state['model'].config.use_cache:
                state['past'] = self.get_state_past(state['outputs'])

        if self.pmi_score_modifier is not None:
            self.component_states = self.pmi_score_modifier.apply_point_wise_mutual_information(
                self.component_states)
        ensemble_state['scores'] = self.score_reduce_strategy.reduce_score(component_states=self.component_states)

    @abstractmethod
    def outputs_from_state(self, state: Dict) -> Seq2SeqLMOutput:
        pass

    @staticmethod
    def get_state_past(state_output: Seq2SeqLMOutput) -> Tuple[
        Tuple[Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]], Optional[Tuple[torch.FloatTensor]]],
        Optional[Tuple[Tuple[torch.FloatTensor]]]]:
        return ((state_output.encoder_last_hidden_state, state_output.encoder_hidden_states,
                 state_output.encoder_attentions),
                state_output.past_key_values)

    def prepare_scores_for_generation(self, scores: torch.Tensor, cur_len: int, max_length: int) -> torch.Tensor:
        if cur_len == 1:
            self.force_token_ids_generation(scores, self.model.config.bos_token_id)
        if cur_len == max_length - 1 and self.model.config.eos_token_id is not None:
            self.force_token_ids_generation(scores, self.model.config.eos_token_id)
        return scores

    def force_token_ids_generation(self, scores: torch.Tensor, token_ids: int):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        all_but_token_ids_mask = torch.tensor(
            [x for x in range(self.model.config.vocab_size) if x not in token_ids],
            dtype=torch.long,
            device=next(self.model.parameters()).device,
        )
        assert len(scores.shape) == 2, "scores should be of rank 2 with shape: [batch_size, vocab_size]"
        scores[:, all_but_token_ids_mask] = -float("inf")

    def get_next_token_candidates_from_scores(self) -> Tuple[torch.Tensor, torch.Tensor]:
        ensemble_state = self.ensemble_state

        next_scores = ensemble_state['scores'] + ensemble_state['beam_scores'][:, None].expand_as(
            ensemble_state['scores'])  # (batch_size * num_beams, vocab_size)

        # re-organize to group the beam together (we are keeping top hypotheses across beams)
        next_scores = next_scores.view(
            ensemble_state['batch_size'], ensemble_state['num_beams'] * ensemble_state['vocab_size']
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = \
            torch.topk(
                next_scores,
                2 * ensemble_state['num_beams'],
                dim=1,
                largest=True,
                sorted=True
            )

        assert next_scores.size() == next_tokens.size() == (
            ensemble_state['batch_size'], 2 * ensemble_state['num_beams'])

        return next_tokens, next_scores

    def compute_next_beam_tokens(self, next_tokens: torch.Tensor, next_scores: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor]:
        ensemble_state = self.ensemble_state
        next_batch_beam = []

        # for each input (note currently if we are doing one multi-doc summary, batch_size is 1 for sure)
        for batch_idx in range(ensemble_state['batch_size']):

            # if we are done with this sentence
            if ensemble_state['done'][batch_idx]:
                assert (
                        len(ensemble_state['generated_hyps'][batch_idx]) >= ensemble_state['num_beams']
                ), "Batch can only be done if at least {} beams have been generated".format(ensemble_state['num_beams'])
                assert (
                        ensemble_state['eos_token_id'] is not None and ensemble_state['pad_token_id'] is not None
                ), "generated beams >= num_beams -> eos_token_id and pad_token have to be defined"
                next_batch_beam.extend(
                    [(0, ensemble_state['pad_token_id'], 0)] * ensemble_state['num_beams'])  # pad the batch
                continue

            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence from each beam
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                    zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs (undo beam offset)
                beam_id = torch.div(beam_token_id, ensemble_state['vocab_size'], rounding_mode='floor')
                token_id = beam_token_id % ensemble_state['vocab_size']

                effective_beam_id = batch_idx * ensemble_state['num_beams'] + beam_id

                # add to generated hypotheses if end of sentence or last iteration
                if (ensemble_state['eos_token_id'] is not None) and (token_id.item() == ensemble_state['eos_token_id']):
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= ensemble_state['num_beams']
                    if is_beam_token_worse_than_top_num_beams:
                        continue

                    hyp_metadata = []
                    for state_idx in range(len(ensemble_state['decoding_stats'])):
                        hyp_metadata.append(ensemble_state['decoding_stats'][state_idx][effective_beam_id])
                    ensemble_state['generated_hyps'][batch_idx].add(
                        ensemble_state['input_ids'][effective_beam_id].clone(), beam_token_score.item(),
                        metadata=hyp_metadata
                    )
                else:
                    # add next predicted token if it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                if len(next_sent_beam) == ensemble_state['num_beams']:
                    # the beam for next step is now full
                    break

            # Check if we're done so that we can save a pad step if all(done)
            ensemble_state['done'][batch_idx] = ensemble_state['done'][batch_idx] or ensemble_state['generated_hyps'][
                batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=ensemble_state['cur_len']
            )

            # update next beam content
            assert len(next_sent_beam) == ensemble_state['num_beams'], "Beam should always be full after loop above"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == ensemble_state['num_beams'] * (batch_idx + 1)

        # stop if are done with every sentence
        if all(ensemble_state['done']):
            return None

        # sanity check / prepare next timestep
        assert len(next_batch_beam) == ensemble_state['batch_size'] * ensemble_state['num_beams']

        ensemble_state['beam_scores'] = ensemble_state['beam_scores'].new([x[0] for x in next_batch_beam])

        beam_tokens = ensemble_state['input_ids'].new([x[1] for x in next_batch_beam])
        # this idx will be used to select the beams sequences to continue -- note the same sequence can be selected
        # and continued in multiple ways
        beam_idx = ensemble_state['input_ids'].new([x[2] for x in next_batch_beam])

        return beam_tokens, beam_idx

    def set_state_decoder_input_ids_to_beam_token_ids(self, beam_tokens: torch.Tensor, beam_idx: torch.Tensor):
        ensemble_state = self.ensemble_state
        for state in self.component_states:
            state['input_ids'] = ensemble_state['input_ids'][beam_idx, :]
            state['input_ids'] = torch.cat([state['input_ids'], beam_tokens.unsqueeze(1)], dim=-1)

            # reorder input_ids according to beam_idx
        ensemble_state['input_ids'] = ensemble_state['input_ids'][beam_idx, :]
        # concat current timestep onto input_ids
        ensemble_state['input_ids'] = torch.cat([ensemble_state['input_ids'], beam_tokens.unsqueeze(1)], dim=-1)

    def update_ensemble_state_decoding_stats(self, beam_tokens: torch.Tensor, beam_idx: torch.Tensor):
        ensemble_state = self.ensemble_state
        for state_idx, component_state in enumerate(self.component_states):
            state_scores = component_state['scores'][beam_idx, beam_tokens]
            max_prob_per_beam = component_state['max_prob'][beam_idx]
            entropy_per_beam = component_state['entropy'][beam_idx]
            # reorder/replace existing state metadata
            next_decoding_stats = []
            for beam_id in beam_idx.cpu().numpy():
                next_decoding_stats.append(copy.deepcopy(ensemble_state['decoding_stats'][state_idx][beam_id]))

            # concat new state metadata horizontally
            state_metadata = [
                {'token': token.item(), 'score': score.item(), 'max_prob_entropy': max_prob_entropy.item(),
                 'entropy': entropy.item()} for token, score, max_prob_entropy, entropy in
                zip(beam_tokens, state_scores, max_prob_per_beam, entropy_per_beam)]
            for beam_id in range(ensemble_state['num_beams']):
                next_decoding_stats[beam_id].append(state_metadata[beam_id])

            ensemble_state['decoding_stats'][state_idx] = next_decoding_stats

    def reorder_internal_states(self, beam_idx: torch.Tensor):
        # re-order internal states Note ensemble_state has no "past", this is only on component_states
        for state in self.component_states:
            state['past'] = (state['past'][0], state['model']._reorder_cache(state['past'][1], beam_idx))

    @staticmethod
    def expand_encoder_outputs(encoder_outputs: BaseModelOutput, expanded_batch_idxs: torch.Tensor) -> BaseModelOutput:
        encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state.index_select(0, expanded_batch_idxs)
        return encoder_outputs

    def print_output_per_step(self):
        for o1_ids in self.ensemble_state['input_ids']:
            o1_text = self.tokenizer.decode(o1_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            print(f'modified_text: {o1_text}')

