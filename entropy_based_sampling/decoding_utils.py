#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

"""
This module provides utility functions and classes for beam search when using dynamic ensemble decoding. It includes
BeamHypotheses class, create_attention_mask function, expand_input_dim_to_num_beams function, and other utility
functions. The module also contains an implementation of ScoreReduceStrategy abstract class and its subclasses for
different score reducing strategies, which determine in DynE how the outputs of the models are combined.

"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import torch


class BeamHypotheses(object):
    def __init__(self, num_beams: int, max_length: int, length_penalty: int, early_stopping: bool):
        """
        A class for maintaining n-best list of hypotheses during beam search.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self) -> int:
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp: torch.Tensor, sum_logprobs: float, metadata: Optional[List] = None):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp, metadata))
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx) for idx, (s, _, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs: float, cur_len: Optional[int] = None) -> bool:
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            if cur_len is None:
                cur_len = self.max_length
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret


def create_attention_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    if (pad_token_id is not None) and (pad_token_id in input_ids):
        attention_mask = input_ids.ne(pad_token_id).long()
    else:
        attention_mask = input_ids.new_ones(input_ids.shape)

    return attention_mask


def expand_input_dim_to_num_beams(input_tensor: torch.Tensor, input_ids_len: int, num_beams: int, batch_size: int,
                                  effective_batch_mult: int,
                                  effective_batch_size: int) -> torch.Tensor:
    expanded_input = input_tensor.unsqueeze(1).expand(batch_size, effective_batch_mult * num_beams, input_ids_len)

    expanded_input = expanded_input.contiguous().view(
        effective_batch_size * num_beams, input_ids_len
    )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

    return expanded_input


def apply_heuristics_to_logits(state: Dict) -> Dict:
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if state['repetition_penalty'] != 1.0:
        enforce_repetition_penalty(
            state['next_token_logits'],
            state['batch_size'],
            state['num_beams'],
            state['input_ids'],
            state['repetition_penalty']
        )

    if state['temperature'] != 1.0:
        state['next_token_logits'] = state['next_token_logits'] / state['temperature']

    return state


def calc_banned_ngram_tokens(prev_input_ids: torch.Tensor, num_hypos: int, no_repeat_ngram_size: int,
                             cur_len: int) -> List:
    # Copied from fairseq for no_repeat_ngram in beam_search
    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def calc_banned_bad_words_ids(prev_input_ids: torch.Tensor, bad_words_ids: List) -> List:
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens):] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def enforce_repetition_penalty(lprobs: torch.Tensor, batch_size: int, num_beams: int, prev_output_tokens: torch.Tensor,
                               repetition_penalty: float):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


class ScoreReduceStrategy(ABC):
    subclasses = {}

    @abstractmethod
    def reduce_score(self, component_states: List[Dict]) -> torch.Tensor:
        pass

    @classmethod
    def register_subclass(cls, score_reduce_type):
        def decorator(subclass):
            cls.subclasses[score_reduce_type] = subclass
            return subclass

        return decorator

    @classmethod
    def get_score_reduce_strategy(cls, score_reduce_type):
        if score_reduce_type not in cls.subclasses:
            raise ValueError('Bad score reduce type {}'.format(score_reduce_type))

        return cls.subclasses[score_reduce_type]()


@ScoreReduceStrategy.register_subclass("Single_document_sum")
class SingleDocumentScoreReduceStrategy(ScoreReduceStrategy):
    def reduce_score(self, component_states: List[Dict]) -> torch.Tensor:
        num_articles = len(component_states)
        assert num_articles == 1
        return component_states[0]['scores']


@ScoreReduceStrategy.register_subclass("DynE")
class AverageScoreReduceStrategy(ScoreReduceStrategy):
    def reduce_score(self, component_states: List[Dict]) -> torch.Tensor:
        stacked_scores = torch.stack([s['scores'] for s in component_states])
        return torch.mean(stacked_scores, dim=0)


@ScoreReduceStrategy.register_subclass("product")
class ProductScoreReduceStrategy(ScoreReduceStrategy):
    def reduce_score(self, component_states: List[Dict]) -> torch.Tensor:
        stacked_scores = torch.stack([s['scores'] for s in component_states])
        return -torch.prod(torch.abs(stacked_scores), dim=0)


@ScoreReduceStrategy.register_subclass("max")
class MaxScoreReduceStrategy(ScoreReduceStrategy):
    def reduce_score(self, component_states: List[Dict]) -> torch.Tensor:
        stacked_scores = torch.stack([s['scores'] for s in component_states])
        return torch.max(stacked_scores, dim=0)[0]


@ScoreReduceStrategy.register_subclass("H_min")
class MinEntropyScoreReduceStrategy(ScoreReduceStrategy):
    def reduce_score(self, component_states: List[Dict]) -> torch.Tensor:
        stacked_entropy = torch.stack([s['entropy'] for s in component_states])
        _, min_entropy_per_beam_indices = torch.min(stacked_entropy, dim=0)
        min_entropy_scores_per_beam = torch.stack(
            [component_states[article_id]['scores'][beam_id, :] for beam_id, article_id in
             enumerate(min_entropy_per_beam_indices)])
        return min_entropy_scores_per_beam


@ScoreReduceStrategy.register_subclass("H_th")
class ThresholdMaxProbabilityScoreReduceStrategy(ScoreReduceStrategy):
    """
    This strategy averages the scores for articles where the max_prob is below the threshold(0.35) and selects the
    article with maximum prob otherwise.
    """
    max_prob_threshold = 0.35

    def get_max_prob_scores(self, component_states: List[Dict]) -> torch.Tensor:
        stacked_max_prob = torch.stack([s['max_prob'] for s in component_states])
        _, max_prob_per_beam_indices = torch.max(stacked_max_prob, dim=0)
        max_prob_scores_per_beam = torch.stack(
            [component_states[article_id]['scores'][beam_id, :] for beam_id, article_id in
             enumerate(max_prob_per_beam_indices)])
        return max_prob_scores_per_beam

    def get_mean_prob_scores_for_beam(self, component_states: List[Dict], beam_id: int) -> torch.Tensor:
        mean_prob_scores_for_beam = torch.stack([
            component_state['scores'][beam_id, :] for component_state in component_states])

        mean_prob_scores_for_beam = torch.mean(mean_prob_scores_for_beam, dim=0)
        return mean_prob_scores_for_beam

    def reduce_score(self, component_states: List[Dict]) -> torch.Tensor:
        num_articles = len(component_states)
        assert num_articles > 0
        if num_articles == 1:
            return component_states[0]['scores']

        num_beams = component_states[0]['num_beams']
        vocab_size = component_states[0]['vocab_size']
        device = component_states[0]['scores'].device
        reduced_scores = torch.zeros((num_beams, vocab_size), device=device)

        prob_above_threshold = torch.stack(
            [s['max_prob'] > self.max_prob_threshold for s in component_states])

        max_prob_scores_per_beam = self.get_max_prob_scores(component_states)

        for beam_id in range(num_beams):
            prob_above_threshold_indices_for_beam = prob_above_threshold[:, beam_id].nonzero()
            if len(prob_above_threshold_indices_for_beam) == 0:
                # set scores equal to the mean of input article scores if all max_probs < threshold
                reduced_scores[beam_id, :] = self.get_mean_prob_scores_for_beam(component_states, beam_id)
            else:
                # set scores equal to the input article score with the highest max prob
                reduced_scores[beam_id, :] = max_prob_scores_per_beam[beam_id, :]

        return reduced_scores
