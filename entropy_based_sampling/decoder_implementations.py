#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

from typing import Dict

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from entropy_based_sampling.decoder_base import BaseDynamicEnsembleDecoder


class BartDynamicEnsembleDecoder(BaseDynamicEnsembleDecoder):
    """
    A dynamic ensemble decoder class for the BART and MBART models.

    Inherits from the BaseDynamicEnsembleDecoder, implementing methods specific
    to BART and MBART models. It is responsible for computing encoder outputs, getting
    encoder outputs batch size, and running forward pass using a state.

    Methods:
        __init__: Initialize the BartDynamicEnsembleDecoder object.
        compute_encoder_outputs: Compute the encoder outputs for the given input_ids and attention_mask.
        get_encoder_outputs_batch_size: Get the batch size of the encoder outputs.
        outputs_from_state: Run forward pass using a state, specifically for states with a 'model' attribute.
    """

    def __init__(self, model: AutoModelForSeq2SeqLM, tokenizer: AutoTokenizer, decoding_hyperparams: Dict,
                 score_reduce_type: str, **kwargs):

        super().__init__(model, tokenizer, decoding_hyperparams, score_reduce_type, **kwargs)

    def compute_encoder_outputs(self, input_ids: torch.Tensor,
                                attention_mask: torch.Tensor) -> BaseModelOutput:

        encoder_outputs = self.encoder(input_ids, attention_mask=attention_mask, return_dict=True)

        return encoder_outputs

    def get_encoder_outputs_batch_size(self, encoder_outputs: BaseModelOutput) -> int:
        return encoder_outputs.last_hidden_state.shape[0]

    def outputs_from_state(self, state: Dict) -> Seq2SeqLMOutput:
        if len(state['past']) == 1:
            encoder_outputs, past_key_values = state['past'], None
        else:
            encoder_outputs = state['past'][0]
            past_key_values = state['past'][1]

        model_inputs = state['model'].prepare_inputs_for_generation(
            state['input_ids'],
            past=past_key_values,
            attention_mask=state['attention_mask'],
            use_cache=True,
            encoder_outputs=encoder_outputs
        )

        outputs = state['model'](**model_inputs)
        return outputs
