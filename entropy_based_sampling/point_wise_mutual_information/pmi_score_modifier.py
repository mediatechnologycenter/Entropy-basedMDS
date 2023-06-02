#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

from typing import List, Dict

import torch

from entropy_based_sampling.point_wise_mutual_information.language_models import LanguageModel


class PointWiseMutualInformationScoreModifier:
    """
       A class for modifying the log probabilities of generated text based on Point-wise Mutual Information (PMI)
       with a pre-trained language model. This class follows the approach described in
       https://arxiv.org/pdf/2210.13210.pdf, but applies it for german, instead of english.
    """
    def __init__(self, language_model_name: str, pmi_lambda: float,
                 pmi_threshold, pmi_is_conditional: bool, pmi_use_log_prob: bool = True, lm_print_best_prediction: bool = False):
        self.language_model = LanguageModel.get_language_model(language_model_name)
        self.pmi_lambda = pmi_lambda
        self.pmi_threshold = pmi_threshold
        self.log_pmi_threshold = torch.log(torch.Tensor([self.pmi_threshold])).item()
        self.pmi_is_conditional = pmi_is_conditional
        self.pmi_use_log_prob = pmi_use_log_prob
        self.lm_print_best_prediction = lm_print_best_prediction

    def apply_point_wise_mutual_information(self, component_states: List[Dict]) -> List[
        Dict]:
        for state in component_states:
            self.apply_point_wise_mutual_information_per_state(state)
        return component_states

    def apply_point_wise_mutual_information_to_ensemble_state(self, ensemble_state: Dict) -> Dict:
        num_beams, input_ids_len = ensemble_state["input_ids"].size()
        if input_ids_len < 4:
            return ensemble_state
        for beam_id in range(num_beams):
            if self.pmi_is_conditional:
                max_log_score = torch.max(ensemble_state['scores'][beam_id], dim=-1).values.item()
                if max_log_score < self.log_pmi_threshold:
                    self.apply_lm_log_prob_to_state_log_prob(ensemble_state, beam_id)
            else:
                self.apply_lm_log_prob_to_state_log_prob(ensemble_state, beam_id)
        return ensemble_state

    def apply_point_wise_mutual_information_per_state(self, state: Dict) -> Dict:
        num_beams, input_ids_len = state["input_ids"].size()
        if input_ids_len < 4:
            return state
        for beam_id in range(num_beams):
            if self.pmi_is_conditional:
                if state['max_prob'][beam_id] < self.pmi_threshold:
                    self.apply_lm_log_prob_to_state_log_prob(state, beam_id)
            else:
                self.apply_lm_log_prob_to_state_log_prob(state, beam_id)
        return state

    def apply_lm_log_prob_to_state_log_prob(self, state, beam_id):
        lm_output = self.language_model.get_next_token_prob(state["input_ids"][beam_id], log_prob=self.pmi_use_log_prob,
                                                            print_best_prediction=self.lm_print_best_prediction)
        beam_scores = state["scores"][beam_id]
        lm_term = self.pmi_lambda * lm_output
        if not self.pmi_use_log_prob:
            beam_scores = torch.exp(beam_scores)
        beam_scores = beam_scores - lm_term
        state["scores"][beam_id] = beam_scores

