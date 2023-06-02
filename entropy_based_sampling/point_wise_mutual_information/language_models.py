#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

"""
Note:
    Specify the following in order to use pmi:
    - Line 57: Specify the Name or Path to the Mbart Language model  inside @LanguageModel.register_subclass("...")
    - Line 95: Specify the Name or Path to the GPT-2 Language model inside @LanguageModel.register_subclass("...")
"""
from abc import abstractmethod, ABC

import torch
from torch.nn import functional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM


class LanguageModel(ABC):
    """
        An abstract base class representing a language model. It serves as a blueprint for language model subclasses
        that provide specific implementations.

        Attributes:
            subclasses (dict): A dictionary to store registered subclasses.

        Methods:
            prepare_input_ids(input_ids: torch.Tensor) -> torch.Tensor:
                Abstract method to prepare input_ids for a specific language model implementation.
            get_next_token_prob(input_ids: torch.Tensor, log_prob: bool = True) -> torch.Tensor:
                Abstract method to get the probabilities of the next token given input_ids.
            register_subclass(score_reduce_type):
                Decorator method to register a subclass in the subclasses dictionary.
            get_language_model(language_model_name):
                Retrieve the appropriate language model subclass based on the language_model_name.
        """
    subclasses = {}

    @abstractmethod
    def prepare_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def get_next_token_prob(self, input_ids: torch.Tensor, log_prob: bool = True) -> torch.Tensor:
        pass

    @classmethod
    def register_subclass(cls, score_reduce_type):
        def decorator(subclass):
            cls.subclasses[score_reduce_type] = subclass
            return subclass

        return decorator

    @classmethod
    def get_language_model(cls, language_model_name):
        if language_model_name not in cls.subclasses:
            raise ValueError('No language model class exists for {}'.format(language_model_name))

        return cls.subclasses[language_model_name](language_model_name)


@LanguageModel.register_subclass("")
class MBartLanguageModel(LanguageModel):
    """
       A subclass of LanguageModel for the MBart model. Implements the methods required for this specific language model.
    """

    def __init__(self, language_model_name: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name, src_lang="de_DE", tgt_lang="de_DE")
        bos_token_id = self.tokenizer.lang_code_to_id["de_DE"]
        self.model = AutoModelForSeq2SeqLM.from_pretrained(language_model_name, bos_token_id=bos_token_id).to(
            self.device)
        self.model.eval()
        self.prepend_tensor = torch.Tensor([self.tokenizer.bos_token_id]).to(device=self.device, dtype=torch.int)
        self.append_tensor = torch.Tensor([self.tokenizer.mask_token_id,
                                           self.tokenizer.eos_token_id, self.tokenizer.cur_lang_code]).to(
            device=self.device, dtype=torch.int)

    def prepare_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.squeeze()
        new_input_ids = torch.cat((self.prepend_tensor, input_ids[2:-1], self.append_tensor), dim=0).unsqueeze(
            dim=0)
        return new_input_ids

    def get_next_token_prob(self, input_ids: torch.Tensor, log_prob: bool = True) -> torch.Tensor:
        new_input_ids = self.prepare_input_ids(input_ids)
        output_logits = self.model(new_input_ids).logits

        masked_index = (input_ids[0] == self.tokenizer.mask_token_id).nonzero().item()
        if log_prob:
            log_probs = functional.log_softmax(output_logits[0, masked_index], dim=-1)
            return log_probs

        probs = functional.softmax(output_logits[0, masked_index], dim=-1)
        return probs


@LanguageModel.register_subclass("")
class Gpt2LanguageModel(LanguageModel):
    """
     A subclass of LanguageModel for the GPT-2 model. Implements the methods required for this specific language model.
    """

    def __init__(self, language_model_name: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_name, src_lang="de_DE", tgt_lang="de_DE")
        self.model = AutoModelForCausalLM.from_pretrained(language_model_name).to(self.device)
        self.model.eval()

    def prepare_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids

    def get_next_token_prob(self, input_ids: torch.Tensor, log_prob: bool = True,
                            print_best_prediction: bool = False) -> torch.Tensor:
        new_input_ids = self.prepare_input_ids(input_ids)
        output_logits = self.model(new_input_ids).logits

        next_word_logits = output_logits[-1, :]

        if log_prob:
            return_probs = functional.log_softmax(next_word_logits, dim=-1)
        else:
            return_probs = functional.softmax(next_word_logits, dim=-1)

        if print_best_prediction:
            self.print_best_word_predictions(return_probs)
        return return_probs

    def print_best_word_predictions(self, probs: torch.Tensor):
        values, predictions = probs.topk(1)
        print(f'lm - best prediction: {self.tokenizer.decode(predictions).split()}, prob: {values}')
