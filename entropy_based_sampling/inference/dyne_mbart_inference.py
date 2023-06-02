#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from entropy_based_sampling.decoder_implementations import \
    BartDynamicEnsembleDecoder


def main():
    model_name = ""
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    bos_token_id = tokenizer.lang_code_to_id["de_DE"]
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=model_name, bos_token_id=bos_token_id)

    decoding_hyperparams = {
        'max_length': 1024,
        'num_beams': 3,
        'min_length': 20,
        'max_tgt_length': 200
    }

    score_reduce_strategy_name = "average"
    test_articles = []

    bart_dynamic_ensemble_decoder = BartDynamicEnsembleDecoder(model, tokenizer, decoding_hyperparams,
                                                               score_reduce_strategy_name)
    summary = bart_dynamic_ensemble_decoder.get_summary(
        test_articles)

    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
