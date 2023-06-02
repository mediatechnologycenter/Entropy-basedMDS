#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

"""
This script allows to evaluate the dynamic ensemble approach for a given multi document test set in the jsonl format.

Functionality includes:

    Initialization of a pretrained or fine-tuned mBART model.
    Configuration and application of a dynamic ensemble decoder to summarize a cluster of articles.
    Evaluation of summarization performance using ROUGE metrics.
    Optional integration with Weights & Biases (wandb) for experiment tracking and logging.

This script expects a JSON Lines (.jsonl) file as an input dataset, where each line represents a cluster of articles,
and each cluster includes the articles and the corresponding gold summary.

To run the script, first adapt the config file(entropy_based_sampling/configs/evaluation/example_eval_config.json) and execute:
    - python3 evaluate.py --config "entropy_based_sampling/configs/evaluation/example_eval_config.json"
"""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import evaluate
import randomname
import tqdm

import numpy as np
import torch

from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM)

from entropy_based_sampling.decoder_implementations import \
    BartDynamicEnsembleDecoder

import wandb

from entropy_based_sampling.point_wise_mutual_information.pmi_score_modifier import PointWiseMutualInformationScoreModifier


def initialize_model(eval_config: Dict):
    """
       Ensemble entropy_based_sampling of a cluster of articles
       """
    model = eval_config['model']
    tokenizer = eval_config['tokenizer']
    decoding_hyperparams = {
        'max_length': eval_config['max_src_length'],
        'min_length': eval_config['min_length'],
        'max_tgt_length': eval_config['max_tgt_length'],
        'num_beams': eval_config['num_beams'],
        'temperature': eval_config['temperature']
    }

    lm_key_name = "pmi_language_model_name"
    pmi_score_modifier = None
    if lm_key_name in eval_config.keys():
        pmi_score_modifier = PointWiseMutualInformationScoreModifier(eval_config[lm_key_name],
                                                                     eval_config['pmi_lambda'],
                                                                     eval_config['pmi_threshold'],
                                                                     eval_config["pmi_is_conditional"],
                                                                     eval_config["pmi_use_log_prob"],
                                                                     lm_print_best_prediction=eval_config[
                                                                         "print_per_step_output"])

    bart_dynamic_ensemble_decoder = BartDynamicEnsembleDecoder(model, tokenizer, decoding_hyperparams,
                                                               eval_config["score_reduce_strategy"],
                                                               print_per_step_output=eval_config[
                                                                   "print_per_step_output"],
                                                               pmi_score_modifier=pmi_score_modifier)
    return bart_dynamic_ensemble_decoder


def summarize_articles(bart_dynamic_ensemble_decoder: BartDynamicEnsembleDecoder, articles: List[str]):
    ensemble_state = \
        bart_dynamic_ensemble_decoder.compute_dynamic_ensemble_decoding(articles)

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(ensemble_state['batch_size']):
        if ensemble_state['done'][batch_idx]:
            continue

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(ensemble_state['num_beams']):
            effective_beam_id = batch_idx * ensemble_state['num_beams'] + beam_id
            final_score = ensemble_state['beam_scores'][effective_beam_id].item()
            final_tokens = ensemble_state['input_ids'][effective_beam_id]

            hyp_metadata = []
            for state_idx in range(len(ensemble_state['decoding_stats'])):
                hyp_metadata.append(ensemble_state['decoding_stats'][state_idx][effective_beam_id])

            ensemble_state['generated_hyps'][batch_idx].add(final_tokens, final_score, metadata=hyp_metadata)

    assert ensemble_state['batch_size'] == 1, 'current logic assumes batch size = 1'

    # sort hyps by score (0 index is first batch, and we're assuming batch_size always = 1 right now)
    sorted_hyps = [(hyp, score, metadata) for score, hyp, metadata in
                   sorted(ensemble_state['generated_hyps'][0].beams, key=lambda b: b[0], reverse=True)]

    print(f'Num hyps in BeamHypotheses: {len(sorted_hyps)}')

    # map token indexes back to strings
    predictions = [bart_dynamic_ensemble_decoder.tokenizer.decode(hyp,
                                                                  skip_special_tokens=True,
                                                                  clean_up_tokenization_spaces=False)
                   for hyp, _, _ in sorted_hyps]

    return predictions, sorted_hyps


def article_to_text(article: Dict, separator_token: str = ' '):
    # just be sure about whitespace
    title = ' '.join(article["title"].strip().split())
    text = ' '.join(article["text"].strip().split())
    return f'{title} {separator_token} {text}'


def main(eval_config: Dict):
    np.random.seed(42)

    if eval_config['evaluation_dataset'].endswith('.jsonl'):
        start_row = eval_config['start_row_to_eval']
        dataset = [json.loads(eval_data) for eval_data in open(eval_config['evaluation_dataset'])]
        num_samples = len(dataset)
        if eval_config['rows_to_eval'] != -1:
            end_row = eval_config['start_row_to_eval'] + eval_config['rows_to_eval']
            if start_row >= num_samples - 1:
                start_row = num_samples - 1
                end_row = num_samples
                dataset = dataset[start_row:end_row]
            else:
                if end_row > num_samples:
                    print(f"End row {end_row} index out of bounds. Limiting end row to {num_samples}")
                    end_row = num_samples
                dataset = dataset[start_row:end_row]
    else:
        raise AssertionError('Right now we only know how to handle .jsonl evaluation dataset')

    eval_prefix = eval_config['eval_prefix']
    test_set_name = Path(eval_config['evaluation_dataset']).stem
    model_name = eval_config["model_id"].split("/")[-1]
    reduce_strategy = eval_config["score_reduce_strategy"]
    temperature = eval_config["temperature"]
    config_dict = {'test_set': test_set_name,
                   'model': model_name,
                   'target_summary_length': eval_config[
                       'max_tgt_length'],
                   'max_cluster_size': eval_config[
                       'max_articles_in_cluster'],
                   'score_reduce_function': reduce_strategy,
                   'temperature': temperature
                   }

    lm_key_name = "pmi_language_model_name"
    if lm_key_name in eval_config.keys():
        for key in eval_config.keys():
            if "pmi" in key:
                config_dict[key] = eval_config[key]

    print()
    print(
        f'------------ DYNE EVAL - Model: {model_name} - Test Set: {test_set_name} - Target Length: {eval_config["max_tgt_length"]} - Cluster '
        f'Size: {eval_config["max_articles_in_cluster"]} - Reduce Function: {reduce_strategy} '
        f'-----------------')

    if eval_config["wandb_on"]:
        wandb_mode = "online"
    else:
        wandb_mode = "disabled"
    wandb_run_name = randomname.get_name() + '_' + '_'.join(
        [str(config_value) for config_value in config_dict.values()])
    wandb_run_group_name = f"dyne_eval_{test_set_name}_group"
    wandb.init(project="dyne_evaluations", entity="background-tool", config=config_dict, name=wandb_run_name,
               mode=wandb_mode, group=wandb_run_group_name)
    wb_summary_table = wandb.Table(columns=["prediction", "reference"])

    if 'predictions' not in eval_config:
        # load pretrained or finetuned transformer model
        print(f'loading pre-trained model: {eval_config["model_id"]}')

        # we have to load fine-tuned models in a different way because of pytorch-lightning
        if eval_config['model_id'].endswith('.ckpt'):
            print("Only huggingface models are currently supported")
            return
        else:
            # transformers pretrained
            model_id = eval_config['model_id']
            eval_config['tokenizer'] = AutoTokenizer.from_pretrained(model_id)
            bos_token_id = 0
            if eval_config["language"] == "german" and "mbart" in model_name:
                bos_token_id = eval_config['tokenizer'].lang_code_to_id["de_DE"]
            eval_config['model'] = AutoModelForSeq2SeqLM.from_pretrained(model_id,
                                                                         bos_token_id=bos_token_id,
                                                                         trust_remote_code=True)

        # Set the model in evaluation mode to deactivate the DropOut modules
        eval_config['model'].eval()

        if torch.cuda.is_available():
            eval_config['model'].to('cuda')

        with torch.no_grad():
            # summarize MDS / entropy_based_sampling dataset with model
            dataset_name = Path(eval_config["evaluation_dataset"]).stem.lower().replace(".", "_")
            if eval_config["log_output"]:
                eval_file_name = eval_prefix + dataset_name + "/" + model_name + "_" + reduce_strategy + "_" + str(
                    eval_config["max_tgt_length"]) + "_" + str(
                    eval_config["max_articles_in_cluster"]) + "_" + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                os.makedirs(os.path.dirname(eval_file_name), exist_ok=True)
                preds_output = open(f'{eval_file_name}_predicted_summaries.out', 'w', buffering=1)
                gold_output = open(f'{eval_file_name}_gold_summaries.out', 'w', buffering=1)
                metadata_file_name = f'{eval_file_name}_decoding_metadata.jsonl'
                metadata_output = open(metadata_file_name, 'w', buffering=1)

            summaries = []
            bart_dynamic_ensemble_decoder = initialize_model(eval_config=eval_config)
            # get summary for each cluster
            # note here we have a macro-batch size of one cluster by definition
            for cluster in tqdm.tqdm(dataset):
                # shuffle articles before selecting topk to use in ensemble
                articles = [article_to_text(a) for a in cluster['articles']]
                np.random.shuffle(articles)
                articles = articles[:eval_config['max_articles_in_cluster']]
                articles = [article.strip() for article in articles if article.strip() != ""]

                if 'min_input_char_length' in eval_config:
                    articles_ = [a for a in articles if len(a) >= eval_config['min_input_char_length']]
                    if len(articles_) == 0:
                        articles_ = [articles[0]]
                    articles = articles_

                gold_summary = " ".join(summary_line.strip() for summary_line in
                                        cluster['summary'].splitlines())  # cluster['summary'].strip()

                predictions, sorted_hyps = summarize_articles(bart_dynamic_ensemble_decoder, articles)
                # sorted_hyps -- (token_idxs, score, metadata)
                # they're in sorted order according to ensemble score, so first one is the best
                # we will have one list of timestamp metadata for each cluster input
                length_penalty = eval_config['length_penalty']
                component_scores = []
                for input_idx, state_metadata in enumerate(sorted_hyps[0][2]):
                    timestep_scores = np.array([o['score'] for o in state_metadata])
                    global_score = np.sum(timestep_scores) / len(timestep_scores) ** length_penalty
                    component_scores.append(global_score)

                component_scores = np.array(component_scores)
                for idx in np.argsort(component_scores)[::-1]:
                    print(f'ARTICLE: {articles[idx][:1500]}')
                    print(f'Input {idx} score: {component_scores[idx]}')
                    print()

                print(f'Ensemble score: {sorted_hyps[0][1]}')
                print(f'Gold: {cluster["summary"]}')
                print(f'Predicted: {predictions[0]}')
                print()

                predicted_summary = predictions[0]

                summaries.append((predicted_summary, gold_summary))

                wb_summary_table.add_data(predicted_summary, gold_summary)

                if eval_config["log_output"]:
                    preds_output.write(f'{predicted_summary}\n')
                    gold_output.write(f'{gold_summary}\n')

                    sorted_hyps_ = []
                    for tok_idxs, score, tok_scores in sorted_hyps:
                        tok_idxs = [int(idx) for idx in tok_idxs.cpu().numpy()]
                        sorted_hyps_.append((tok_idxs, score, tok_scores))
                    sorted_hyps = sorted_hyps_

                    metadata_output.write(
                        json.dumps(
                            {
                                'cluster': cluster,
                                'predictions': predictions,
                                'inputs_used': articles,
                                'component_scores': list(component_scores),
                                'decoding_metadata': sorted_hyps
                            })
                        + '\n')
            if eval_config["log_output"]:
                preds_output.close()
                gold_output.close()
                metadata_output.close()

            # Evaluation
            hyps, refs = zip(*summaries)
    else:
        # Evaluate on user-supplied predictions
        print(f'Evaluating predictions in {eval_config["predictions"]} '
              f'against gold summaries in {eval_config["evaluation_dataset"]}')
        hyps = [predictions.strip() for predictions in open(eval_config['predictions'])]
        # Note this is only single-reference currently
        refs = [json.loads(c)['summary'].strip() for c in open(eval_config['evaluation_dataset'])]
        assert len(hyps) == len(refs)

    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=hyps, references=refs)
    print(results)
    wandb.log(results)
    wandb.log({"prediction_vs_reference_summaries": wb_summary_table})
    if eval_config["log_output"]:
        wandb.save(metadata_file_name)

    # End evaluation


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to evaluation config file'
    )

    parser.add_argument(
        '--max_tgt_length',
        type=int,
        required=False,
        help='Maximum length of the generated summary. If set, overwrites the value in the config fle'
    )

    parser.add_argument(
        '--max_articles_in_cluster',
        type=int,
        required=False,
        help='Maximum articles in a cluster of source documents. If set, overwrites the value in the config fle'
    )

    parser.add_argument(
        '--score_reduce_strategy',
        type=str,
        required=False,
        help='Which reduce strategy should be used during dynamic ensemble deocoding. If set, overwrites the value in '
             'the config fle '
    )

    args = parser.parse_args()

    if not args.config:
        print("No configuration file provided")
        exit()

    with open(args.config, "r") as config:
        eval_config = json.load(config)
        if args.max_tgt_length:
            eval_config['max_tgt_length'] = args.max_tgt_length

        if args.max_articles_in_cluster:
            eval_config['max_articles_in_cluster'] = args.max_articles_in_cluster

        if args.score_reduce_strategy:
            eval_config['score_reduce_strategy'] = args.score_reduce_strategy

        temperature_key = 'temperature'
        if temperature_key not in eval_config.keys():
            eval_config[temperature_key] = 1
        return eval_config


if __name__ == '__main__':
    main(parse_args())
