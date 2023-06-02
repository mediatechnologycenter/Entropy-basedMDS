# Entropy-based Sampling for Abstractive Multi-document Summarization in Low-resource Settings

This repository features a novel approach to multi-document summarization, focusing on German news data. Our approach employs a combination of entropy-based sampling techniques and dynamic ensemble decoding.
The codebase is a fork from [dynamic ensemble decoding](https://github.com/chrishokamp/dynamic-transformer-ensembles)(DynE), to which we have
introduced a number of modifications and enhancements. Below is a summary of our key contributions:

1. **Introduction of Entropy-Based Sampling Approaches:** We've developed and integrated entropy-based sampling approaches to the original DynE codebase. This is primarily encapsulated within four key files:
    - [decoder_base.py](entropy_based_sampling%2Fdecoder_base.py): Contains the base class which implements the main functionality for dynamic ensemble decoding.
    - [decoder_implementations.py](entropy_based_sampling%2Fdecoder_implementations.py): Contains the specific implementations for the decoder base class, depending on the model architecture.
    - [decoding_utils.py](entropy_based_sampling%2Fdecoding_utils.py): This file implements our entropy-based sampling strategies and provides utility functions for decoding.
    - [pmi_score_modifier.py](entropy_based_sampling%2Fpoint_wise_mutual_information%2Fpmi_score_modifier.py): Implements the point-wise mutual information approach.

2. **Compatibility Upgrade:** We've updated the codebase to ensure compatibility with newer versions of Huggingface's transformer libraries (v4.23.1 or later).

For more details, please refer to the paper.

## Installation

1. Clone this repository to your local machine.
2. Install requirements:

```
pip install requirements.txt
```

## Running the Evaluation

1. Adapt the provided configuration file as explained in the Configuration section.
2. From the root directory, execute:

```
python evaluate.py --config "configs/evaluation/example_eval_config.json"
```

### Configuration

- To run the evaluation, modify the parameters in the ```configs/evaluation/example_eval_config.json```.

| Parameter               | Description                                                                                                                                                                 |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| evaluation_dataset      | The path to the mds dataset. The dataset should be in jsonl format with each line being a cluster of source articles and a gold summary.                                    |
| model_id                | Name or path to the sds model.                                                                                                                                              |
| language                | Specify language (default="german").                                                                                                                                        |
| num_beams               | Number of beams for beam search during decoding.                                                                                                                            |
| max_src_length          | Maximum context length that the model can process (default: 1024).                                                                                                          |
| min_length              | Minimum length of the generated summary.                                                                                                                                    |
| max_tgt_length          | Maximum length of the generated summary.                                                                                                                                    |
| max_articles_in_cluster | Maximum number of source articles to consider when applying DynE (default:5).                                                                                               |
| start_row_to_eval       | Row/Cluster in the dataset where the evaluation should start (default:0).                                                                                                   |
| rows_to_eval            | Number of rows/clusters to evaluate (default:-1).                                                                                                                           |
| eval_prefix             | Path where the evaluation outputs should be stored.                                                                                                                         |
| length_penalty          | Length penalty during generation (default:2).                                                                                                                               |
| wandb_on                | Boolean indicating whether to report evaluation results to Weights and Biases (default:false).                                                                              |
| log_output              | Boolean indicating whether to store output summaries, decoding meta-data, and gold summaries to a file specified in eval_prefix (default:false).                            |
| print_per_step_output   | Boolean indicating whether to print the output tokens at each generation step (default:false).                                                                              |
| score_reduce_strategy   | Strategy to use for combining the log probabilities during DynE (default:"average"). See entropy_based_sampling/configs/dyne_reduce_strategy_names.txt for possible values. |
| temperature             | Temperature during generation (default: 1).                                                                                                                                 |

- To apply point-wise mutual information(pmi) approach, add the following additional parameters to the config file:

| Parameter               | Description                                                                                                                       |
|-------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| pmi_language_model_name | Name or path to the language model trained on the casual language modeling task. We fine-tuned a gpt-2 model on german news data. |
| pmi_lambda              | Float value between 0 and 1. Lamdba value of the pmi approach (default:0.25).                                                     |
| pmi_threshold           | Float value between 0 and 1. Threshold for the pmi approach (default:0.35).                                                       |

## Dataset

- For more information, refer to README under ```dataset```.
  
  

