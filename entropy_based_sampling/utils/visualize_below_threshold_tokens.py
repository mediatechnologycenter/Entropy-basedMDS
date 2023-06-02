#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

"""
This script provides utilities to analyze and manipulate meta-data related to a given model's output, particularly the
decoding features associated with certain tokens.

The functions allow for extraction of maximum consecutive tokens below a specified threshold and highlighting of tokens
below the threshold from a collection of decoding meta-data.

Note:
    Specify the following variables:
    - model_name = "Name or path to sds model"
    - root_dir = "Path to the root directory containing the predictions (The predictions should contain one summary per
    line and the file should have a .pred ending)"
    - clusters_to_visualize = "Ids of the clusters to visualize from the mds dataset"
"""
import os.path
from typing import Dict
from entropy_based_sampling.utils.decoding_meta_data_class import DecodingMetaData, DecodingFeatures
from entropy_based_sampling.utils.misc import get_file_paths_of_type
from entropy_based_sampling.utils.misc import get_reduce_method_from_file_path


def get_max_consecutive_below_threshold_tokens(reduce_methods_decoding_meta_data: Dict, cluster_id: int,
                                               decoding_feature: DecodingFeatures):
    for reduce_method, decoding_meta_data in reduce_methods_decoding_meta_data.items():
        return decoding_meta_data.get_max_consecutive_tokens_below_threshold(
            cluster_id, decoding_feature)


def highlight_below_threshold_tokens(reduce_methods_decoding_meta_data: Dict, cluster_id: int,
                                     decoding_feature: DecodingFeatures, threshold: float):
    for reduce_method, decoding_meta_data in reduce_methods_decoding_meta_data.items():
        return decoding_meta_data.get_highlighted_below_threshold_tokens(
            cluster_id, decoding_feature, threshold)


def main():
    model_name = ""
    root_dir = ""
    meta_data_root_dir = os.path.join(root_dir, "meta_data")
    normalize = False

    decoding_meta_data_paths = get_file_paths_of_type(meta_data_root_dir, 'jsonl')
    reduce_methods_decoding_meta_data = {}
    for decoding_meta_data_path in decoding_meta_data_paths:
        reduce_method = get_reduce_method_from_file_path(decoding_meta_data_path)
        reduce_methods_decoding_meta_data[reduce_method] = DecodingMetaData(model_name, reduce_method,
                                                                            decoding_meta_data_path,
                                                                            normalize=normalize)

    threshold = 0.35
    clusters_to_visualize = []
    highlighted_texts = []
    for cluster in clusters_to_visualize:
        highlighted_text = highlight_below_threshold_tokens(reduce_methods_decoding_meta_data, cluster,
                                                            DecodingFeatures.MAX_PROB_ENTROPY,
                                                            threshold)
        highlighted_texts.append(highlighted_text)
        print(highlighted_text)


if __name__ == '__main__':
    main()
