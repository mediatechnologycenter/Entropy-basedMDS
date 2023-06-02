#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

import json
from enum import Enum
from typing import List
import re

import numpy as np
import pandas
from decouple import config
from matplotlib import pyplot as plt
import seaborn as sns
from numpy import ndarray
from transformers import AutoTokenizer
from itertools import groupby


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class DecodingFeatures(Enum):
    SCORE = "score"
    MAX_PROB_ENTROPY = "max_prob_entropy"
    ENTROPY = "entropy"


class DecodingMetaData:
    """
    This class provides an interface for handling, visualizing and extracting information from the metadata associated
    with the decoding phase of a language model.

    :param model_name: The name of the model, used for retrieving the corresponding tokenizer from the Hugging Face model hub.
    :param reduce_method: The reduce method used for the model.
    :param decoding_meta_data_path: The path to the file containing the decoding metadata.
    :param normalize: A boolean indicating whether to normalize the data or not. Default is True.

    Instance Variables:
        - tokenizer: The tokenizer associated with the model.
        - reduce_method: The reduce method used for the model.
        - decoding_meta_data_all_clusters: The loaded decoding metadata for all clusters.
        - input_article_feature_all_clusters_dict: A dictionary mapping each feature to its corresponding value for all clusters.

    The class provides several methods to interact with the metadata such as retrieving source articles, getting
    generated tokens, highlighting tokens below a certain threshold, plotting heatmaps of features, and more.
    """
    def __init__(self, model_name: str, reduce_method: str, decoding_meta_data_path: str, normalize: bool = True):
        HUGGINGFACE_TOKEN = config("HUGGINGFACE_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_auth_token=HUGGINGFACE_TOKEN)
        self.reduce_method = reduce_method
        self.decoding_meta_data_all_clusters = self.load_decoding_meta_data(decoding_meta_data_path)
        self.input_article_feature_all_clusters_dict = {}
        for feature in DecodingFeatures:
            self.input_article_feature_all_clusters_dict[feature] = self.get_input_article_feature_all_clusters(
                str(feature.value), normalize=normalize)

    def get_input_article_feature_all_clusters(self, decoding_feature: str, normalize: bool = True):
        input_article_feature_per_cluster = []
        for decoding_meta_data in self.decoding_meta_data_all_clusters:
            input_article_feature_per_cluster.append(self.get_input_article_feature_one_cluster(
                decoding_meta_data, decoding_feature, normalize))
        return input_article_feature_per_cluster

    def get_cluster_source_articles(self, cluster_id: int):
        cluster_source_articles = {}
        for article_id, article_text in enumerate(self.decoding_meta_data_all_clusters[cluster_id]['inputs_used']):
            cluster_source_articles[article_id] = article_text
        return cluster_source_articles

    def get_sum_score_and_variance_per_article(self, cluster_id: int) -> (ndarray, float):
        sum_score = self.input_article_feature_all_clusters_dict[DecodingFeatures.SCORE][cluster_id].sum(axis=1)
        sum_score_var = np.var(sum_score)
        return sum_score, sum_score_var

    def get_generated_tokens(self, cluster_id: int) -> List[str]:
        hyp_tokens = [re.sub('Ġ', '', self.tokenizer.convert_ids_to_tokens(id))
                      for id in self.decoding_meta_data_all_clusters[cluster_id]['decoding_metadata'][0][0]]
        # cut off first token because input hyps don't use it
        hyp_tokens = hyp_tokens[1:]
        return hyp_tokens

    def get_max_consecutive_tokens_below_threshold(self, cluster_id: int, decoding_feature: DecodingFeatures,
                                                   threshold: float) -> int:
        data = self.input_article_feature_all_clusters_dict[decoding_feature][cluster_id].T
        below_threshold_data = data < threshold
        below_threshold_data = np.logical_and.reduce(below_threshold_data, 1)
        max_consecutive_tokens_below_threshold = 0
        for key, group in groupby(below_threshold_data):
            if not key:
                continue
            num_tokens_below_threshold = len(list(group))
            if max_consecutive_tokens_below_threshold > num_tokens_below_threshold:
                continue
            max_consecutive_tokens_below_threshold = num_tokens_below_threshold
        return max_consecutive_tokens_below_threshold

    def get_highlighted_below_threshold_tokens(self, cluster_id: int, decoding_feature: DecodingFeatures,
                                               threshold: float) -> str:
        data = self.input_article_feature_all_clusters_dict[decoding_feature][cluster_id].T
        below_threshold_data = data < threshold
        below_threshold_data = np.logical_and.reduce(below_threshold_data, 1)
        generated_tokens = self.get_generated_tokens(cluster_id)
        highlighted_summary = ""
        for token_below_threshold, token in zip(below_threshold_data[1:], generated_tokens[1:]):
            if token_below_threshold:
                highlighted_summary += f"{bcolors.OKBLUE}{token}{bcolors.ENDC}"
            else:
                highlighted_summary += token
        highlighted_summary = highlighted_summary.replace('▁', ' ').strip()
        return highlighted_summary

    def plot_input_feature_heatmap_for_cluster(self, cluster_id: int, decoding_feature: DecodingFeatures,
                                               save_fig: bool = False,
                                               plt_save_path: str = None) -> plt:
        data = self.input_article_feature_all_clusters_dict[decoding_feature][cluster_id].T
        generated_tokens = self.get_generated_tokens(cluster_id)
        plt.rcParams['figure.figsize'] = [10, 20]
        hist_plot = sns.heatmap(data < 0.35,
                                yticklabels=generated_tokens,
                                linewidth=0.5,
                                cmap='Blues'
                                )
        plt.tight_layout()
        plt.title(str(cluster_id) + "_" + self.reduce_method + "_" + str(decoding_feature.value))
        plt.show()
        if save_fig:
            plt_name = f"Cluster_{cluster_id}_{self.reduce_method}.png"
            if plt_save_path is None:
                print("Please specify root path for saving the plots!")
                return
            hist_figure = hist_plot.get_figure()
            hist_figure.savefig(plt_save_path + plt_name)

    def plot_multiple_features_for_cluster(self, cluster_id: int, decoding_features: List[DecodingFeatures]):
        data_list = []
        for decoding_feature in decoding_features:
            data_list.append(self.input_article_feature_all_clusters_dict[decoding_feature][cluster_id].T)
        generated_tokens = self.get_generated_tokens(cluster_id)
        test_array = np.empty((200, 2))
        test_array[:, 0] = data_list[0][:, 1]
        test_array[:, 1] = data_list[1][:, 1]
        df = pandas.DataFrame(test_array)
        plt.rcParams['figure.figsize'] = [20, 10]
        corr = df.corr()
        heatmap = sns.heatmap(corr, annot=True, cmap="Blues", fmt='.1g')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def load_decoding_meta_data(decoding_meta_data_path) -> List:
        # rows are inputs
        # columns are timesteps
        # values are input-specific logprobs
        decoding_metadata = [json.loads(l) for l in
                             open(decoding_meta_data_path)]
        return decoding_metadata

    @staticmethod
    def get_input_article_feature_one_cluster(cluster_decoding_meta_data,
                                              decoding_feature: str, normalize: bool = True) -> ndarray:
        input_article_feature_per_cluster = [[t[decoding_feature] for t in input_score]
                                             for input_score in cluster_decoding_meta_data['decoding_metadata'][0][2]]
        input_article_feature_per_cluster = np.vstack(input_article_feature_per_cluster)

        # column norm
        input_article_feature_per_cluster_mean = input_article_feature_per_cluster.mean(axis=0)
        input_article_feature_per_cluster_std = input_article_feature_per_cluster.std(axis=0)
        input_article_feature_per_cluster_norm = input_article_feature_per_cluster - input_article_feature_per_cluster_mean
        if np.all(input_article_feature_per_cluster_norm == 0):
            print("All articles are the same")
            return input_article_feature_per_cluster_norm
        if not normalize:
            return input_article_feature_per_cluster

        input_article_feature_per_cluster_norm /= input_article_feature_per_cluster_std

        return input_article_feature_per_cluster_norm
