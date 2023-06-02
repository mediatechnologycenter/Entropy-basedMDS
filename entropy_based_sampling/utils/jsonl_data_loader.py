#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

import json
from abc import ABC, abstractmethod
from typing import List, Dict

from tqdm import tqdm


class MDSDatasetDataLoader(ABC):
    """
          An abstract class to load multi document summarization datasets.
    """
    @abstractmethod
    def load_data(self, dataset_path: str, max_articles_to_get: int = -1):
        pass

    @abstractmethod
    def get_source_articles(self, max_articles_per_cluster: int = -1) -> List:
        pass


class JsonlDataLoader(MDSDatasetDataLoader):
    """
       A concrete class that implements the abstract MDSDatasetDataLoader interface. It is designed to handle datasets
       that are in the 'jsonl' (JSON Lines text format) format.

       This class has methods to load the dataset from a given path, limit the number of articles to retrieve, convert
       article dictionaries to plain text, and retrieve source articles up to a maximum limit per cluster.
    """
    def __init__(self, dataset_path: str, max_articles_to_get: int):
        self.dataset = self.load_data(dataset_path, max_articles_to_get)

    def load_data(self, dataset_path: str, max_articles_to_get: int = -1):
        dataset = [json.loads(data) for data in open(dataset_path)][:max_articles_to_get]
        return dataset

    @classmethod
    def article_to_text(cls, article: Dict, separator_token: str = ' '):
        # just be sure about whitespace
        title = ' '.join(article["title"].strip().split())
        text = ' '.join(article["text"].strip().split())
        return f'{title} {separator_token} {text}'

    def get_source_articles(self, max_articles_per_cluster: int = -1) -> List:
        source_articles_list = []
        for cluster in tqdm(self.dataset):
            # shuffle articles before selecting topk to use in ensemble
            articles = [self.article_to_text(a) for a in cluster['articles']]
            articles = articles[:max_articles_per_cluster]
            source_articles_list.append(articles)
        return source_articles_list
