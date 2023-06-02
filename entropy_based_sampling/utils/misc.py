#   SPDX-License-Identifier: Apache-2.0
#   © (2023) ETH Zurich and other contributors, see AUTHORS.txt for details

"""
This script contains utility functions for file manipulation and data processing.

The file manipulation utilities include reading and writing to text, JSON, JSONL and pickle files, obtaining paths of
certain file types within a directory, and storing dictionaries to files. The data processing functions involve handling
lists and dictionaries, including operations for extracting non-overlapping elements and logging dictionary contents.

Note: TypeVar "T" is used for type hinting where the specific type will be determined at runtime.
"""

import glob
import json
import logging
import pickle
import re
from typing import Any, List, Dict, TypeVar, OrderedDict


T = TypeVar("T")


def read_lines(file_name: str):
    with open(file_name, 'r', encoding='utf8') as file:
        lines = file.readlines()
    return [line[:-1] for line in lines]


def write_lines(lines: List[str], out_file_path: str):
    with open(out_file_path, 'w') as out_file:
        for line in lines:
            out_file.write(line)
            out_file.write('\n')


def read_json(file_name: str) -> Dict:
    with open(file_name) as file:
        json_data = json.load(file)
        return json_data


def write_jsonl(filename: str, data: List[Dict[str, str]]):
    with open(filename, 'w+') as out:
        for ddict in data:
            jout = json.dumps(ddict, ensure_ascii=False) + '\n'
            out.write(jout)


def read_jsonl(file_name: str, load_ordered_dict: bool = False) -> List[Dict]:
    json_lines_content = []
    object_hook = None
    if load_ordered_dict:
        object_hook = OrderedDict
    with open(file_name, 'r') as file:
        for line in file:
            json_lines_content.append(json.loads(line, object_hook=object_hook))
    return json_lines_content


def write_json(file_name: str, json_content: Dict):
    with open(file_name, 'w') as file:
        json.dump(json_content, file, ensure_ascii=False)


def write_pickle(file_name: str, content: Any):
    with open(file_name, 'wb') as file:
        pickle.dump(content, file, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(file_name: str):
    with open(file_name, 'rb') as file:
        pickle_data = pickle.load(file)
        return pickle_data


def write_lines(lines: List[str], file_name: str, skip_first: bool = False):
    with open(file_name, 'w', encoding='utf8') as file:
        file.write(lines[0].replace("\n", " "))
        if skip_first:
            lines = lines[1:]
        for line in lines[1:]:
            file.write('\n' + line.replace("\n", " "))


def dict_to_file(dictionnary: Dict, file_path: str, as_json: bool = True):
    """ Stores dict to file. """
    with open(file_path, "w") as file:
        if as_json:
            file.write(json.dumps(dictionnary, indent=4, sort_keys=True))
        else:
            for key, value in sorted(dictionnary.items()):
                file.write(f"{key} = {value}\n")


def log_dict(dictionnary: Dict, info: str):
    """ Log content from dict. """
    logging.info(f"***** {info} *****")
    for key, value in sorted(dictionnary.items()):
        logging.info(f"  {key} = {value}")


def get_file_paths_of_type(root_dir: str, file_type: str) -> List[str]:
    path = f'{root_dir}/*.{file_type}'
    file_paths = glob.glob(path)
    return file_paths


def get_distinct_colors(num_of_colors: int) -> List[str]:
    distinct_colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6',
                       '#bcf60c', '#fabebe', '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3',
                       '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000']
    assert num_of_colors < len(distinct_colors)
    return distinct_colors[:num_of_colors]


def get_non_overlap_from_list(list_1: List[T], list_2: List[T]) -> List[T]:
    return [x for x in list_1 if x not in list_2]


def get_non_overlap_from_all_lists(list_1: List[T], lists: List[List[T]]) -> List[T]:
    non_overlap_set = set(list_1)
    for _list in lists:
        if _list == list_1:
            continue
        non_overlap_set = non_overlap_set.difference(_list)
    return list(non_overlap_set)


def load_ref_and_pred_summaries(root_dir: str, file_exts: List[str]) -> Dict[str, List[str]]:
    ref_and_pred_summaries = {}
    for file_ext in file_exts:
        file_paths = get_file_paths_of_type(root_dir, file_ext)
        for file_path in file_paths:
            reduce_method = get_reduce_method_from_file_path(file_path)
            summaries = read_lines(file_path)
            ref_and_pred_summaries[reduce_method] = summaries
    return ref_and_pred_summaries


def get_reduce_method_from_file_path(file_path: str) -> str:
    file_reduce_method = file_path.split('/')[-1].split('_')[-1].split('.')[0]
    return file_reduce_method


def get_num_words_in_string(text: str) -> int:
    return len(re.findall(r'\w+', text))


def get_average_num_words_in_list_of_strings(string_list: str) -> int:
    num_of_strings = len(string_list)
    if num_of_strings == 0:
        return 0
    average_num_words = 0
    for string in string_list:
        average_num_words += get_num_words_in_string(string)
    average_num_words /= num_of_strings
    return average_num_words
