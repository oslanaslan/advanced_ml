'''
Task 3 help functions
'''
import os
import re
import random
from typing import List
from collections import Counter, OrderedDict, defaultdict
from copy import copy, deepcopy

import numpy as np
from tqdm import tqdm
from sklearn.base import BaseEstimator


DATA_PATH = 'data'
KARENINA_PATH = os.path.join(DATA_PATH, 'AnnaKarenina.txt')
WARANDPEACE_RU_PATH = os.path.join(DATA_PATH, 'WarAndPeace.txt')
WARANDPEACE_ENG_PATH = os.path.join(DATA_PATH, 'WarAndPeaceEng.txt')

RND_SEED = 1234
EPS = 1e-7


def fix_all() -> None:
    '''
    Fix all random seeds
    '''
    random.seed(RND_SEED)
    np.random.seed(RND_SEED)


def read_data() -> List[str]:
    '''
    Read data from disk
    '''
    regex_ru = re.compile("[^а-яА-ЯёЁ ]")
    regex_en = re.compile("[^a-zA-Z ]")

    with open(KARENINA_PATH, "r") as file:
        karenina_str = file.read().lower().replace("\n", " ")
        karenina_str = regex_ru.sub("", karenina_str)
        karenina_str = karenina_str.strip(" ")
    with open(WARANDPEACE_RU_PATH, "r") as file:
        warandpeace_ru_str = file.read().lower().replace("\n", " ")
        warandpeace_ru_str = regex_ru.sub("", warandpeace_ru_str)
        warandpeace_ru_str = warandpeace_ru_str.strip(" ")
    with open(WARANDPEACE_ENG_PATH, "r") as file:
        warandpeace_en_str = file.read().lower().replace("\n", " ")
        warandpeace_en_str = regex_en.sub("", warandpeace_en_str)
        warandpeace_en_str = warandpeace_en_str.strip(" ")

    return karenina_str, warandpeace_ru_str, warandpeace_en_str


def get_ngrams(text: str, n: int = 1) -> list:
    text = deepcopy(text)
    text = [text[i: i + n] for i in range(len(text) - n + 1)]
    return text

def get_tokens_freq(tokens: list, dens: bool = False) -> dict:
    text_cnt = Counter(tokens)
    text_cnt = OrderedDict(text_cnt.most_common())
    norm = sum(text_cnt.values())
    if dens:
        text_freq_dict = {c: s / norm for c, s in text_cnt.items()}
    else:
        text_freq_dict = dict(text_cnt)
    return text_freq_dict

def encode_text(text: str, n: int = 1) -> str:
    text = get_ngrams(text, n)
    tokens = list(get_tokens_freq(text).keys())
    old_tokens = deepcopy(tokens)
    random.seed(RND_SEED)
    random.shuffle(tokens)
    mapping = dict(zip(old_tokens, tokens))
    new_text = list(map(lambda x: mapping[x], text))
    return "".join(new_text)

def decode_text(text: str, freq_dict: list = None, n: int = None, mapping: dict = None) -> str:
    tokens = get_ngrams(text, n)
    mapping = mapping if mapping else dict(zip(list(get_tokens_freq(tokens=tokens).keys()), freq_dict))
    new_text = list(map(lambda x: mapping[x], tokens))
    return "".join(new_text)

def acc_metric(y_pred: str, y_true: str) -> float:
    res = sum(list(map(
        lambda x: x[0] == x[1],
        list(zip(y_pred, y_true))
    )))
    res /= min(len(y_pred), len(y_true))
    return res

def compare_encodings(text, sample: str, n: int = 1):
    tokens = get_ngrams(text, n)
    freq = list(get_tokens_freq(tokens).keys())
    encoded_text = encode_text(sample, n=n)
    decoded_text = decode_text(encoded_text, freq, n=n)
    score = acc_metric(decoded_text, sample)

    return encoded_text, decoded_text, score

class MCMCModel(BaseEstimator):
    def __init__(self, freq_dict: dict, n: int = 1, encoded_sample: str = None):
        self.mapping_1 = list(set("".join(list(freq_dict.keys())))) if not encoded_sample \
             else list(set(encoded_sample))
        self.freq_dict = defaultdict(lambda: EPS, freq_dict)
        self.mapping_2 = list(set("".join(list(freq_dict.keys()))))
        self.n = n
        self.final_map = None
    
    def score_function(self, decoded_text: str, n_score: int) -> float:
        decoded_tokens = get_ngrams(decoded_text, n_score)
        decoded_freq = get_tokens_freq(decoded_tokens, dens=False)
        score = 0
        for tok in decoded_tokens:
            score += decoded_freq[tok] * np.log(self.freq_dict[tok])
        return score
    
        
    def step(self, current_score: float, proposed_score: float) -> bool:
        if proposed_score > current_score:
            return True
        else:
            return np.random.binomial(1, np.exp(proposed_score - current_score))
    
        
    def fit(self, encoded_text: str, n: int, n_iter: int, sample: str = None):
        def swap(seq):
            idx = range(len(seq))
            i1, i2 = random.sample(idx, 2)
            seq_copy = copy(seq)
            seq_copy[i1], seq_copy[i2] = seq_copy[i2], seq_copy[i1]
            return seq_copy

        current_score = -np.inf
        pbar = tqdm(range(n_iter))

        for _ in pbar:
            prop_mapping = swap(self.mapping_2)
            prop_decoded_text = decode_text(
                text=encoded_text,
                n=self.n,
                mapping=dict(zip(
                    self.mapping_1,
                    prop_mapping)))
            prop_score = self.score_function(
                decoded_text=prop_decoded_text,
                n_score=n)
            if self.step(current_score, prop_score):
                self.mapping_2 = copy(prop_mapping)
                current_score = prop_score
                if sample:
                    enc_acc = acc_metric(prop_decoded_text, sample)
                    pbar.set_postfix_str(f"Accuracy: {enc_acc:.2f}")
            
        self.final_map = dict(zip(self.mapping_1, self.mapping_2))

        decoded_text = list(map(lambda x: self.final_map[x], encoded_text))
        return decoded_text
            
    def transform(self, text: str):
        decoded = list(map(lambda x: self.final_map[x], text))
        return "".join(decoded)


if __name__ == '__main__':
    karenina_str, warandpeace_ru_str, warandpeace_en_str = read_data()
    karenina_sample = karenina_str[:1000]
    anna_karenina_mcmc_2 = MCMCModel(freq_dict=get_tokens_freq(get_ngrams(karenina_str, 2), dens=True), n=1)
    anna_karenina_mcmc_2.fit(encoded_text=encode_text(karenina_sample, n=1), 
                         n=2, 
                         n_iter=30000, 
                         sample=karenina_sample)