from os.path import join as opj
import torch
import numpy as np
import random
import re
import sys

import torch.nn as nn


import pandas as pd
from sklearn.model_selection import StratifiedKFold

def fix_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True # this can slow down speed
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  # stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode = 'w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1):
        if '\r' in message: is_file = 0
        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
        if is_file == 1:
            self.file.write(message)
            self.file.flush()


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def delete_eng(sent):
    korean_sent = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", sent)
    return korean_sent

def load_splited_csv(args):
    train_csv = pd.read_csv(opj(args.data_root_dir, "original/train.csv"))
    test_csv = pd.read_csv(opj(args.data_root_dir, "original/test.csv"))

    train_csv = train_csv[args.train_features]
    test_csv = test_csv[args.test_features]

    train_csv["요약문_연구내용"] = train_csv["요약문_연구내용"].fillna('NAN')
    test_csv["요약문_연구내용"] = test_csv["요약문_연구내용"].fillna('NAN')

    for feature in args.train_features[:-1]:
        try:
            train_csv['data'] += train_csv[feature]
        except:
            train_csv["data"] = train_csv[feature]

    for feature in args.test_features[:-1]:
        try:
            test_csv["data"] += test_csv[feature]
        except:
            test_csv["data"] = test_csv[feature]

    strat_kfold = StratifiedKFold(n_splits=args.n_folds, random_state=args.seed, shuffle=True)
    train_csv["fold"] = 123123
    for fold, (_, idx) in enumerate(strat_kfold.split(train_csv, train_csv["label"])):
        train_csv.loc[idx, "fold"] = fold
    train_csv["idx"] = [x for x in range(len(train_csv))]
    return train_csv, test_csv

def load_splited_csv2(args):
    train_csv = pd.read_csv(opj(args.data_root_dir, "category_add/train_add.csv"))
    test_csv = pd.read_csv(opj(args.data_root_dir, "original/test.csv"))

    train_csv = train_csv[args.train_features]
    test_csv = test_csv[args.test_features]

    for feature in args.train_features[:-1]:
        try:
            train_csv['data'] += train_csv[feature]
        except:
            train_csv["data"] = train_csv[feature]

    for feature in args.test_features[:-1]:
        try:
            test_csv["data"] += test_csv[feature]
        except:
            test_csv["data"] = test_csv[feature]

    strat_kfold = StratifiedKFold(n_splits=args.n_folds, random_state=args.seed, shuffle=True)
    train_csv["fold"] = 123123
    for fold, (_, idx) in enumerate(strat_kfold.split(train_csv, train_csv["label"])):
        train_csv.loc[idx, "fold"] = fold
    train_csv["idx"] = [x for x in range(len(train_csv))]
    return train_csv, test_csv


def make_middle_category(args):
    train = pd.read_csv(opj(args.data_root_dir, "original/train.csv"))
    df = pd.read_csv(opj(args.data_root_dir, "category_add/custom_labels_mapping.csv"))
    df = df[['대분류_label', '중분류_label', '소분류_label']]
    df = df.set_index(['소분류_label'])

    dict1 = {}
    dict2 = {}
    for i, j in df.to_dict()['대분류_label'].items():
        dict1[i] = j
    for i, j in df.to_dict()['중분류_label'].items():
        dict2[i] = j

    train['label_m'] = train['label'].apply(lambda x: dict2[x])
    train['label_l'] = train['label'].apply(lambda x: dict1[x])
    train.to_csv(opj(args.data_root_dir, "category_add/train_add.csv"), index=False)
