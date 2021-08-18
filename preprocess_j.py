from os.path import join as opj
import os
import pickle
from tqdm import tqdm

from transformers import BertTokenizer, BertPreTrainedModel, AutoTokenizer
import pandas as pd

from utils_j import *

def tokenize(args):
    train = pd.read_csv(opj(args.data_root_dir, "original/train.csv"))
    train = train[args.train_features]
    if "요약문_연구내용" in args.train_features:
        train["요약문_연구내용"] = train["요약문_연구내용"].fillna("NAN")
    if "요약문_한글키워드" in args.train_features:
        train["요약문_한글키워드"] = train["요약문_한글키워드"].fillna("NAN")

    for feature in args.train_features[:-1]:
        try:
            train["data"] += train[feature]
        except:
            train["data"] = train[feature]

    tokenizer = BertTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir, do_lower_case=False)

    #### train 토크나이징 ####
    input_ids = []
    attention_mask = []
    token_type_ids = []
    labels = []
    for _sent, label in tqdm(zip(train["data"], train["label"])):
        try:
            korean_sent = delete_eng(_sent)
            _dict = tokenizer.encode_plus(
                text=korean_sent,
                max_length=args.max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True,
                add_special_tokens=True
            )
            input_id, attention_mask_, token_type_id = _dict["input_ids"], _dict["attention_mask"], _dict[
                "token_type_ids"]

            input_ids.append(input_id)
            attention_mask.append(attention_mask_)
            token_type_ids.append(token_type_id)
            labels.append(label)


        except Exception as e:
            print(e)
            pass
    input_ids = np.asarray(input_ids, dtype=np.int32)
    attention_mask = np.asarray(attention_mask, dtype=np.int32)
    token_type_ids = np.asarray(token_type_ids, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)

    train_data = {}
    train_data["input_ids"] = input_ids
    train_data["attention_mask"] = attention_mask
    train_data["token_type_ids"] = token_type_ids
    train_data["labels"] = labels

    os.makedirs(args.tokenized_data_dir, exist_ok=True)
    tokenized_train_data_path = opj(args.tokenized_data_dir,
                                    "{}_{}_{}_train.pickle".format(args.model_name.replace("/", "_"), args.max_len,
                                                                   args.train_features))
    with open(tokenized_train_data_path, "wb") as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    test = pd.read_csv(opj(args.data_root_dir, "original/test.csv"))
    test = test[args.test_features]

    if "요약문_연구내용" in args.test_features:
        test["요약문_연구내용"] = test["요약문_연구내용"].fillna("NAN")
    if "요약문_한글키워드" in args.test_features:
        test["요약문_한글키워드"] = test["요약문_한글키워드"].fillna("NAN")

    for features in args.test_features:
        try:
            test["data"] += test[feature]
        except:
            test["data"] = test[feature]

    #### test 토크나이징 ####
    input_ids = []
    attention_mask = []
    token_type_ids = []

    for _sent in tqdm(test["data"]):
        try:
            korean_sent = delete_eng(_sent)
            _dict = tokenizer.encode_plus(
                text=korean_sent,
                max_length=args.max_len,
                pad_to_max_length=True,
                return_attention_mask=True,
                truncation=True,
                add_special_tokens=True
            )
            input_id, attention_mask_, token_type_id = _dict["input_ids"], _dict["attention_mask"], _dict[
                "token_type_ids"]

            input_ids.append(input_id)
            attention_mask.append(attention_mask_)
            token_type_ids.append(token_type_id)

        except:
            pass

    input_ids = np.asarray(input_ids, dtype=np.int32)
    attention_mask = np.asarray(attention_mask, dtype=np.int32)
    token_type_ids = np.asarray(token_type_ids, dtype=np.int32)

    test_data = {}
    test_data["input_ids"] = input_ids
    test_data["attention_mask"] = attention_mask
    test_data["token_type_ids"] = token_type_ids

    tokenized_test_data_path = opj(args.tokenized_data_dir,
                                   "{}_{}_{}_test.pickle".format(args.model_name.replace("/", "_"), args.max_len,
                                                                 args.test_features))
    with open(tokenized_test_data_path, "wb") as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

def tokenize2(args):
    train = pd.read_csv(opj(args.data_root_dir, "category_add/train_add.csv"))
    train = train[args.train_features]
    if "사업명" in args.train_features:
        train["사업명"] = train["사업명"].apply(lambda x: x + " ")
    if "요약문_연구내용" in args.train_features:
        train["요약문_연구내용"] = train["요약문_연구내용"].fillna("NAN")
    if "요약문_한글키워드" in args.train_features:
        train["요약문_한글키워드"] = train["요약문_한글키워드"].fillna("NAN")
    for feature in args.train_features[:-2]:
        try:
            train["data"] += train[feature]
        except:
            train["data"] = train[feature]

    test = pd.read_csv(opj(args.data_root_dir, "original/test.csv"))
    test = test[args.test_features]
    if "사업명" in args.test_features:
        test["사업명"] = test["사업명"].apply(lambda x: x + " ")
    if "요약문_연구내용" in args.test_features:
        test["요약문_연구내용"] = test["요약문_연구내용"].fillna("NAN")
    if "요약문_한글키워드" in args.test_features:
        test["요약문_한글키워드"] = test["요약문_한글키워드"].fillna("NAN")

    for feature in args.test_features:
        try:
            test["data"] += test[feature]
        except Exception as e:
            test["data"] = test[feature]

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    #### 최적의 max len 탐색 ####
    if args.max_len is None:
        _len_list = []
        for i in range(len(train["data"])):
            _len_list.append(len(tokenizer.tokenize(delete_eng(train["data"][i]))))
        train_max_len = max(_len_list)

        _len_list = []
        for i in range(len(test["data"])):
            _len_list.append(len(tokenizer.tokenize(delete_eng(test["data"][i]))))

        test_max_len = max(_len_list)
        args.max_len = min(max(train_max_len, test_max_len) + 2, 200)

    input_ids = []
    attention_mask = []
    token_type_ids = []
    s_labels = []
    m_labels = []

    for _iter, (_sent, s_label, m_label) in tqdm(enumerate(zip(train["data"], train["label"], train["label_m"]))):
        korean_sent = delete_eng(_sent)
        _dict = tokenizer.encode_plus(
            text=korean_sent,
            max_length=args.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            add_special_tokens=True
        )
        input_id, attention_mask_, token_type_id = _dict["input_ids"], _dict["attention_mask"], _dict[
            "token_type_ids"]

        input_ids.append(input_id)
        attention_mask.append(attention_mask_)
        token_type_ids.append(token_type_id)
        s_labels.append(s_label)
        m_labels.append(m_label)

    input_ids = np.asarray(input_ids, dtype=np.int64)
    attention_mask = np.asarray(attention_mask, dtype=np.int64)
    token_type_ids = np.asarray(token_type_ids, dtype=np.int64)
    s_labels = np.asarray(s_labels, dtype=np.int64)
    m_labels = np.asarray(m_labels, dtype=np.int64)

    train_data = {}
    train_data["input_ids"] = input_ids
    train_data["attention_mask"] = attention_mask
    train_data["token_type_ids"] = token_type_ids
    train_data["s_labels"] = s_labels
    train_data["m_labels"] = m_labels

    os.makedirs(args.tokenized_data_dir, exist_ok=True)
    tokenized_train_data_path = opj(args.tokenized_data_dir,
                                    "{}_{}_{}_train.pickle".format(args.model_name.replace("/", "_"), args.max_len,
                                                                   args.train_features))
    with open(tokenized_train_data_path, "wb") as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)

    input_ids = []
    attention_mask = []
    token_type_ids = []

    for _sent in tqdm(test["data"]):
        korean_sent = delete_eng(_sent)
        _dict = tokenizer.encode_plus(
            text=korean_sent,
            max_length=args.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            add_special_tokens=True
        )
        input_id, attention_mask_, token_type_id = _dict["input_ids"], _dict["attention_mask"], _dict["token_type_ids"]
        input_ids.append(input_id)
        attention_mask.append(attention_mask_)
        token_type_ids.append(token_type_id)

    input_ids = np.asarray(input_ids, dtype=np.int32)
    attention_mask = np.asarray(attention_mask, dtype=np.int32)
    token_type_ids = np.asarray(token_type_ids, dtype=np.int32)

    test_data = {}
    test_data["input_ids"] = input_ids
    test_data["attention_mask"] = attention_mask
    test_data["token_type_ids"] = token_type_ids
    tokenized_test_data_path = opj(args.tokenized_data_dir,
                                   "{}_{}_{}_test.pickle".format(args.model_name.replace("/", "_"), args.max_len,
                                                                 args.test_features))
    with open(tokenized_test_data_path, "wb") as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)









