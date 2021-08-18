# ------ LIBRARY -------#
from transformers import AutoTokenizer, ElectraTokenizer
import pandas as pd
import numpy as np
import pickle
import tqdm
import re
import os

def climate_tokenizer(sent, MAX_LEN, tokenizer):
    encoded_dict = tokenizer.encode_plus(
        text=sent,
        add_special_tokens=True,
        max_length=MAX_LEN,
        pad_to_max_length=True,
        return_attention_mask=True,
        truncation=True)

    input_id = encoded_dict['input_ids']
    attention_mask = encoded_dict['attention_mask']
    token_type_id = encoded_dict['token_type_ids']

    return input_id, attention_mask, token_type_id


def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-하-ㅣ]", " ", sent)
    return sent_clean


def preprocessing_train(args, i):
    pt_name = args.pretrained[i]
    pt = pt_name.replace('/', '_')
    max_len = args.max_len[i]

    train = pd.read_csv('./data_s/train.csv')

    if 'kykim/funnel-kor-base' == pt:
        tokenizer = ElectraTokenizer.from_pretrained(pt_name)

        train = train[['사업명', '요약문_한글키워드', '사업_부처명', '과제명', 'label']]
        train['요약문_한글키워드'] = train['요약문_한글키워드'].fillna('nan')
        train['data'] = train["사업명"] + train['요약문_한글키워드'] + train["사업_부처명"] + train['과제명']

    else:
        tokenizer = AutoTokenizer.from_pretrained(pt_name)

        train = train[['사업명', '사업_부처명', '과제명', 'label']]
        train['data'] = train['사업명'] + train['사업_부처명'] + train['과제명']


    input_ids = []
    attention_masks = []
    token_type_ids = []
    train_data_labels = []

    for train_sent, train_label in tqdm.tqdm(zip(train['data'], train['label'])):
        try:
            input_id, attention_mask, token_type_id = climate_tokenizer(clean_text(train_sent), MAX_LEN=max_len, tokenizer=tokenizer)
            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            #########################################
            train_data_labels.append(train_label)

        except Exception as e:
            print(e)
            pass

    train_input_ids = np.array(input_ids, dtype=int)
    train_attention_masks = np.array(attention_masks, dtype=int)
    train_token_type_ids = np.array(token_type_ids, dtype=int)
    ###########################################################

    # save
    train_data = {}

    train_data['input_ids'] = train_input_ids
    train_data['attention_mask'] = train_attention_masks
    train_data['token_type_ids'] = train_token_type_ids
    train_data['targets'] = np.asarray(train_data_labels, dtype=np.int32)

    os.makedirs(f'./data_s/{pt}/', exist_ok=True)
    with open(f'./data_s/{pt}/train_data_{max_len}.pickle', 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)


def preprocessing_test(args, i):
    pt_name = args.pretrained[i]
    pt = pt_name.replace('/', '_')
    max_len = args.max_len[i]

    test = pd.read_csv('./data_s/test.csv')

    if 'kykim/funnel-kor-base' == pt:
        tokenizer = ElectraTokenizer.from_pretrained(pt_name)

        test = test[['사업명', '요약문_한글키워드', '사업_부처명', '과제명']]
        test['요약문_한글키워드'] = test['요약문_한글키워드'].fillna('nan')
        test['data'] = test["사업명"] + test['요약문_한글키워드'] + test["사업_부처명"] + test['과제명']

    else:
        tokenizer = AutoTokenizer.from_pretrained(pt_name)

        test = test[['사업명', '사업_부처명', '과제명']]
        test['data'] = test['사업명'] + test['사업_부처명'] + test['과제명']


    input_ids = []
    attention_masks = []
    token_type_ids = []

    for test_sent in tqdm.tqdm(test['data']):
        try:
            input_id, attention_mask, token_type_id = climate_tokenizer(clean_text(test_sent), MAX_LEN=max_len, tokenizer=tokenizer)

            input_ids.append(input_id)
            attention_masks.append(attention_mask)
            token_type_ids.append(token_type_id)
            #########################################

        except Exception as e:
            print(e)
            pass

    test_input_ids = np.array(input_ids, dtype=int)
    test_attention_masks = np.array(attention_masks, dtype=int)
    test_token_type_ids = np.array(token_type_ids, dtype=int)
    ###########################################################

    # save
    test_data = {}
    test_data['input_ids'] = test_input_ids
    test_data['attention_mask'] = test_attention_masks
    test_data['token_type_ids'] = test_token_type_ids

    os.makedirs(f'./data_s/{pt}/', exist_ok=True)
    with open(f'./data_s/{pt}/test_data_{max_len}.pickle', 'wb') as f:
        pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)


