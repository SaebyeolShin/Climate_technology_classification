from os.path import join as opj
import os
import sys
import pickle
import re
import random
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import f1_score, accuracy_score

from transformers import BertTokenizer, BertPreTrainedModel
import transformers
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import torch_optimizer as optim

from utils_j import *
from preprocess_j import *
from dataset_j import *
from model_j import *

parser = argparse.ArgumentParser()
#### data ####
parser.add_argument("--train_features", default=['사업명', "요약문_한글키워드", "사업_부처명", '과제명', '요약문_연구내용', 'label'])
parser.add_argument("--test_features", default=['사업명', "요약문_한글키워드", "사업_부처명", '과제명', '요약문_연구내용'])
parser.add_argument("--n_folds", type=int, default=5)
parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased")
parser.add_argument("--max_len", type=int, default=250)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--n_classes", type=int, default=46)
#### train ####
parser.add_argument("--n_epochs", type=int, default=3)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay_ratio", type=float, default=1e-6)

#### config ####
parser.add_argument("--seed", type=int, default=123123)
parser.add_argument("--cuda_visible_devices", type=str, default="1")
parser.add_argument("--model_save_dir", type=str, default="./save_models")

args = parser.parse_args()
args.data_root_dir = "./data_j"
args.cache_dir = "bert_cache"
args.tokenized_data_dir = opj(args.data_root_dir, "tokenized_data")

parser2 = argparse.ArgumentParser()

#### data ####
parser2.add_argument("--train_features", default=["사업명", "요약문_한글키워드", "사업_부처명", "과제명", "label", "label_m"])
parser2.add_argument("--test_features", default=["사업명", "요약문_한글키워드", "사업_부처명", '과제명'])
parser2.add_argument("--n_folds", type=int, default=5)
parser2.add_argument("--model_name", type=str, default="klue/roberta-large")
parser2.add_argument("--max_len", default=164)
parser2.add_argument("--num_workers", type=int, default=8)
parser2.add_argument("--n_classes", type=int, default=46)
parser2.add_argument("--n_middle_classes", type=int, default=15)

#### train ####
parser2.add_argument("--n_epochs", type=int, default=3)
parser2.add_argument("--batch_size", type=int, default=32)
parser2.add_argument("--lr", type=float, default=1e-5)
parser2.add_argument("--weight_decay_ratio", type=float, default=1e-6)

#### config ####
parser2.add_argument("--seed", type=int, default=123123)
parser2.add_argument("--model_save_dir", type=str, default="./save_models")

args2 = parser2.parse_args()
args2.data_root_dir = "./data_j"
args2.tokenized_data_dir = opj(args2.data_root_dir, "tokenized_data")

fix_seed(args.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices


def train_bert_base(args):
    logger = Logger()
    args.log_dir = f"./log/{args.model_name}_{args.max_len}_{args.train_features}"
    os.makedirs(args.log_dir, exist_ok=True)
    logger.open(opj(args.log_dir, "logger.txt"))
    train_csv, test_csv = load_splited_csv(args)
    tokenized_train_data_path = opj(args.tokenized_data_dir, f"{args.model_name}_{args.max_len}_{args.train_features}_train.pickle")
    with open(tokenized_train_data_path, "rb") as f:
        tokenized_train_data = pickle.load(f)

    scaler = GradScaler()
    for fold in range(args.n_folds):
        train_idx = train_csv[train_csv["fold"] != fold]["idx"].values
        valid_idx = train_csv[train_csv["fold"] == fold]["idx"].values

        #### fold로 train, valid 나누기 ####
        train_data = {"input_ids": tokenized_train_data["input_ids"][train_idx],
                      "attention_mask": tokenized_train_data["attention_mask"][train_idx],
                      "token_type_ids": tokenized_train_data["token_type_ids"][train_idx],
                      "labels": tokenized_train_data["labels"][train_idx]}

        valid_data = {"input_ids": tokenized_train_data["input_ids"][valid_idx],
                      "attention_mask": tokenized_train_data["attention_mask"][valid_idx],
                      "token_type_ids": tokenized_train_data["token_type_ids"][valid_idx],
                      "labels": tokenized_train_data["labels"][valid_idx]}

        train_dataset = climate_dataset(train_data, is_test=False)
        valid_dataset = climate_dataset(valid_data, is_test=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

        model = BertForMultiLabelSequenceClassification.from_pretrained(args.model_name, num_labels=args.n_classes,
                                                              return_dict=False)
        model.cuda()
        model = nn.DataParallel(model)

        criterion_CE = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay_ratio)

        best_f1_score = -1
        best_epoch = 1
        best_state_dict = None
        save_path = None
        for epoch in range(1, args.n_epochs + 1):
            model.train()
            train_loss = AverageMeter()
            labels = []
            preds = []
            with tqdm(train_loader) as train_loop:
                for data in train_loop:
                    input_id = data['input_id'].cuda()
                    attention_mask_ = data["attention_mask_"].cuda()
                    token_type_id = data["token_type_id"].cuda()
                    label = data["label"].cuda()


                    with autocast():
                        output = model(input_id, attention_mask_, token_type_id)
                        _loss = criterion_CE(output, label)
                        train_loss.update(_loss.item())

                    scaler.scale(_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    labels.extend(label.detach().cpu().numpy())
                    pred = output.argmax(dim=1).detach().cpu().numpy()
                    preds.extend(pred)
                    break

            train_f1_score = f1_score(labels, preds, average="macro")

            model.eval()
            valid_loss = AverageMeter()
            labels = []
            preds = []
            with torch.no_grad():
                with tqdm(valid_loader) as valid_loop:
                    for data in valid_loop:
                        input_id = data["input_id"].cuda()
                        attention_mask_ = data["attention_mask_"].cuda()
                        token_type_id = data["token_type_id"].cuda()
                        label = data["label"].cuda()

                        with autocast():
                            output = model(input_id, attention_mask_, token_type_id)
                            _loss = criterion_CE(output, label)
                            valid_loss.update(_loss.item())

                        labels.extend(label.detach().cpu().numpy())
                        pred = output.argmax(dim=1).detach().cpu().numpy()
                        preds.extend(pred)
                        break



            valid_f1_score = f1_score(labels, preds, average="macro")

            if valid_f1_score > best_f1_score:
                logger.write(f"best score: {best_f1_score} -> {valid_f1_score}\n")
                best_f1_score = valid_f1_score
                best_epoch = epoch
                model_name = args.model_name.replace("/", "_")
                save_path = opj(args.model_save_dir, f"{model_name}_{fold}_{epoch}_{best_f1_score}.pth")
                best_state_dict = {"model": model.state_dict(),
                                   "max_len": args.max_len,
                                   "model_name": args.model_name,
                                   "train_features": args.train_features,
                                   "test_features": args.test_features,
                                   "is_middle": False}


            logger.write(f"train loss: {train_loss.avg}, train f1 score: {train_f1_score}\n")
            logger.write(f"valid loss: {valid_loss.avg}, valid f1 score: {valid_f1_score}\n")

        logger.write("best f1 score: {}\n".format(best_f1_score))
        logger.write("best epoch: {}\n".format(best_epoch))
        os.makedirs(args.model_save_dir, exist_ok=True)
        torch.save(best_state_dict, save_path)

def train_robert_base_middle(args):
    logger = Logger()
    args.log_dir = f"./log/{args.model_name}_{args.max_len}_{args.train_features}"
    os.makedirs(args.log_dir, exist_ok=True)
    logger.open(opj(args.log_dir, "logger.txt"))
    train_csv, test_csv = load_splited_csv2(args)
    tokenized_train_data_path = opj(args.tokenized_data_dir,
                                    "{}_{}_{}_train.pickle".format(args.model_name.replace("/", "_"), args.max_len, args.train_features))
    with open(tokenized_train_data_path, "rb") as f:
        tokenized_train_data = pickle.load(f)

    scaler = GradScaler()
    for fold in range(args.n_folds):
        train_idx = train_csv[train_csv["fold"] != fold]["idx"].values
        valid_idx = train_csv[train_csv["fold"] == fold]["idx"].values

        #### fold로 train, valid 나누기 ####
        train_data = {"input_ids": tokenized_train_data["input_ids"][train_idx],
                      "attention_mask": tokenized_train_data["attention_mask"][train_idx],
                      "token_type_ids": tokenized_train_data["token_type_ids"][train_idx],
                      "s_labels": tokenized_train_data["s_labels"][train_idx],
                      "m_labels": tokenized_train_data["m_labels"][train_idx]}

        valid_data = {"input_ids": tokenized_train_data["input_ids"][valid_idx],
                      "attention_mask": tokenized_train_data["attention_mask"][valid_idx],
                      "token_type_ids": tokenized_train_data["token_type_ids"][valid_idx],
                      "s_labels": tokenized_train_data["s_labels"][valid_idx],
                      "m_labels": tokenized_train_data["m_labels"][valid_idx]}

        train_dataset = climate_dataset2(train_data, is_test=False)
        valid_dataset = climate_dataset2(valid_data, is_test=False)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  pin_memory=True)

        model = custom_XLMRoberta(args)
        model.cuda()
        model = nn.DataParallel(model)

        criterion_CE = nn.CrossEntropyLoss()
        optimizer = optim.Lookahead(optim.RAdam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr), alpha=0.5, k=5)

        best_f1_score = -1
        best_epoch = 1
        best_state_dict = None
        save_path = None
        for epoch in range(1, args.n_epochs + 1):
            model.train()
            train_loss = AverageMeter()
            labels = []
            preds = []
            with tqdm(train_loader) as train_loop:
                for data in train_loop:
                    input_id = data["input_id"].cuda()
                    attention_mask_ = data["attention_mask_"].cuda()
                    token_type_id = data["token_type_id"].cuda()
                    s_label = data["s_label"].cuda()
                    m_label = data["m_label"].cuda()

                    with autocast():
                        s_output, m_output = model(input_id, attention_mask_)
                        _loss_1 = criterion_CE(s_output, s_label)
                        _loss_2 = criterion_CE(m_output, m_label)

                        _loss = _loss_1 + _loss_2
                        train_loss.update(_loss.item())

                    scaler.scale(_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    break


                labels.extend(s_label.detach().cpu().numpy())
                pred = s_output.argmax(dim=1).detach().cpu().numpy()
                preds.extend(pred)

            train_f1_score = f1_score(labels, preds, average="macro")

            model.eval()
            valid_loss = AverageMeter()
            labels = []
            preds = []
            with torch.no_grad():
                with tqdm(valid_loader) as valid_loop:
                    for data in valid_loop:
                        input_id = data["input_id"].cuda()
                        attention_mask_ = data["attention_mask_"].cuda()
                        token_type_id = data["token_type_id"].cuda()
                        s_label = data["s_label"].cuda()
                        m_label = data["m_label"].cuda()

                        with autocast():
                            s_output, m_output = model(input_id, attention_mask_)
                            _loss_1 = criterion_CE(s_output, s_label)
                            _loss_2 = criterion_CE(m_output, m_label)

                            _loss = _loss_1 + _loss_2
                            valid_loss.update(_loss.item())
                        labels.extend(s_label.detach().cpu().numpy())
                        pred = s_output.argmax(dim=1).detach().cpu().numpy()
                        preds.extend(pred)
                        break



            valid_f1_score = f1_score(labels, preds, average="macro")

            if valid_f1_score > best_f1_score:
                logger.write(f"best score: {best_f1_score} -> {valid_f1_score}\n")
                best_f1_score = valid_f1_score
                best_epoch = epoch
                model_name = args.model_name.replace("/", "_")
                save_path = opj(args.model_save_dir, f"{model_name}_{fold}_{epoch}_{best_f1_score}.pth")
                best_state_dict = {"model": model.state_dict(),
                                   "max_len": args.max_len,
                                   "model_name": args.model_name,
                                   "train_features": args.train_features,
                                   "test_features": args.test_features,
                                   "is_middle": True}


            logger.write(f"train loss: {train_loss.avg}, train f1 score: {train_f1_score}\n")
            logger.write(f"valid loss: {valid_loss.avg}, valid f1 score: {valid_f1_score}\n")

        logger.write("best f1 score: {}\n".format(best_f1_score))
        logger.write("best epoch : {}\n".format(best_epoch))
        os.makedirs(args.model_save_dir, exist_ok=True)
        torch.save(best_state_dict, save_path)

if __name__ == "__main__":
    tokenize(args)
    train_bert_base(args)
    make_middle_category(args2)
    tokenize2(args2)
    train_robert_base_middle(args2)
