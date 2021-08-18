# ------ LIBRARY -------#
from transformers import XLMRobertaForSequenceClassification, FunnelForSequenceClassification, get_linear_schedule_with_warmup
from preprocessing_s import preprocessing_train, preprocessing_test
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from model_s import ClimateDataSet
import torch_optimizer as optim
import torch.cuda.amp as amp
import torch.nn as nn
import pandas as pd
import numpy as np
import argparse
import pickle
import random
import torch
import tqdm
import sys
import os

# ------ ARGS ------ #
parser = argparse.ArgumentParser(description='Climate args')

parser.add_argument('--pretrained', '-pt', help = 'pretrained_model', default = ['klue/roberta-large','kykim/funnel-kor-base'])
parser.add_argument('--dir_','-dir_', help = 'dir', default = './saved_models_s')
parser.add_argument('--max_len','-max', help = 'max_len', default = [87, 371])
parser.add_argument('--epochs', '-e' ,type=int, help='epochs', default=3)
parser.add_argument('--gpu', help='gpu', default='1')
parser.add_argument('--initial_checkpoint','-checkpoint', help='initial_checkpoint', default=None)
parser.add_argument('--batch_size', '-batch', type=int, help='batch_size', default=32)
parser.add_argument('--start_lr', help='start_lr', default= 1e-5)

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")


# ------ UTILS ------ #
def set_seeds(seed=92):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # for faster training, but not deterministic
set_seeds()

def load_data(pt):

    train = pd.read_csv('./data_s/train.csv')
    test = pd.read_csv('./data_s/test.csv')

    if 'kykim/funnel-kor-base' == pt:

        train = train[['사업명', '요약문_한글키워드', '사업_부처명', '과제명', 'label']]
        train['요약문_한글키워드'] = train['요약문_한글키워드'].fillna('nan')
        train['data'] = train["사업명"] + train['요약문_한글키워드'] + train["사업_부처명"] + train['과제명']

        test = test[['사업명', '요약문_한글키워드', '사업_부처명', '과제명']]
        test['요약문_한글키워드'] = test['요약문_한글키워드'].fillna('nan')
        test['data'] = test["사업명"] + test['요약문_한글키워드'] + test["사업_부처명"] + test['과제명']

    else:

        train = train[['사업명', '사업_부처명', '과제명', 'label']]
        train['data'] = train["사업명"] + train["사업_부처명"] + train['과제명']

        test = test[['사업명', '사업_부처명', '과제명']]
        test['data'] = test["사업명"] + test["사업_부처명"] + test['과제명']


    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    train['fold'] = -1
    for n_fold, (_, v_idx) in enumerate(skf.split(train, train['label'])):
        train.loc[v_idx, 'fold'] = n_fold
    train['id'] = [x for x in range(len(train))]

    return train, test

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


    def flush(self):
        pass


# ------ TRAIN ------ #
def do_valid(net, valid_loader):

    val_loss = 0
    target_lst = []
    pred_lst = []
    logit = []
    loss_fn = nn.CrossEntropyLoss()

    net.eval()
    for t, data in enumerate(tqdm.tqdm(valid_loader)):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        target = data['targets'].to(device)

        with torch.no_grad():
            with amp.autocast():
                output = net(ids, mask)
                output = output[0]

                loss = loss_fn(output, target)

            val_loss += loss
            target_lst.extend(target.detach().cpu().numpy())
            pred_lst.extend(output.argmax(dim=1).tolist())
            logit.extend(output.tolist())

        val_mean_loss = val_loss / len(valid_loader)
        validation_score = f1_score(y_true=target_lst, y_pred=pred_lst, average='macro')
        break
    return val_mean_loss, validation_score, logit


def run_train(args, i):

    pt_name = args.pretrained[i]
    pt = pt_name.replace("/", "_")
    max_len = args.max_len[i]
    log = Logger()

    out_dir = args.dir_ + f'/{pt}/'
    os.makedirs(out_dir, exist_ok=True)
    log.open(out_dir + '/log.train.txt', mode='a')

    train, test = load_data(pt)
    with open(f'./data_s/{pt}/train_data_{max_len}.pickle', 'rb') as f:
        train_data = pickle.load(f)
    log.write('load dataset' + '\n')

    for n_fold in range(5):

        trn_idx = train[train['fold'] != n_fold]['id'].values
        val_idx = train[train['fold'] == n_fold]['id'].values

        train_dict = {'input_ids': train_data['input_ids'][trn_idx],
                      'attention_mask': train_data['attention_mask'][trn_idx],
                      'token_type_ids': train_data['token_type_ids'][trn_idx],
                      'targets': train_data['targets'][trn_idx]}

        val_dict = {'input_ids': train_data['input_ids'][val_idx],
                    'attention_mask': train_data['attention_mask'][val_idx],
                    'token_type_ids': train_data['token_type_ids'][val_idx],
                    'targets': train_data['targets'][val_idx]}

        train_dataset = ClimateDataSet(data=train_dict)
        valid_dataset = ClimateDataSet(data=val_dict)
        trainloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True)
        validloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)


        if 'kykim/funnel-kor-base' == pt_name:
            net = FunnelForSequenceClassification.from_pretrained(pt_name, num_labels=46)
            log.write('load kykim/funnel-kor-base model' + '\n')
        else:
            net = XLMRobertaForSequenceClassification.from_pretrained(pt_name, num_labels=46)
            log.write('load klue/roberta-large model' + '\n')

        net.to(device)

        if len(args.gpu) > 1:
            net = nn.DataParallel(net)

        if args.initial_checkpoint is not None:
            f = torch.load(args.initial_checkpoint)
            net.load_state_dict(f, strict=True)  # True
            log.write('load saved models' + '\n')


        scaler = amp.GradScaler()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Lookahead(optim.RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.start_lr), alpha=0.5, k=5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(trainloader) * args.epochs)

        # ----
        best_score = -1
        best_state_dict = None

        for epoch in range(1, args.epochs + 1):
            train_loss = 0

            target_lst = []
            pred_lst = []
            log.write(f'-------------------')
            log.write(f'{epoch}epoch start')
            log.write(f'-------------------' + '\n')

            for t, data in enumerate(tqdm.tqdm(trainloader)):

                ids = data['ids'].to(device)
                mask = data['mask'].to(device)
                target = data['targets'].to(device)

                # ------------
                net.train()
                optimizer.zero_grad()

                with amp.autocast():
                    output = net(ids, mask)
                    output = output[0]

                    loss = loss_fn(output, target)
                    train_loss += loss

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if scheduler is not None:
                        scheduler.step()

                    target_lst.extend(target.detach().cpu().numpy())
                    pred_lst.extend(output.argmax(dim=1).tolist())

                    train_loss = train_loss / len(trainloader)
                    train_score = f1_score(y_true=target_lst, y_pred=pred_lst, average='macro')

                    valid_loss, valid_score, logit = do_valid(net, validloader)

                    if valid_score > best_score:
                        best_score = valid_score
                        best_state_dict = net.state_dict()
                        log.write('best model saved' + '\n')

                    log.write(f'train loss : {train_loss:.4f}, train f1 score : {train_score : .4f}' + '\n')
                    log.write(f'valid loss : {valid_loss:.4f}, valid f1 score : {valid_score : .4f}' + '\n')
                break
        torch.save(best_state_dict, out_dir + f'/{n_fold}f_{best_score:.4f}_s.pth')

if __name__ == "__main__":

    # preprocessing
    for i in range(2):
        preprocessing_train(args, i)
        preprocessing_test(args, i)

        # train (5 fold)
        run_train(args, i)