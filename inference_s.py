# ------ LIBRARY -------#
import os

from transformers import XLMRobertaForSequenceClassification, FunnelForSequenceClassification
from train_s import device, set_seeds, args
from torch.utils.data import DataLoader
from model_s import ClimateDataSet
from os.path import join as opj
import torch.cuda.amp as amp
import torch.nn as nn
from glob import glob
import pandas as pd
import numpy as np
import pickle
import torch
import tqdm

set_seeds()

def do_predict(net, valid_loader):

    pred_lst = []
    logit = []
    net.eval()
    for t, data in enumerate(tqdm.tqdm(valid_loader)):

        ids = data['ids'].to(device)
        mask = data['mask'].to(device)

        with torch.no_grad():
            with amp.autocast():
                output = net(ids, mask)[0]

            pred_lst.extend(output.argmax(dim=1).tolist())
            logit.extend(output.tolist())
        break
    return pred_lst, logit


def run_predict(model_path, pt, i):

    max_len = args.max_len[i]
    pt_name = pt.replace('_', '/')

    with open(f'./data_s/{pt}/test_data_{max_len}.pickle', 'rb') as f:
        test_data = pickle.load(f)

    print('test load')

    test_dataset = ClimateDataSet(data=test_data, test=True)
    testloader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    print('set testloader')

    ## net ----------------------------------------

    if 'kykim/funnel-kor-base' == pt_name:
        net = FunnelForSequenceClassification.from_pretrained(pt_name, num_labels=46)
    else:
        net = XLMRobertaForSequenceClassification.from_pretrained(pt_name, num_labels=46)
    net.to(device)

    if len(args.gpu) > 1:
        net = nn.DataParallel(net)

    f = torch.load(model_path)
    net.load_state_dict(f, strict=True)  # True
    print('load saved models')
    # ------------------------
    # validation
    preds, logit = do_predict(net, testloader)  # outputs

    print('complete predict')

    return preds, logit


# ------ INFERENCE ------ #
if __name__ == "__main__":

    for i in range(2):

        pt_name = args.pretrained[i]
        pt = pt_name.replace("/", "_")
        out_dir = args.dir_ + f'/{pt}/'

        file_list = glob(opj(out_dir, "*.pth"))

        preds1, logit1 = run_predict(file_list[0], pt, i)
        preds2, logit2 = run_predict(file_list[1], pt, i)
        preds3, logit3 = run_predict(file_list[2], pt, i)
        preds4, logit4 = run_predict(file_list[3], pt, i)
        preds5, logit5 = run_predict(file_list[4], pt, i)

        logit1 = np.array(logit1)
        logit2 = np.array(logit2)
        logit3 = np.array(logit3)
        logit4 = np.array(logit4)
        logit5 = np.array(logit5)

        dir_, fn = os.path.split(f"./csvs/{pt}_1.csv")
        os.makedirs(dir_, exist_ok=True)
        dir_, fn = os.path.split(f"./csvs/{pt}_2.csv")
        os.makedirs(dir_, exist_ok=True)
        dir_, fn = os.path.split(f"./csvs/{pt}_3.csv")
        os.makedirs(dir_, exist_ok=True)
        dir_, fn = os.path.split(f"./csvs/{pt}_4.csv")
        os.makedirs(dir_, exist_ok=True)
        dir_, fn = os.path.split(f"./csvs/{pt}_5.csv")
        os.makedirs(dir_, exist_ok=True)

        result1 = pd.DataFrame(logit1).to_csv(f"./csvs/{pt}_1.csv", index=False)
        result2 = pd.DataFrame(logit2).to_csv(f"./csvs/{pt}_2.csv", index=False)
        result3 = pd.DataFrame(logit3).to_csv(f"./csvs/{pt}_3.csv", index=False)
        result4 = pd.DataFrame(logit4).to_csv(f"./csvs/{pt}_4.csv", index=False)
        result5 = pd.DataFrame(logit5).to_csv(f"./csvs/{pt}_5.csv", index=False)

