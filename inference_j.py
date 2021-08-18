import pickle
import argparse
from tqdm import tqdm
import os
from os.path import join as opj
from glob import glob

import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from model_j import BertForMultiLabelSequenceClassification, custom_XLMRoberta
from dataset_j import climate_dataset


parser_inference = argparse.ArgumentParser()

parser_inference.add_argument("--save_dir", type=str, default="./save_models")
parser_inference.add_argument("--pickle_dir", type=str, default="./data_j/tokenized_data")
parser_inference.add_argument("--batch_size", type=int, default=32)
parser_inference.add_argument("--num_workers", type=int, default=8)
parser_inference.add_argument("--n_classes", type=int, default=46)
parser_inference.add_argument("--n_middle_classes", type=int, default=15)
parser_inference.add_argument("--result_save_dir", type=str, default="./csvs")
args = parser_inference.parse_args()

def run_ensemble_soft_logit_each_save(args):
    model_paths = glob(opj(args.save_dir, "*.pth"))
    for model_path in model_paths:
        dir_, model_fn = os.path.split(model_path)
        model_fn = model_fn[:-4]
        state_dict = torch.load(model_path)
        model_name = state_dict["model_name"]
        args.model_name = model_name
        model_name = model_name.replace("/", "_")
        max_len = state_dict["max_len"]
        test_features = state_dict["test_features"]
        is_middle = state_dict["is_middle"]
        pickle_path = opj(args.pickle_dir, f"{model_name}_{max_len}_{test_features}_test.pickle")
        with open(pickle_path, "rb") as f:
            test_dict = pickle.load(f)

        test_dataset = climate_dataset(data=test_dict, is_test=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size,
                                 num_workers=args.num_workers, shuffle=False, pin_memory=True)

        model = None
        if "bert-base-multilingual-cased" == model_name:
            model = BertForMultiLabelSequenceClassification.from_pretrained(model_name, num_labels=args.n_classes,
                                                                            return_dict=False)
        elif "klue_roberta-large" == model_name:
            model = custom_XLMRoberta(args)
        model = model.cuda()
        model.eval()


        with tqdm(enumerate(test_loader)) as test_loop:
            for t, data in test_loop:
                input_id = data["input_id"].cuda()
                attention_mask_ = data["attention_mask_"].cuda()
                token_type_id = data["token_type_id"].cuda()

                with torch.no_grad():
                    if not is_middle:
                        with autocast():
                            output = model(input_id, attention_mask_, token_type_id)
                    else:
                        with autocast():
                            output, _ = model(input_id, attention_mask_)
                if t == 0:
                    outputs = output.clone()
                else:
                    outputs = torch.cat([outputs, output], dim=0)
                break
        df = pd.DataFrame(outputs.detach().cpu().numpy())
        os.makedirs(args.result_save_dir, exist_ok=True)
        df.to_csv(opj(args.result_save_dir, f"{model_fn}.csv"), index=False)

if __name__ == "__main__":
    run_ensemble_soft_logit_each_save(args)