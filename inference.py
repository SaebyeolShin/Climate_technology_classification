from glob import glob
from os.path import join as opj
import argparse

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--csv_dir", type=str, default="./csvs")
parser.add_argument("--final_submit_name", type=str, default="1_2_6_10.csv")
args = parser.parse_args()

if __name__=="__main__":
    csv_paths = glob(opj(args.csv_dir, "*.csv"))
    result = None
    sample_submission = pd.read_csv("./sample_submission.csv")
    for csv_path in csv_paths:
        csv = pd.read_csv(csv_path)

        if result is None:
            result = csv
        else:
            result += csv

    sample_submission["label"] = result.idxmax(axis=1)
    sample_submission.to_csv(args.final_submit_name, index=False)