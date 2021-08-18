#! /bin/bash

python train_s.py && python train_j.py && python inference_s.py && python inference_j.py && python inference.py
