# Relation-Network-PyTorch
Implementation of Relation Network using PyTorch v1.0 (Python3). Original paper: https://arxiv.org/abs/1706.01427

## WORK IN PROGRESS

# Implementation details
This implementation tests the Relation Network model against the babi dataset, available at https://research.fb.com/downloads/babi/

# Prerequisites
* Download the babi dataset (20 tasks) and place it under `babi/` folder in the project root
* Create a folder `models` to store saved models
* Create a folder `plots` to store saved plots

# Launch me
The main script is `launchBabi.py`.
Run it with `python launchBabi.py [options]`.
Options are listed and explained with `python launchBabi.py --help`.
In particular you can select the babi tasks to train and test and if you want to read all the facts or only the marked-relevant facts for each question.

# Requirements
Run `pip install -r requirements.txt`
