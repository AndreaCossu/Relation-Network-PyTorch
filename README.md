# Relation-Network-PyTorch

Original Repository: https://github.com/AndreaCossu/Relation-Network-PyTorch

Implementation of Relation Network. Original paper: https://arxiv.org/abs/1706.01427

Implementation of Recurrent Relational Network. Original paper: https://arxiv.org/abs/1711.08028

This repository uses PyTorch v1.0 (Python3).

## WORK IN PROGRESS

# Implementation details
This implementation tests the Relation Network model (RN) and the Recurrent Relational Network model (RRN) against the babi dataset, available at https://research.fb.com/downloads/babi/

# Prerequisites
* Download the babi dataset (20 tasks) and place it under `babi/` folder in the project root
* Create a folder `saved_models` to store saved models
* Create a folder `plots` to store saved plots

## nltk configurations
* Run following command in terminal: "/Applications/Python 3.7/Install Certificates.command"
(gives authorization for nltk)
* Run following command in python:
```python
import nltk
nltk.download('punkt')
```


# Train and test RN
* Model implementation is inside `src/models/RN.py`
* Train and test functions are inside `task/babi_task/rn/train.py`
* The main script is `launch_rn_babi.py`.
  * Run it with `python launch_rn_babi.py [options]`.
  * Options are listed and explained with `python launch_rn_babi.py --help`.
  * In particular you can select which babi tasks to train and test. You can choose to read all the facts or just the relevant ones.

# Train and test RRN
* Model implementation is inside `src/models/RRN.py`
* Train and test functions are inside `task/babi_task/rrn/train.py`
* The main script is `launch_rrn_babi.py`.
  * Run it with `python launch_rrn_babi.py [options]`.
  * Options are listed and explained with `python launch_rrn_babi.py --help`.
  * In particular you can select which babi tasks to train and test. You can choose to read all the facts or just the relevant ones.

# Requirements
Run `pip install -r requirements.txt`
