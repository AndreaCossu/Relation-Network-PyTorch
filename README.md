# Relation-Network-PyTorch
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

# Train and test RN
* Model implementation is inside `src/models/RN.py`
* The main script is `launch_rn_babi.py`.
  * Run it with `python launch_rn_babi.py [options]`.
  * Options are listed and explained with `python launch_rn_babi.py --help`.
  * In particular you can select which babi tasks to train and test. You can only choose to read all the facts or just the relevant ones.

# Train and test RRN
* Model implementation is inside `src/models/RRN.py`

***I am working on this part right now.***

# Requirements
Run `pip install -r requirements.txt`
