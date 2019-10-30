# Relation-Network-PyTorch
Implementation of Relation Network. Original paper: https://arxiv.org/abs/1706.01427

Implementation of Recurrent Relational Network. Original paper: https://arxiv.org/abs/1711.08028

This repository uses PyTorch v1.3 (Python3.7).

# Implementation details
This implementation tests the Relation Network model (RN) and the Recurrent Relational Network model (RRN) against the babi dataset, available at https://research.fb.com/downloads/babi/

# Weights and Biases
This repository uses Weights and Biases (W&B) to monitor experiments. You can create a free account on W&B (https://www.wandb.com/) or comment out the (few) lines starting with `wandb`. Without W&B, accuracy and loss plots will still be created and saved locally in the `results` folder.

# Prerequisites
* Run `pip install -r requirements.txt`
* Create a folder `babi`
* Download the babi dataset (20 tasks) already divided and preprocessed in train, validation and test at this link: https://drive.google.com/drive/folders/1HkO_w7hbxxWZV4nbGl0Y2w_AuSa4ktub?usp=sharing and put the downloaded folder inside `babi`.
* Download the word dictionaries at: https://drive.google.com/drive/folders/1Ktd4kL1FJBiY_R_HoqlicjmRemkq8xzC?usp=sharing and put the downloaded folder inside `babi`.
* Create a folder `results` to store experiment results (both plots and saved models)

# Train and test RN
* Model implementation is inside `src/models/RN.py`
* Train and test functions are inside `task/babi_task/rn/train.py`
* The main script is `launch_rn_babi.py`.
  * Run it with `python launch_rn_babi.py experiment_name [options]`.
  * Options are listed and explained with `python launch_rn_babi.py --help`.

To reproduce results execute `python launch_rn_babi.py test --learning_rate 1e-4 --batch_size 20 --epochs 50 ` and then check under `results/test` to see the results. If you want to do the final test on the test set instead of validation set, use `--test_on_test` option. The final accuracy on test set is (task: accuracy):
* 1: 0.973
* 2: 0.828
* 3: 0.786
* 4: 0.884
* 5: 0.957
* 6: 0.985
* 7: 0.962
* 8: 0.996
* 9: 0.979
* 10: 0.967
* 11: 0.958
* 12: 0.968
* 13: 0.973
* 14: 0.972
* 15: 0.97
* 16: 0.463
* 17: 0.545
* 18: 0.957
* 19: 0.483
* 20: 0.999


## Observations
* Batchify babi is essential to training performance, both in terms of convergence time and in terms of final accuracy.
* In order to batchify babi it is necessary to pad supporting facts both on #words and #facts dimensions.
* Relu activation dramatically improves accuracy, but only when using `batch size > 1`. If `batch size == 1` relu prevents learning, while tanh achieves `~74%` accuracy on the joint dataset.
* You can find the version with `batch size == 1` in the branch `no_batch`.

# Train and test RRN
* Model implementation is inside `src/models/RRN.py`
* Train and test functions are inside `task/babi_task/rrn/train.py`
* The main script is `launch_rrn_babi.py`.
  * Run it with `python launch_rrn_babi.py experiment_name [options]`.
  * Options are listed and explained with `python launch_rrn_babi.py --help`.

To reproduce results execute `python launch_rrn_babi.py test --cuda --epochs 500 --batch_size 512 --weight_decay 1e-5 --learning_rate 2e-4` and then check under `results/rrn/test` to see the results. If you want to do the final test on the test set instead of validation set, use `--test_on_test` option. The final accuracy on validation set is (task: accuracy):

* 1: 0.974
* 2: 0.736
* 3: 0.674
* 4: 1.0
* 5: 0.922
* 6: 0.953
* 7: 0.994
* 8: 0.975
* 9: 0.931
* 10: 0.855
* 11: 0.876
* 12: 0.939
* 13: 0.912
* 14: 0.846
* 15: 1.0
* 16: 0.454
* 17: 0.980
* 18: 0.995
* 19: 0.498  
* 20: 0.999
