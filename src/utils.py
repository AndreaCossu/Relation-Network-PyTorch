import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import random
import os
import csv
import pickle
import wandb
from torch.utils.data import Dataset

def plot_results(folder, avg_train_losses, val_losses, avg_train_accuracies, val_accuracies):
    plt.figure()
    plt.plot(range(len(avg_train_losses)), avg_train_losses, 'b', label='train')
    plt.plot(range(len(val_losses)), val_losses, 'r', label='val')
    plt.legend(loc='best')

    plt.savefig(os.path.join(folder, 'loss.png'))

    plt.figure()
    plt.plot(range(len(avg_train_accuracies)), avg_train_accuracies, 'b', label='train')
    plt.plot(range(len(val_accuracies)), val_accuracies, 'r', label='val')
    plt.legend(loc='best')

    plt.savefig(os.path.join(folder, 'accuracy.png'))

class BabiDataset(Dataset):
    """Babi Dataset """

    def __init__(self, stories):

        self.stories = stories

    def __len__(self):
        return len(self.stories)

    def __getitem__(self, idx):
        question, answer, facts, label, ordering = self.stories[idx]


        return (question, answer, facts, label, ordering)

def batchify(data_batch):
    '''
    Custom collate_fn for dataset

    It is not possible to batchify facts since there is a different value
    for both dimension (different number of facts and different number of words per fact).
    It is not convenient to batchify ordering since it has to be concatenated to
    each fact for each question.
    '''

    if len(data_batch) == 1: # if batch_size == 1
        return data_batch[0]

    q_s = []
    a_s = []
    f_s = []
    l_s = []
    o_s = []

    for el in data_batch:
        q_s.append(el[0])
        a_s.append(el[1])
        f_s.append(el[2])
        l_s.append(el[3])
        o_s.append(el[4])

    # lengths is used only for pack sequence with LSTM
    #lengths_f = torch.tensor([el.size(0) for el in f_s]).long() # number of facts
    #lengths_q = torch.tensor([el.size(0) for el in q_s]).long() # number of words

    rows, columns = max([ el.size(0) for el in f_s] ), max([ el.size(1) for el in f_s] )
    ff = torch.ones(len(f_s), rows, columns, dtype=torch.long)*157 # len(dictionary) as pad value
    for i, t in enumerate(f_s):
        r, c = t.size(0), t.size(1)
        ff[i, :r, :c] = t

    return ( pad_sequence(q_s, batch_first=True, padding_value=157), torch.stack(a_s, dim=0), ff.view(-1, ff.size(2)), torch.stack(l_s, dim=0), o_s)

def save_stories(stories, valid, name):
    if valid:
        with open(os.path.join(saving_stories_valid,name+'.data'), 'wb') as f:
            pickle.dump(stories, f)
    else:
        with open(os.path.join(saving_stories_not_valid,name+'.data'), 'wb') as f:
            pickle.dump(stories, f)

def load_stories(valid, name):
    if valid:
        with open(os.path.join(saving_stories_valid,name+'.data'), 'rb') as f:
            stories = pickle.load(f)
    else:
        with open(os.path.join(saving_stories_not_valid,name+'.data'), 'rb') as f:
            stories = pickle.load(f)

    return stories

def save_dict(dictionary, valid):
    if valid:
        with open(saving_path_dict_valid, 'wb') as f:
            pickle.dump(dictionary, f)
        with open(saving_path_dict_valid_plain,'w') as f:
            for item in dictionary:
                f.write("%s\n" % item)
    else:
        with open(saving_path_dict_not_valid, 'wb') as f:
            pickle.dump(dictionary, f)
        with open(saving_path_dict_not_valid_plain,'w') as f:
            for item in dictionary:
                f.write("%s\n" % item)


def load_dict(valid):
    if valid:
        with open(saving_path_dict_valid, 'rb') as f:
            dictionary = pickle.load(f)
    else:
        with open(saving_path_dict_not_valid, 'rb') as f:
            dictionary = pickle.load(f)

    return dictionary

def get_run_folder(dest):
    default=results_folder

    target = os.path.join(default, dest)
    if not os.path.isdir(target):
        try:
            os.makedirs(target)

        except OSError:
            print("Error when creating experiment folder")
            target = default

    return target

def write_test(folder, losses, accs):
    with open(os.path.join(folder, 'test_accs.csv'), 'w') as f:
        w = csv.writer(f)
        for key, val in accs.items():
            w.writerow([key, val])
    with open(os.path.join(folder, 'test_losses.csv'), 'w') as f:
        w = csv.writer(f)
        for key, val in losses.items():
            w.writerow([key, val])


def get_answer(output, target, vocabulary=None):
    '''
    :param output: tensor representing output of the model
    :param vocabulary: vocabulary of all the words

    :return correct: average accuracy
    :return answer: if vocabulary is not None, the string representation of the outputs, else None
    '''

    with torch.no_grad():
        idx = torch.argmax(output, dim=1)
        correct = (idx == target).sum().item()
        correct /= float(output.size(0))

        if vocabulary is not None:
            answer = [vocabulary[id.item()] for id in idx]
        else:
            answer = None

        return correct, answer

def split_train_validation(stories, labels, perc_validation=0.2, shuffle=True):
    '''
    :param stories: stories structure already vectorized
    :param labels: list of task ids of the stories

    :return train_stories: 100*(1-perc_validation)% of stories
    :return validation_stories: 100*perc_validation% stories
    '''

    # stratify maintains the same percentage of babi tasks for train and validation
    train_stories, validation_stories = train_test_split(stories, test_size=perc_validation, shuffle=shuffle, stratify=labels)

    return train_stories, validation_stories


def random_idx_gen(start,end):
    indices = list(range(start,end))

    while True:
        random.shuffle(indices)
        for el in indices:
            yield el



def save_models(models, result_folder, path):
    '''
    :param models: iterable of (models to save, name)
    :param paths: saving path
    '''
    dict_m = {}
    for model, name in models:
        dict_m[name] = model.state_dict()

    torch.save(dict_m, os.path.join(result_folder, path))


def load_models(models, result_folder, path):
    '''
    :param models: iterable of models to save
    :param paths: iterable of saving paths
    '''

    checkpoint = torch.load(os.path.join(result_folder, path))
    for model, name in models:
        model.load_state_dict(checkpoint[name])


saving_path_dict_valid = 'babi/dicts/dict_20_en-valid-10k.data'
saving_path_dict_not_valid = 'babi/dicts/dict_20_en-10k.data'

saving_path_dict_valid_plain = 'babi/dicts/dict_20_en-valid-10k.txt'
saving_path_dict_not_valid_plain = 'babi/dicts/dict_20_en-10k.txt'

saving_stories_valid  = 'babi/vectorized_en-10k/'
saving_stories_not_valid  = 'babi/vectorized_en-valid-10k/'

results_folder = 'results'
saving_path_rn = 'rn.tar'
saving_path_rrn = 'rrn.tar'
names_models = ['LSTM', 'RN', 'RRN', 'MLP']

files_names_train_en_valid = [
    'qa1_train.txt',
    'qa2_train.txt',
    'qa3_train.txt',
    'qa4_train.txt',
    'qa5_train.txt',
    'qa6_train.txt',
    'qa7_train.txt',
    'qa8_train.txt',
    'qa9_train.txt',
    'qa10_train.txt',
    'qa11_train.txt',
    'qa12_train.txt',
    'qa13_train.txt',
    'qa14_train.txt',
    'qa15_train.txt',
    'qa16_train.txt',
    'qa17_train.txt',
    'qa18_train.txt',
    'qa19_train.txt',
    'qa20_train.txt'
]

files_names_val_en_valid = [
    'qa1_valid.txt',
    'qa2_valid.txt',
    'qa3_valid.txt',
    'qa4_valid.txt',
    'qa5_valid.txt',
    'qa6_valid.txt',
    'qa7_valid.txt',
    'qa8_valid.txt',
    'qa9_valid.txt',
    'qa10_valid.txt',
    'qa11_valid.txt',
    'qa12_valid.txt',
    'qa13_valid.txt',
    'qa14_valid.txt',
    'qa15_valid.txt',
    'qa16_valid.txt',
    'qa17_valid.txt',
    'qa18_valid.txt',
    'qa19_valid.txt',
    'qa20_valid.txt'
]

files_names_test_en_valid = [
    'qa1_test.txt',
    'qa2_test.txt',
    'qa3_test.txt',
    'qa4_test.txt',
    'qa5_test.txt',
    'qa6_test.txt',
    'qa7_test.txt',
    'qa8_test.txt',
    'qa9_test.txt',
    'qa10_test.txt',
    'qa11_test.txt',
    'qa12_test.txt',
    'qa13_test.txt',
    'qa14_test.txt',
    'qa15_test.txt',
    'qa16_test.txt',
    'qa17_test.txt',
    'qa18_test.txt',
    'qa19_test.txt',
    'qa20_test.txt'
]


files_names_test_en = [
    'qa1_single-supporting-fact_test.txt',
    'qa2_two-supporting-facts_test.txt',
    'qa3_three-supporting-facts_test.txt',
    'qa4_two-arg-relations_test.txt',
    'qa5_three-arg-relations_test.txt',
    'qa6_yes-no-questions_test.txt',
    'qa7_counting_test.txt',
    'qa8_lists-sets_test.txt',
    'qa9_simple-negation_test.txt',
    'qa10_indefinite-knowledge_test.txt',
    'qa11_basic-coreference_test.txt',
    'qa12_conjunction_test.txt',
    'qa13_compound-coreference_test.txt',
    'qa14_time-reasoning_test.txt',
    'qa15_basic-deduction_test.txt',
    'qa16_basic-induction_test.txt',
    'qa17_positional-reasoning_test.txt',
    'qa18_size-reasoning_test.txt',
    'qa19_path-finding_test.txt',
    'qa20_agents-motivations_test.txt',
]

files_names_train_en = [
    'qa1_single-supporting-fact_train.txt',
    'qa2_two-supporting-facts_train.txt',
    'qa3_three-supporting-facts_train.txt',
    'qa4_two-arg-relations_train.txt',
    'qa5_three-arg-relations_train.txt',
    'qa6_yes-no-questions_train.txt',
    'qa7_counting_train.txt',
    'qa8_lists-sets_train.txt',
    'qa9_simple-negation_train.txt',
    'qa10_indefinite-knowledge_train.txt',
    'qa11_basic-coreference_train.txt',
    'qa12_conjunction_train.txt',
    'qa13_compound-coreference_train.txt',
    'qa14_time-reasoning_train.txt',
    'qa15_basic-deduction_train.txt',
    'qa16_basic-induction_train.txt',
    'qa17_positional-reasoning_train.txt',
    'qa18_size-reasoning_train.txt',
    'qa19_path-finding_train.txt',
    'qa20_agents-motivations_train.txt'
]
