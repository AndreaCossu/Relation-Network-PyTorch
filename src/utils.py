import torch
from sklearn.model_selection import train_test_split

def get_answer(output, target, vocabulary=None):
    '''
    :param output: tensor representing output of the model
    :param vocabulary: vocabulary of all the words

    :return correct: 1 if the answer is correct, 0 otherwise
    :return answer: if vocabulary is not None, the string representation of the output, else None
    '''

    idx = torch.argmax(output)
    correct = (idx == target).item()

    if vocabulary is not None:
        answer = vocabulary[idx.item()]
    else:
        answer = None

    assert(correct == 1 or correct == 0)
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


def save_models(models, path):
    '''
    :param models: iterable of (models to save, name)
    :param paths: saving path
    '''
    dict_m = {}
    for model, name in models:
        dict_m[name] = model.state_dict()

    torch.save(dict_m, path)

def load_models(models, path):
    '''
    :param models: iterable of models to save
    :param paths: iterable of saving paths
    '''

    checkpoint = torch.load(path)
    for model, name in models:
        model.load_state_dict(checkpoint[name])



saving_path_models = 'models/models.tar'
names_models = ['LSTM', 'RN']

files_names_test = [
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

files_names_train = [
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
