import torch

def save_models(models, paths):
    '''
    :param models: iterable of models to save
    :param paths: iterable of saving paths
    '''
    for i in range(len(models)):
        torch.save(models[i].state_dict(), paths[i])

def load_models(models, paths):
    '''
    :param models: iterable of models to save
    :param paths: iterable of saving paths
    '''

    for i in range(len(models)):
        models[i].load_state_dict(torch.load(paths[i]))



saving_paths_models = [
    'models/lstm.pt',
    'models/rn.pt'
]

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
