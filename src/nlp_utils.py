import torch
import os
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence


def vectorize_babi(stories, dictionary, device):
    '''
    :param stories: structure produced by read_babi function
    :param dictionary: list of words produced by read_babi function

    :return stories_v: the new stories structure. Each element of the list is a list containing:
        0) question (L_q)
        1) answer (1)
        2) facts (n_facts, L_f).
        3) label (1)
        4) ordering (n_facts) relative facts ordering
    '''

    stories_v = []

    for q,a,facts,label,ordering in stories:

        q_v = torch.tensor( [dictionary.index(el) for el in q], device=device).long()
        a_v = torch.tensor(dictionary.index(a), device=device).long()
        l_v = torch.tensor(label, device=device).long()
        o_v = torch.tensor(ordering, device=device).float()
        f_v = [ torch.tensor([dictionary.index(el) for el in fact], device=device).long() for fact in facts]


        stories_v.append((q_v, a_v, pad_sequence(f_v, batch_first=True), l_v, o_v))

    return stories_v



def read_babi(path_babi, to_read, babi_tasks, only_relevant=False):
    '''
    :param path_babi: absolute path to babi file to parse
    :param to_read: list of babi tasks filenames to parse
    :param babi_tasks: list of ids of tasks to process
    :param only_relevant: if True returns only relevant facts for each question, else return all previous facts inside the story. Default False.

    :return stories: list of lists. Each sublist is a list containing:
            0) question - list of words
            1) answer - single word
            2) facts - list of list of words, one list for each fact
            3) task - id of the task of the story

    :return dictionary: a list containing all the words in the babi files

    :return labels: list of task label for each story in stories
    '''

    labels = []
    dictionary = []
    stories = []

    for file, label in zip(to_read, babi_tasks):

        with open(os.path.join(path_babi, file), 'r') as f:

            for line in f:
                line = line.lower()
                tokens = word_tokenize(line)
                index, tokens = int(tokens[0]), tokens[1:]

                if index == 1:
                    # new story has started
                    facts = {}


                if '?' in tokens:
                    # question found
                    question_index = tokens.index('?')
                    question_tokens = tokens[:question_index]
                    answer = tokens[question_index + 1]

                    for el in question_tokens:
                        if el not in dictionary:
                            dictionary.append(str(el))

                    if answer not in dictionary:
                        dictionary.append(str(answer))

                    if only_relevant:
                        support = list(map(int, tokens[question_index+2:]))
                        facts_substory = list([facts[idx] for idx in support])
                        facts_ordering = support
                    else:
                        if len(facts) <= 20:
                            facts_substory = list(facts.values())
                            facts_ordering = list(facts.keys())
                        else:
                            facts_substory = list(facts.values())[-20:]
                            facts_ordering = list(facts.keys())[-20:]

                    labels.append(label)

                    stories.append([question_tokens, answer, facts_substory, label, facts_ordering])

                else:
                    # fact

                    # update dictionary
                    for el in tokens:
                        if el not in dictionary:
                            dictionary.append(str(el))

                    facts[index] = tokens

    return stories, dictionary, labels
