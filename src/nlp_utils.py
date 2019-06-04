import torch
from nltk import word_tokenize
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from nltk.tokenize import RegexpTokenizer
import numpy as np


def vectorize_gqa(batch_question, batch_answer, dictionary_question , dictionary_answer, batch_size, device, MAX_QUESTION_LENGTH):
    """
    batch de 2 en este ejemplo:
    * question_batch = ["cual es el clima", "como estas"] -> [[5,17,34,54] [23,54]]
    * answer_ground_truth_batch = ["soleado", "super bien"] -> [[53], [32]]
    """
    question_v = []
    answers_v = torch.empty(batch_size, device=device).long()
    
    
    """
    a = []
    for i in range(100000):
        a.append(torch.rand(1, 100, 100)

    b = torch.Tensor(100000, 100, 100)
    torch.cat(a, out=b)
    """
    
    # <PADDING> Index: len(dictionary_question)
    padding_symbol_Index = len(dictionary_question)
    
    tokenizer = RegexpTokenizer(r'\w+')

    for question in batch_question:
        """
        from nltk.tokenize import RegexpTokenizer

        tokenizer = RegexpTokenizer(r'\w+')
        tokenizer.tokenize('Eighty-seven miles to go, yet.  Onward!')
        
        Output:
        ['Eighty', 'seven', 'miles', 'to', 'go', 'yet', 'Onward']
        """
        
        words = tokenizer.tokenize(question)
        q_v = [dictionary_question.index(words[i]) if i < len(words) else padding_symbol_Index for i in range(MAX_QUESTION_LENGTH)]
        # q_v = [dictionary_question.index(words[i]) for i in range(MAX_QUESTION_LENGTH) if i < len(words) else paddingIndex]
        question_v.append(q_v)
        
    
    question_tensor = torch.FloatTensor(question_v, device=device).long()
    
    for i, answer in enumerate(batch_answer):        
        answers_v[i] = dictionary_answer.index(answer)
    
    return question_tensor, answers_v
 

# def convert2tensor(x):
#     numpy_array = np.array(x)
#     x = torch.FloatTensor(x)
#     return x

def vectorize_babi(stories, dictionary, batch_size, device):
    '''
    :param stories: structure produced by read_babi function
    :param dictionary: list of words produced by read_babi function

    :return stories_v: the new stories structure. Each element of the list is a list containing:
        0) batch of questions (B, L)
        1) batch of answers (B)
        2) batch of facts (B, max_n_facts, max_L).
        3) batch of labels (B)
        4) batch of num_facts (number of facts inside each substory of the batch) (B)

    '''

    stories_v = []

    questions_v = []
    answers_v = torch.empty(batch_size, device=device).long()
    labels_v = torch.empty(batch_size, device=device).long()
    num_facts = []

    facts_v = []
    for i in range(len(stories)):
        q, a, facts, label = stories[i]

        q_v = torch.tensor([dictionary.index(el) for el in q], device=device).long()
        questions_v.append(q_v)

        answers_v[i % batch_size] = dictionary.index(a)

        labels_v[i % batch_size] = label
        num_facts.append(len(facts))
        single_facts = [ torch.tensor([dictionary.index(el) for el in fact], device=device).long() for fact in facts ]
        facts_v += single_facts


        if ((i+1) % batch_size) == 0:
            stories_v.append([])
            stories_v[-1].append(pad_sequence(questions_v, batch_first=True))
            stories_v[-1].append(answers_v)
            facts_v = pad_sequence(facts_v, batch_first=True)

            ff = []
            base = 0
            for el in num_facts:
                ff.append(facts_v[base:base+el])
                base += el

            stories_v[-1].append(pad_sequence(ff, batch_first=True))
            stories_v[-1].append(labels_v)
            stories_v[-1].append(num_facts)

            answers_v = torch.empty(batch_size, device=device).long()
            labels_v = torch.empty(batch_size, device=device).long()
            num_facts = []
            facts_v = []
            questions_v = []


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
    dictionary = ['PAD']
    stories = []

    for task in range(len(babi_tasks)):
        file = to_read[task]
        label = babi_tasks[task]

        with open(path_babi + file) as f:

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
                    else:
                        facts_substory = list(facts.values())
                        if len(facts_substory) > 20:
                            facts_substory = facts_substory[-20:]

                    labels.append(label)
                    stories.append([question_tokens, answer, facts_substory, label])

                else:
                    # fact

                    # update dictionary
                    for el in tokens:
                        if el not in dictionary:
                            dictionary.append(str(el))

                    facts[index] = tokens

    return stories, dictionary, labels
