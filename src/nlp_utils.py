'''
See https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
for a list of methods to efficiently load word embeddings in python
'''

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import os
import torch
from nltk import word_tokenize

class Embeddings():

    def __init__(self, path_raw_embeddings, embedding_name, dimension, device):
        '''
        :param path_raw_embeddings: absolute path to embeddings source file
        :param embedding_name: string representing embeddings file, e.g.: 'glove.6B.50d.txt'
        :param dimension: dimension of the embeddings

        :return glove_model: dict-like structure of embeddings, indexed by words
        '''

        path_gensim_embeddings = path_raw_embeddings.rsplit('/',1)[0] + "/" + "gensim_" + embedding_name + ".txt"

        self.device = device

        if not os.path.isfile(path_gensim_embeddings):
            print("Preloaded embeddings not found, creating...\n")
            glove2word2vec(glove_input_file=path_raw_embeddings, word2vec_output_file=path_gensim_embeddings)

        self.embeddings = KeyedVectors.load_word2vec_format(path_gensim_embeddings, binary=False)
        self.unk = torch.zeros(dimension, requires_grad=False, device=device)

    def get(self, key):
        '''
        :param key: string
        :return embedding: embedding of the key
        '''

        if key in self.embeddings:
            return torch.tensor(self.embeddings[key], device=self.device)
        else:
            return self.unk


def read_babi(path_babi):
    '''
    :param path_babi: absolute path to babi file to parse

    facts and questions are lists in which for each story there is an inner list.
    In facts each inner list contains tuples with:
        1) ID of the story (from 1 to number_stories)
        2) ID of the fact within story
        2) list of tokens of the fact
    In questions each inner list contains tuples with:
        1) ID of the story (from 1 to number_stories)
        2) IDs of the previous facts of the same story
        2) list of tokens of the questions
        3) single token representing answer
        4) list of supporting facts IDs
    '''

    facts = []
    questions = []
    n_stories = 0
    facts_ids = []

    with open(path_babi) as f:

        for line in f:
            line = line.lower()
            tokens = word_tokenize(line)
            index, tokens = tokens[0], tokens[1:]

            if index == '1':
                # new story has started
                facts.append([])
                questions.append([])
                n_stories += 1
                facts_ids = []

            if '?' in tokens:
                # question found
                question_index = tokens.index('?')
                question_tokens = tokens[:question_index]
                answer = tokens[question_index + 1]
                support = tokens[question_index+2:]


                questions[n_stories-1].append((n_stories, facts_ids, question_tokens, answer, support ))
            else:
                # fact
                facts_ids.append(index)
                facts[n_stories-1].append((n_stories, index, tokens))

    return facts, questions

def is_a_fact_id(x):
    try:
        int(x)
        return True
    except ValueError:
        return False
