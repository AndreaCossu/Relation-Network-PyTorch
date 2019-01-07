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


def read_babi_list(path_babi, embedding):
    '''
    :param path_babi: absolute path to babi file to parse
    :param embedding: class to retrieve embeddings for words

    facts and questions are lists in which for each story there is an inner list.
    In facts each inner list contains tuples with:
        1) ID of the fact within story
        2) list of tokens embeddings of the fact
    In questions each inner list contains tuples with:
        1) ID of the query within the story
        2) IDs of the previous facts of the same story
        3) list of tokens embeddings of the questions
        4) single token embedding representing answer
        5) list of supporting facts IDs
    '''

    facts = []
    questions = []
    n_stories = 0
    facts_ids = []

    with open(path_babi) as f:

        for line in f:
            line = line.lower()
            tokens = word_tokenize(line)
            index, tokens = int(tokens[0]), tokens[1:]

            if index == 1:
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
                support = list(map(int, tokens[question_index+2:]))

                question_tokens = list(map(embedding.get, question_tokens))
                answer = embedding.get(answer)

                questions[n_stories-1].append((index, facts_ids, question_tokens, answer, support ))
            else:
                # fact
                facts_ids.append(index)
                tokens = list(map(embedding.get, tokens))
                facts[n_stories-1].append((index, tokens))

    return facts, questions
