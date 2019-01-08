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
        self.unk = torch.zeros(dimension, dtype=torch.float32, requires_grad=False, device=device)

    def get(self, key):
        '''
        :param key: string
        :return embedding: embedding of the key
        '''

        if key in self.embeddings:
            return torch.tensor(self.embeddings[key], dtype=torch.float32, device=self.device, requires_grad=False)
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


def get_question_encoding(q, emb_dim, lstm, h, device):
    '''
    :param q: single question structure
    :param emb_dim: dimension of word embedding
    :param lstm: LSTM to process query
    :param h: hidden state of the LSTM

    :return query_emb: LSTM final embedding of the query
    :return query_target: tensor representing word embedding of target answer
    :return h: final hidden state of LSTM
    '''

    words = q[2]
    query_target = q[3]
    query_tensor = torch.zeros(len(words), emb_dim, requires_grad=False, device=device)
    for i in range(len(words)):
        query_tensor[i,:] = words[i]
    query_tensor = query_tensor.unsqueeze(0)
    query_emb, h = lstm(query_tensor, h)
    query_emb = query_emb.squeeze()
    query_emb = query_emb[-1,:]

    return query_emb, query_target, h

def get_facts_encoding(story_f, hidden_dim, emb_dim, q_id, lstm, h, device):
    '''
    :param story_f: facts of the current story
    :param hidden_dim: hidden dimension of the LSTM
    :param emb_dim: dimension of word embedding
    :param q_id: ID of the current question
    :param lstm: LSTM to process facts
    :param h: hidden state of LSTM

    :return facts_emb: final embedding of LSTM of all facts before query
    :return h: final hidden state of the LSTM
    '''

    facts_emb = torch.zeros(len(story_f), hidden_dim, requires_grad=True, device=device)
    fact_tensor = torch.zeros(len(story_f), 30, emb_dim, requires_grad=False, device=device) # len(words)

    ff = 0
    while story_f[ff][0] < q_id: # check IDs of fact wrt ID of query
        fact = story_f[ff]

        words = fact[1]
        for i in range(len(words)):
            fact_tensor[ff,i,:] = words[i]

        ff += 1
        if ff == len(story_f):
            ff -= 1
            break


    facts_emb, h = lstm(fact_tensor, h)

    return facts_emb[:,-1,:], h
