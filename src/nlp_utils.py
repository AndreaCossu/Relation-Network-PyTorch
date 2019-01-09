'''
See https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
for a list of methods to efficiently load word embeddings in python
'''

import torch
from nltk import word_tokenize


def read_babi_list(path_babi):
    '''
    :param path_babi: absolute path to babi file to parse

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

    dictionary = []
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

                for el in question_tokens:
                    if el not in dictionary:
                        dictionary.append(str(el))

                if answer not in dictionary:
                    dictionary.append(str(answer))

                question_tokens = list(map(dictionary.index, question_tokens))

                questions[n_stories-1].append((index, facts_ids, question_tokens, answer, support ))
            else:
                # fact
                facts_ids.append(index)
                for el in tokens:
                    if el not in dictionary:
                        dictionary.append(str(el))
                tokens = list(map(dictionary.index, tokens))
                facts[n_stories-1].append((index, tokens))

    return facts, questions, dictionary


def get_question_encoding(q, emb_dim, lstm, h, device):
    '''
    :param q: single question structure
    :param emb_dim: dimension of word embedding
    :param lstm: LSTM to process query
    :param h: hidden state of the LSTM

    :return query_emb: LSTM final embedding of the query
    :return h: final hidden state of LSTM
    '''

    query_tensor = torch.tensor(q[2], device=device).long()

    query_emb, h = lstm.process_query(query_tensor, h)
    query_emb = query_emb.squeeze()
    query_emb = query_emb[-1,:]

    return query_emb, h

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

    max_len_fact = 10
    fact_tensor = torch.zeros(len(story_f), max_len_fact, requires_grad=False, device=device).long() # len(words)

    for ff in range(len(story_f)):
        if story_f[ff][0] > q_id: # check IDs of fact wrt ID of query
            break

        fact_tensor[ff, :len(story_f[ff][1])] = torch.tensor(story_f[ff][1], device=device).long()


    facts_emb, h = lstm.process_facts(fact_tensor, h)

    return facts_emb[:,-1,:], h
