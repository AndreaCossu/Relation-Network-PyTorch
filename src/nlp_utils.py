import torch
from nltk import word_tokenize


def vectorize_babi(stories, dictionary, device):
    '''
    :param stories: structure produced by read_babi function
    :param dictionary: list of words produced by read_babi function

    :return stories_v: the new stories structure with torch.Tensor representing each sentence
                    by using word position in the dictionary.
    '''
    stories_v = []

    for q, a, facts in stories:
        q_v = torch.tensor([dictionary.index(el) for el in q], device=device).long()
        a_v = torch.tensor([dictionary.index(a)], device=device).long()
        facts_v = []
        for fact in facts:
            facts_v.append( torch.tensor([dictionary.index(el) for el in fact], device=device).long() )

        facts_padded = torch.nn.utils.rnn.pad_sequence(facts_v, batch_first=True)

        stories_v.append([q_v, a_v, facts_padded])

    return stories_v


def read_babi(path_babi, to_read, only_relevant=False):
    '''
    :param path_babi: absolute path to babi file to parse
    :param to_read: list of babi tasks filenames to parse
    :param only_relevant: if True returns only relevant facts for each question, else return all previous facts inside the story. Default False.

    :return stories: list of lists. Each sublist is a list containing:
            0) question - list of words
            1) answer - single word
            2) facts - list of list of words, one list for each fact

    :return dictionary: a list containing all the words in the babi files
    '''

    dictionary = ['PAD']
    stories = []
    for file in to_read:
        with open(path_babi + file) as f:

            for line in f:
                line = line.lower()
                tokens = word_tokenize(line)
                index, tokens = int(tokens[0]), tokens[1:]

                if index == 1:
                    # new story has started
                    facts = []


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

                    facts_substory = list(facts)

                    stories.append([question_tokens, answer, facts_substory])

                else:
                    # fact

                    # update dictionary
                    for el in tokens:
                        if el not in dictionary:
                            dictionary.append(str(el))

                    facts.append(tokens)

    return stories, dictionary
