import torch.nn as nn
import torch
from torch.nn.functional import one_hot

class LSTM(nn.Module):

    def __init__(self, hidden_dim, batch_size, vocabulary_size, dim_embedding, layers, device):

        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.layers = layers

        self.embeddings = nn.Embedding(vocabulary_size, dim_embedding).to(self.device)

        self.lstm_q = nn.LSTM(dim_embedding, hidden_dim, num_layers=self.layers, batch_first = True)
        self.lstm_f = nn.LSTM(dim_embedding, hidden_dim, num_layers=self.layers, batch_first = True)

    def process_query(self, x, h):
        '''
        :param x: (B, n_words_q)
        '''

        emb = self.embeddings(x.unsqueeze(0)) # (B, n_words_q, dim_emb)

        processed, h = self.lstm_q(emb, h) # (B, n_words_q, hidden_dim_q)

        return processed, h

    def process_facts(self, x, h):
        '''
        :param x: (n_facts, n_words_facts)
        '''

        emb = self.embeddings(x) # (n_facts, n_words_facts, dim_emb)

        processed, h = self.lstm_f(emb, h) # (n_facts, n_words_facts, hidden_dim_f)

        return processed, h

    def reset_hidden_state_query(self):
        h_q = (
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_q

    def reset_hidden_state_fact(self, num_facts):
        h_f = (
            torch.zeros(self.layers, num_facts, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, num_facts, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_f



class LSTM_noemb(nn.Module):

    def __init__(self, hidden_dim, num_classes, batch_size, vocabulary_size, layers, device):

        super(LSTM_noemb, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.num_classes = num_classes

        self.lstm_q = nn.LSTM(self.vocabulary_size, hidden_dim, num_layers=self.layers, batch_first = True)
        self.lstm_f = nn.LSTM(self.vocabulary_size, hidden_dim, num_layers=self.layers, batch_first = True)

    def process_query(self, x, h):
        '''
        :param x: (B, n_words_q)
        '''

        x = one_hot(x, self.num_classes) # (B, n_words_q, num_classes)
        processed, h = self.lstm_q(x, h) # (B, n_words_q, hidden_dim_q)

        return processed, h

    def process_facts(self, x, h):
        '''
        :param x: (n_facts, n_words_facts)
        '''

        x = one_hot(x, self.num_classes) # (B, n_words_q, num_classes)
        processed, h = self.lstm_f(x, h) # (n_facts, n_words_facts, hidden_dim_f)

        return processed, h

    def reset_hidden_state_query(self):
        h_q = (
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_q

    def reset_hidden_state_fact(self, num_facts):
        h_f = (
            torch.zeros(self.layers, num_facts, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, num_facts, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_f
