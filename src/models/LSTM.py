import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, hidden_dim, batch_size, vocabulary_size, dim_embedding, layers, device):

        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.layers = layers
        
        #PADDING
        number_of_extra_simbols = 1 #1 por padding
        vocabulary_size = vocabulary_size + number_of_extra_simbols 
        
        self.embeddings = nn.Embedding(vocabulary_size, dim_embedding).to(self.device)

        self.lstm_q = nn.LSTM(dim_embedding, hidden_dim, num_layers=self.layers, batch_first = True)
        self.lstm_f = nn.LSTM(dim_embedding, hidden_dim, num_layers=self.layers, batch_first = True)

    def process_question(self, x, h):

        emb = self.embeddings(x) # B, L, D

        processed, h = self.lstm_q(emb, h) # B, L, H

        return processed, h

    def reset_hidden_state(self):
        # hidden is composed by hidden and cell state vectors
        h_q = (torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True))
        return h_q
