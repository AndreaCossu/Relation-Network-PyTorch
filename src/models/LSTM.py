import torch.nn as nn
import torch

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

        emb = self.embeddings(x) # B, L, D

        processed, h = self.lstm_q(emb, h) # B, L, H

        return processed, h

    def process_facts(self, x, h):

        emb = self.embeddings(x) # B, n_facts, L, D

        processed, h = self.lstm_f(emb.view(-1,emb.size(2),emb.size(3)), h) # B*n_facts, L, H

        return processed.view(x.size(0), x.size(1), x.size(2), -1), h

    def reset_hidden_state(self, b):
        # hidden is composed by hidden and cell state vectors
        h_q = (
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True)
            )

        h_f = (
            torch.zeros(self.layers, b, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, b, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_q, h_f
