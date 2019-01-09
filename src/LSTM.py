import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, hidden_dim, batch_size, vocabulary_size, dim_embedding, device):

        super(LSTM, self).__init__()

        self.embeddings = nn.Embedding(vocabulary_size, dim_embedding)
        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim


        self.lstm_q = nn.LSTM(dim_embedding, hidden_dim, batch_first = True).to(self.device)
        self.lstm_f = nn.LSTM(dim_embedding, hidden_dim, batch_first = True).to(self.device)

    def process_query(self, x, h):

        emb = self.embeddings(x)
        processed, h = self.lstm_q(emb.unsqueeze(0), h)

        return processed, h

    def process_facts(self, x, h):

        emb = self.embeddings(x)

        processed, h = self.lstm_f(emb, h)

        return processed, h

    def reset_hidden_state(self, b):
        # hidden is composed by hidden and cell state vectors
        h_q = (
            torch.randn(1, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True),
            torch.randn(1, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True)
            )

        h_f = (
            torch.randn(1, b, self.hidden_dim, device=self.device, requires_grad=True),
            torch.randn(1, b, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_q, h_f
