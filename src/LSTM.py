import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, input_dim, num_layers, hidden_dim, batch_size, device):
        '''
        :param input_dim: dimension of the word embedding
        '''

        super(LSTM, self).__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.device = device
        self.dropout = 0.8 if self.num_layers > 1 else 0.0

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.dropout, batch_first=True).to(self.device)


    def forward(self, x, h):
        '''
        :param x: (batch, words, embedding_dim) input tensor
        '''

        processed, h = self.lstm(x, h)

        return processed, h

    def reset_hidden_state(self, b=None):
        if b is None:
        # hidden is composed by hidden and cell state vectors
            return (torch.randn(self.num_layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True),
                torch.randn(self.num_layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True)
                )
        else:
            return (torch.randn(self.num_layers, b, self.hidden_dim, device=self.device, requires_grad=True),
                torch.randn(self.num_layers, b, self.hidden_dim, device=self.device, requires_grad=True)
                )
