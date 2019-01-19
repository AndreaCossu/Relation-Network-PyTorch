import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, g=False):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.g = g

        self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dims[0])])

        for i in range(1,len(self.hidden_dims)):
            self.linears.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
        self.linears.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        self.activation = torch.tanh

    def forward(self, x):
        '''
        :param x: (n_features)
        '''

        for i in range(len(self.hidden_dims)):
            x = self.linears[i](x)
            x = self.activation(x)

        out = self.linears[-1](x)
        if self.g:
            out = self.activation(out)

        return out
