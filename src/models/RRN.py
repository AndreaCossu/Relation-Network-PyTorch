import torch
import torch.nn as nn
from torch.nn import LSTM
from src.models import MLP

class RRN(nn.Module):

    def __init__(self, adjacency_matrix, dim_input, message_dim, output_dim, f_dims, o_dims, device,  g_layers=1, edge_attribute_dim=0, single_output=False):
        '''
        :param adjacency_matrix: (N x N) tensor containing 0 or 1 representing graph adjacency matrix
        :param edge_attribute_dim: 0 if edges have no attributes, else an integer. Default 0.
        :param single_output: True if RRN emits only one output at a time, False if it emits as many outputs as units. Default False.
        '''

        super(RRN, self).__init__()

        self.adjacency_matrix = adjacency_matrix
        self.n_units = self.adjacency_matrix.size(0)
        self.dim_input = dim_input
        self.dim_hidden = dim_input
        self.message_dim = message_dim
        self.output_dim = output_dim

        self.device = device

        self.f_dims = f_dims
        self.o_dims = o_dims
        self.g_layers = g_layers

        self.edge_attribute_dim = 0 if self.edge_attributes is None else edge_attribute_dim
        self.single_output = single_output

        input_f_dim = 2 * self.dim_hidden + self.edge_attribute_dim
        self.f = MLP(input_f_dim, self.f_dims, self.message_dim)

        input_g_dim = self.dim_input + self.message_dim
        self.g = LSTM(input_g_dim, self.dim_hidden, num_layers=self.g_layers, batch_first=True)

        input_o_dim = (self.dim_hidden * self.n_units) if self.single_output else self.dim_hidden
        self.o = MLP(input_o_dim, self.o_dims, self.output_dim)

    def forward(self, x, hidden, h, edge_attribute=None):
        '''
        This can be called repeatedly after hidden states are set.

        :param x: inputs to the RRN nodes
        :param hidden: hidden states of RRN nodes
        :param h: hidden and cell states of g
        :param edge_attributes: (edge_attribute_dim) tensor containing edge attribute or None if edges have no attributes. Default None.
        '''


        messages = torch.zeros(self.n_units, self.n_units, self.message_dim, device=self.device)

        for i in range(len(self.n_units)):
            for j in range(len(self.n_units)):
                if self.adjacency_matrix[i,j] == 1:
                    if edge_attribute is None:
                        input_f = torch.cat((hidden[i], hidden[j]))
                    else:
                        input_f = torch.cat((hidden[i], hidden[j], edge_attribute))
                    messages[i,j] = self.f(input_f)

        # sum_messages[i] contains the sum of the messages incoming to node i
        sum_messages = torch.sum(messages, dim=0)

        input_g = torch.cat((x, sum_messages))

        out, h = self.g(input_g, h)

        hidden = out[:,-1,:].squeeze()

        if self.single_output:
            sum_hidden = torch.sum(hidden, dim=0)
            out = self.o(sum_hidden)
        else:
            out = self.o(hidden)

        return out, hidden, h

    def reset_g(self, b):
        # hidden is composed by hidden and cell state vectors
        h = (
            torch.zeros(self.g_layers, b, self.dim_hidden, device=self.device, requires_grad=True),
            torch.zeros(self.g_layers, b, self.dim_hidden, device=self.device, requires_grad=True)
            )
        return h
