import torch
import torch.nn as nn
from torch.nn import LSTM
from src.models.MLP import MLP

class RRN(nn.Module):

    def __init__(self, input_dim_mlp, hidden_dims_mlp, dim_hidden, message_dim, output_dim, f_dims, o_dims, device, batch_size, g_layers=1, edge_attribute_dim=0, single_output=False, tanh=False, dropout=False):

        super(RRN, self).__init__()

        self.dim_hidden = dim_hidden
        self.dim_input = dim_hidden
        self.message_dim = message_dim
        self.output_dim = output_dim
        self.batch_size = batch_size

        self.device = device

        self.f_dims = f_dims
        self.o_dims = o_dims
        self.g_layers = g_layers

        self.edge_attribute_dim = edge_attribute_dim
        self.single_output = single_output

        input_f_dim = 2 * self.dim_hidden + self.edge_attribute_dim
        self.f = MLP(input_f_dim, self.f_dims, self.message_dim, tanh=tanh, dropout=dropout)

        input_gmlp_dim = self.dim_input + self.message_dim
        output_gmlp_dim = 128
        self.g_mlp = MLP(input_gmlp_dim, self.f_dims, output_gmlp_dim, tanh=tanh, dropout=dropout)
        self.g = LSTM(output_gmlp_dim, self.dim_hidden, num_layers=self.g_layers, batch_first=True)

        input_o_dim = self.dim_hidden
        self.o = MLP(input_o_dim, self.o_dims, self.output_dim, tanh=tanh, dropout=dropout)

        self.input_mlp = MLP(input_dim_mlp, hidden_dims_mlp, dim_hidden, tanh=tanh, nonlinear=False, dropout=dropout)

        self.dropout_layer = nn.Dropout(p=0.5)

    def process_input(self, x):
        return self.input_mlp(x)


    def forward(self, x, hidden, h, edge_attribute=None):

        n_facts = hidden.size(1)

        hi = hidden.repeat(1, n_facts, 1)
        hj = hidden.unsqueeze(2)
        hj = hj.repeat(1,1,n_facts,1).view(hidden.size(0),-1,hidden.size(2))
        if edge_attribute is not None:
            ea = edge_attribute.unsqueeze(1)
            ea = ea.repeat(1,hi.size(1),1)
            input_f = torch.cat((hj,hi,ea), dim=2)
        else:
            input_f = torch.cat((hi,hj), dim=2)

        messages = self.f(input_f.view(-1, input_f.size(2)))
        messages = self.dropout_layer(messages)
        messages = messages.view(hidden.size(0),hidden.size(1),hidden.size(1), self.message_dim)
        # sum_messages[i] contains the sum of the messages incoming to node i
        '''
        Sum over dim=2 implies that each row (message direction) is source node -> target node.
        Sum over dim=1 implies that each row (message direction) is source node <- target node.
        '''
        sum_messages = torch.sum(messages, dim=2) # B, N_facts, Message_dim
        input_g_mlp = torch.cat((x, sum_messages), dim=2)
        input_g = self.g_mlp(input_g_mlp.view(-1, input_g_mlp.size(2)))

        # the following version use recurrent network over facts (time dimension is 1 = #facts)
        #out_g, h = self.g(input_g.view(input_g_mlp.size(0), input_g_mlp.size(1), -1), h)

        # the following version user recurrent network over time steps (time=LEARNING STEPS)
        out_g, h = self.g(input_g.unsqueeze(1), h)
        out_g = out_g.squeeze(1).view(input_g_mlp.size(0), input_g_mlp.size(1), -1)

        if self.single_output:
            sum_hidden = torch.sum(out_g, dim=1)
            out = self.o(sum_hidden)
        else:
            out = self.o(hidden)

        return out, out_g, h

    def reset_g(self, b):
        # hidden is composed by hidden and cell state vectors
        h = (
            torch.zeros(self.g_layers, b, self.dim_hidden, device=self.device, requires_grad=True),
            torch.zeros(self.g_layers, b, self.dim_hidden, device=self.device, requires_grad=True)
            )
        return h
