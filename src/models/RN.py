import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.MLP import MLP

class RelationNetwork(nn.Module):

    def __init__(self, object_dim, hidden_dims_g, output_dim_g, hidden_dims_f, output_dim_f, device, attentional=True):
        '''
        :param object_dim: Equal to LSTM hidden dim. Dimension of the single object to be taken into consideration from g.
        '''

        super(RelationNetwork, self).__init__()

        self.object_dim = object_dim
        self.query_dim = self.object_dim
        self.input_dim_g = 2 * self.object_dim + self.query_dim # g analyzes pairs of objects
        self.hidden_dims_g = hidden_dims_g
        self.output_dim_g = output_dim_g
        self.input_dim_f = self.output_dim_g
        self.hidden_dims_f = hidden_dims_f
        self.output_dim_f = output_dim_f
        self.device = device
        self.attentional = attentional

        self.g = MLP(self.input_dim_g, self.hidden_dims_g, self.output_dim_g, nonlinear=True)
        self.f = MLP(self.input_dim_f, self.hidden_dims_f, self.output_dim_f)

        self.query_expand = MLP(self.query_dim, self.hidden_dims_g, self.output_dim_g, nonlinear=True)


    def forward(self, x, q=None):
        '''
        :param x: (batch, n_features)
        :param q: query, optional.
        '''


        x_i = x.unsqueeze(0)
        x_i = x_i.repeat(x.size(0),1,1)
        x_j = x.unsqueeze(1)
        x_j = x_j.repeat(1,x.size(0),1)
        pair_concat = torch.cat((x_i,x_j), dim=2).view(-1, x.size(1)*2)
        if q is not None:
            pair_concat = torch.cat((pair_concat, q.unsqueeze(0).repeat(pair_concat.size(0),1)), dim=1)

        relations = self.g(pair_concat)

        if self.attentional:
            q_exp = self.query_expand(q)
            c = F.softmax(torch.mv(relations, q_exp), dim=0)
            relations = relations * c.view(-1,1)

        embedding = torch.sum(relations, dim=0) # (output_dim_g)

        out = self.f(embedding) # (output_dim_f)

        return out
