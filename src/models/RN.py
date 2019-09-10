import torch
import torch.nn as nn
from src.models.MLP import MLP

class RelationNetwork(nn.Module):

    def __init__(self, object_dim, hidden_dims_g, output_dim_g, hidden_dims_f, output_dim_f, batch_size, device):
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
        self.batch_size = batch_size
        self.device = device

        self.g = MLP(self.input_dim_g, self.hidden_dims_g, self.output_dim_g, nonlinear=True)
        self.f = MLP(self.input_dim_f, self.hidden_dims_f, self.output_dim_f)


    def forward(self, x, q=None):
        '''
        :param x: (n_facts, hidden_dim_f)
        :param q: (hidden_dim_q) query, optional.
        '''

        x = x.unsqueeze(0)
        if q is not None:
            q = q.unsqueeze(0)

        n_facts = x.size(1)

        xi = x.repeat(1, n_facts, 1)
        xj = x.unsqueeze(2)
        xj = xj.repeat(1,1,n_facts,1).view(x.size(0),-1,x.size(2))
        if q is not None:
            q = q.unsqueeze(1)
            q = q.repeat(1,xi.size(1),1)
            pair_concat = torch.cat((xi,xj,q), dim=2) # (B, n_facts*n_facts, 2*hidden_dim_f+hidden_dim_q)
        else:
            pair_concat = torch.cat((xi,xj), dim=2) # (B, n_facts*n_facts, 2*hidden_dim_f)


        relations = self.g(pair_concat) # (B, n_facts*n_facts, hidden_dim_g)

        embedding = torch.sum(relations, dim=1) # (B, hidden_dim_g)

        out = self.f(embedding) # (B, hidden_dim_f)

        return out
