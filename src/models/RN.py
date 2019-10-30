import torch
import torch.nn as nn
from src.models.MLP import MLP

class RelationNetwork(nn.Module):

    def __init__(self, object_dim, hidden_dims_g, output_dim_g, hidden_dims_f, output_dim_f, dropout, tanh, batch_size, wave_penc, device):

        super(RelationNetwork, self).__init__()

        if not wave_penc:
            self.object_dim = object_dim + 40 # 40 is the length of the one-of-k positional encoding of max 20 facts
        else:
            self.object_dim = object_dim

        self.query_dim = object_dim
        self.input_dim_g = 2 * self.object_dim + self.query_dim # g analyzes pairs of objects
        self.hidden_dims_g = hidden_dims_g
        self.output_dim_g = output_dim_g
        self.input_dim_f = self.output_dim_g
        self.hidden_dims_f = hidden_dims_f
        self.output_dim_f = output_dim_f
        self.batch_size = batch_size
        self.device = device

        self.g = MLP(self.input_dim_g, self.hidden_dims_g, self.output_dim_g, tanh=tanh, nonlinear=True, dropout=dropout)
        self.f = MLP(self.input_dim_f, self.hidden_dims_f, self.output_dim_f, tanh=tanh, dropout=dropout)


    def forward(self, x, q=None):


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

        relations = self.g(pair_concat.view(-1, pair_concat.size(2))) # (n_facts*n_facts, hidden_dim_g)
        relations = relations.view(pair_concat.size(0), pair_concat.size(1), self.output_dim_g)

        embedding = torch.sum(relations, dim=1) # (hidden_dim_g)

        out = self.f(embedding) # (hidden_dim_f)

        return out
