import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

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

    def forward(self, x):
        '''
        :param x: (n_features)
        '''

        for i in range(len(self.hidden_dims)):
            x = self.linears[i](x)
            x = F.relu(x)

        out = self.linears[-1](x)
        if self.g:
            out = F.relu(out)

        return out

class RelationNetwork(nn.Module):

    def __init__(self, object_dim, hidden_dims_g, output_dim_g, hidden_dims_f, output_dim_f, device, mode=4):
        '''
        :param object_dim: Equal to LSTM hidden dim. Dimension of the single object to be taken into consideration from g.
        :param mode: one of {0,1,2,3,4}. Each mode specifies a way of creating pairs of objects.
        0 creates pairs between each different object and ordering matters.
        1 creates pairs between each (even equal) object and ordering matters.
        2 creates pairs between each different object and ordering does not matter.
        3 creates pairs between each (even equal) object and ordering does not matter.
        4 creates pairs between each adjacent object following the order.
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
        self.mode = mode

        self.g = MLP(self.input_dim_g, self.hidden_dims_g, self.output_dim_g, g=True).to(self.device)
        self.f = MLP(self.input_dim_f, self.hidden_dims_f, self.output_dim_f).to(self.device)


    def _generate_pairs(self, x):
        '''
        :param x: (batch, n_features)
        :return pairs: list of pairs (tuples) of the form (oi, oj)
        '''

        if self.mode == 0:
            pairs = list(itertools.combinations(x, 2))
        elif self.mode == 1:
            pairs = list(itertools.combinations_with_replacement(x,2))
        elif self.mode == 2:
            pairs = list(itertools.permutations(x,2))
        elif self.mode == 3:
            pairs = list(itertools.product(x, repeat=2))
        else: # mode = 4
            a,b = itertools.tee(x) # creates two iterators on x
            next(b, None) # discard first element on the second iterator
            pairs = list(zip(a,b))

        return pairs

    def forward(self, x, q=None):
        '''
        :param x: (batch, n_features)
        :param q: query, optional.
        '''

        if x.size(0) == 1:
            x = x.squeeze()
            pairs = [ [x, torch.zeros_like(x, requires_grad=False, device=self.device)] ]
        else:
            pairs = self._generate_pairs(x)

        pair_concat = torch.empty(len(pairs), self.input_dim_g, requires_grad=False, device=self.device)

        for i in range(len(pairs)):
            pair = pairs[i]
            if q is not None:
                pair_concat[i,:] = torch.cat((pair[0], pair[1], q))
            else:
                pair_concat[i, :] = torch.cat((pair[0], pair[1]))

        relations = self.g(pair_concat)

        embedding = torch.sum(relations, dim=0) # (output_dim_g)

        out = self.f(embedding) # (output_dim_f)

        return out
