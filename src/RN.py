import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim

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

        return out

class RelationNetwork(nn.Module):

    def __init__(self, object_dim, hidden_dims_g, output_dim_g, hidden_dims_f, output_dim_f, device, query_dim=0, self_loop=True, ordered=True):
        '''
        :param object_dim: dimension of the single object to be taken into consideration from g
        :param self_loop: when True, during generation of pairs, it generates also the pair (o,o) for each object o. Default True.
        :param ordered: when True, during generation of pairs, (o1,o2) and (o2, o1) are considered different pairs. Default True.
        '''
        super(RelationNetwork, self).__init__()

        self.object_dim = object_dim
        self.query_dim = query_dim
        self.input_dim_g = 2 * self.object_dim + self.query_dim # g analyzes pairs of objects
        self.hidden_dims_g = hidden_dims_g
        self.output_dim_g = output_dim_g
        self.input_dim_f = self.output_dim_g
        self.hidden_dims_f = hidden_dims_f
        self.output_dim_f = output_dim_f
        self.device = device
        self.self_loop = self_loop
        self.ordered = ordered

        self.g = MLP(self.input_dim_g, self.hidden_dims_g, self.output_dim_g).to(self.device)
        self.f = MLP(self.input_dim_f, self.hidden_dims_f, self.output_dim_f).to(self.device)


    def _generate_pairs(self, x):
        '''
        :param x: (batch, n_features)
        :return pairs: list of pairs (tuples) of the form (oi, oj)
        '''

        if (not self.self_loop) and (not self.ordered):
            pairs = list(itertools.combinations(x, 2))
        elif self.self_loop and (not self.ordered):
            pairs = list(itertools.combinations_with_replacement(x,2))
        elif (not self.self_loop) and self.ordered:
            pairs = list(itertools.permutations(x,2))
        elif self.self_loop and self.ordered:
            pairs = list(itertools.product(x, repeat=2))

        return pairs

    def forward(self, x, q=None):
        '''
        :param x: (batch, n_features)
        :param q: query, optional.
        '''

        pairs = self._generate_pairs(x)
        relations = torch.zeros(len(pairs), self.output_dim_g, device=self.device, requires_grad=True)

        for i in range(len(pairs)):
            pair = pairs[i]
            pair_concat = torch.cat((pair[0], pair[1]))
            if q is not None:
                pair_concat = torch.cat((pair_concat, q))
            relations[i, :] = self.g(pair_concat)

        embedding = torch.sum(relations, dim=0) # (output_dim_g)

        out = self.f(embedding) # (output_dim_f)

        return out


if __name__ == '__main__':
    device = 'cpu'
    obj_dim = 8
    hidden_dims_g = [20, 40]
    hidden_dims_f = [10, 20]
    output_dim_g = 60
    query_dim = 8
    output_dim_f = 2
    batch_size = 6

    X = torch.randn(batch_size,obj_dim) # 6 objects with dimension 8
    q = torch.randn(query_dim)

    rn = RelationNetwork(obj_dim, hidden_dims_g, output_dim_g, hidden_dims_f, output_dim_f, device, query_dim)

    result = rn(X,q)

    print(result)
