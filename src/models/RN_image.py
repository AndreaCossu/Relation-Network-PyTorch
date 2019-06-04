import torch
import torch.nn as nn
from src.models.MLP import MLP

class RelationNetwork(nn.Module):

    def __init__(self, object_dim, query_dim, hidden_dims_g, output_dim_g, drops_g, drop_prob_g, hidden_dims_f, output_dim_f, drops_f, drop_prob_f, batch_size, device):
        '''
        :param object_dim: Equal to LSTM hidden dim. Dimension of the single object to be taken into consideration from g.
        '''

        super(RelationNetwork, self).__init__()

        self.object_dim = object_dim
        self.query_dim = query_dim
        self.input_dim_g = 2 * self.object_dim + self.query_dim # g analyzes pairs of objects

        self.hidden_dims_g = hidden_dims_g
        self.output_dim_g = output_dim_g
        self.drops_g = drops_g
        self.drop_prob_g = drop_prob_g
        
        self.input_dim_f = self.output_dim_g
        self.hidden_dims_f = hidden_dims_f
        self.output_dim_f = output_dim_f
        self.drops_f = drops_f
        self.drop_prob_f = drop_prob_f

        self.batch_size = batch_size
        self.device = device

        self.g = MLP(self.input_dim_g, self.hidden_dims_g, self.output_dim_g, self.drops_g, nonlinear=True, drop_prob=self.drop_prob_g)
        self.f = MLP(self.input_dim_f, self.hidden_dims_f, self.output_dim_f, self.drops_f, nonlinear=True, drop_prob=self.drop_prob_f)
        # MLP: __init__(self, input_dim, hidden_dims, output_dim, dropout, nonlinear=False, drop_porb=0.5):



    def forward(self, x, q=None, objectNums=0):
        '''
        :param x: (batch, n_facts, length_fact)
        :param q: (batch, length_q) query, optional.
        :param objectNums: int how many object in the image
        '''
        # x = x[:objectNums]
        n_facts = x.size(1)
        
        
        """
        
        print(x.shape)
        print(x[0])
        print(x[0].type)
        print("___________")
        print(x[0][0])
        print(x[0][0].type)
        print("___________")        
        print(x[0][0][0])
        print(x[0][0][0].type)
        
        """

        xi = x.repeat(1, n_facts, 1)
        xj = x.unsqueeze(2)
        xj = xj.repeat(1,1,n_facts,1).view(x.size(0),-1,x.size(2))
        if q is not None:
            q = q.unsqueeze(1)
            q = q.repeat(1,xi.size(1),1)
            pair_concat = torch.cat((xi.long(),xj.long(),q.long()), dim=2).long()
        else:
            pair_concat = torch.cat((xi,xj), dim=2).long()


        relations = self.g(pair_concat)

        embedding = torch.sum(relations, dim=1) # (output_dim_g)

        out = self.f(embedding) # (output_dim_f)

        return out
