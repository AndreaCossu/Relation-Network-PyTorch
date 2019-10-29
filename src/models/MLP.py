import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, relu=False, nonlinear=False, dropout=False):
        '''
        :param nonlinear: if False last layer is linear, if True is nonlinear. Default False.
        '''

        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.nonlinear = nonlinear
        self.use_dropout = dropout

        self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dims[0])])
        if self.use_dropout:
            self.dropouts = nn.ModuleList( [nn.Dropout(p=0.5) for _ in range(len(self.hidden_dims))] )
            # self.batchnorms = nn.ModuleList([ nn.BatchNorm1d(self.hidden_dims[i]) for i in range(len(self.hidden_dims)) ])

        for i in range(1,len(self.hidden_dims)):
            self.linears.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))
        self.linears.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        if not relu:
            self.activation = torch.tanh
        else:
            self.activation = torch.relu

    def forward(self, x):
        '''
        :param x: (n_features)
        '''

        x = self.linears[0](x)
        x = self.activation(x)
        '''
        if self.use_dropout:
            # x = self.batchnorms[0](x)
            x = self.dropouts[0](x)
        '''

        for i in range(1,len(self.hidden_dims)):
            x = self.linears[i](x)
            x = self.activation(x)
            if self.use_dropout and ((i==len(self.hidden_dims)-1) or (i==len(self.hidden_dims)-2)):
                # x = self.batchnorms[i](x)
                x = self.dropouts[i](x)

        out = self.linears[-1](x)
        if self.nonlinear:
            out = self.activation(out)

        return out
