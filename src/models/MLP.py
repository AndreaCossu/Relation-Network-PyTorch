import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dims, output_dim, drop_layers, nonlinear=False, drop_prob=0.5):
        '''
        :param nonlinear: if False last layer is linear, if True is nonlinear. Default False.
        drop_layers: list indicating witch hidden layers has dropout
        '''

        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.nonlinear = nonlinear
        self.drop_layers = drop_layers

        # -- Fist Layer MLP (get input) --
        self.linears = nn.ModuleList([nn.Linear(self.input_dim, self.hidden_dims[0])])
        
        # -- Middle Layers MLP --
        for i in range(1,len(self.hidden_dims)):
            self.linears.append(nn.Linear(self.hidden_dims[i-1], self.hidden_dims[i]))

        # -- Last MLP Layer --
        self.linears.append(nn.Linear(self.hidden_dims[-1], self.output_dim))

        # -- Dropout Layer --

        # Before:
        #self.dropouts = nn.ModuleList([nn.Dropout(p=0.5) for i in range(len(hidden_dims))])
        

        #self.dropouts = nn.ModuleList()
        self.dropouts = []
        for i in range(len(drop_layers)):
            if drop_layers[i]:
                self.dropouts.append(nn.Dropout(p=drop_prob))
            else:
                self.dropouts.append(None)
        
        # -- Relu --
        self.activation = nn.ReLU(inplace=True)
        #tanh
        #self.activation = torch.tanh


    def forward(self, x):
        '''
        :param x: (n_features)
        '''
        for i in range(0,len(self.hidden_dims)):
            x = self.linears[i](x)
            x = self.activation(x)
            if self.drop_layers[i]:
                x = self.dropouts[i](x)

        out = self.linears[-1](x)
        if self.nonlinear:
            out = self.activation(out)

        return out
