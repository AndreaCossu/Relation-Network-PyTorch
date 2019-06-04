#%%
import torch
f1 = [11, 12, 13, 14]
f2 = [21, 22, 23, 24]
f3 = [31, 32, 33, 34]
f4 = [41, 42, 43, 44]
f5 = [51, 52, 53, 54]
f6 = [61, 62, 63, 64]

batch_1 = [f1, f2, f3]
batch_2 = [f4, f5, f6]
x = [
        batch_1,
        batch_2,
    ]
    
x = torch.Tensor(x)

print(x.size())

def forward(x, q=None):
        '''
        :param x: (batch, n_facts, length_fact)
        :param q: (batch, length_q) query, optional.
        '''

        # dummy: [[f1, f2, f3],
        #         [f1, f2, f3],
        #         [f1, f2, f3],
        #         [f1, f2, f3]]
        # batch: 4, facts: 3, length: 2

        n_facts = x.size(1)

        xi = x.repeat(1, n_facts, 1)
        print("xi:")
        print(xi)
        xj = x.unsqueeze(2)
        print("xj:")
        print(xj)
        xj = xj.repeat(1, 1, n_facts, 1).view(x.size(0), -1, x.size(2))
        print("xj:")
        print(xj)
forward(x)
