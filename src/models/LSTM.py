import torch.nn as nn
import torch
from torch.nn.functional import one_hot
#import torch.nn.utils.rnn as torchrnn



class LSTM(nn.Module):

    def __init__(self, hidden_dim, batch_size, vocabulary_size, dim_embedding, layers, device, dropout=False, wave_penc=False):

        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.device = device
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.use_dropout = dropout
        self.wave_penc = wave_penc

        self.embeddings = nn.Embedding(vocabulary_size+1, dim_embedding).to(self.device)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=0.5)

        self.lstm_q = nn.LSTM(dim_embedding, hidden_dim, num_layers=self.layers, batch_first = True)
        self.lstm_f = nn.LSTM(dim_embedding, hidden_dim, num_layers=self.layers, batch_first = True)

    def process_query(self, x, h):
        '''
        :param x: (B, n_words_q)
        '''


        emb = self.embeddings(x) # (B, n_words_q, dim_emb)

        if self.use_dropout:
            emb = self.dropout(emb)

        _, h = self.lstm_q(emb, h) # (B, n_words_q, hidden_dim_q)

        return h[0].squeeze(), h

    def process_facts(self, x, h):
        '''
        :param x: (n_facts, n_words_facts)
        '''

        emb = self.embeddings(x) # (n_facts, n_words_facts, dim_emb)
        if self.use_dropout:
            emb = self.dropout(emb)

        _, h = self.lstm_f(emb, h) # (n_facts, n_words_facts, hidden_dim_f)

        processed = h[0].squeeze().view(self.batch_size,-1,self.hidden_dim)

        if not self.wave_penc:
            # simple oneofk without random noise
            #oneofk = torch.eye(20)[:processed.size(1)].repeat(self.batch_size,1,1).to(self.device)
            oneofk = self.one_of_k(processed.size())
            final = torch.cat( (processed, oneofk), dim=2) # add positional encoding in one-of-k (max 20 facts)
        else:
            final = processed + self.wave_positional_encoding(processed.size(0), processed.size(1))

        return final, h

    def one_of_k(self, size):
        offs = torch.randint(0,21,(size[0],)) # random offsets (0-20)
        eye = torch.eye(40)
        return torch.stack(  [eye[o:size[1]+o] for o in offs], dim=0 ).to(self.device)

    def wave_positional_encoding(self, batch, facts):
        '''
        Use positional encoding like in Transformer paper - Attention is all you need
        '''

        waves_penc = torch.empty(batch, facts).float()

        for i in range(waves_penc.size(1)):
            if i % 2 == 0:
                waves_penc[:,i] = torch.sin(torch.tensor(i / 10000.**( float((2*i)) / float(self.hidden_dim))))
            else:
                waves_penc[:,i] = torch.cos(torch.tensor(i / 10000.**( float((2*i)) / float(self.hidden_dim))))

        return waves_penc.unsqueeze(2).repeat(1,1,self.hidden_dim).to(self.device)

    def reset_hidden_state_query(self):
        h_q = (
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, self.batch_size, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_q

    def reset_hidden_state_fact(self, num_facts):
        h_f = (
            torch.zeros(self.layers, num_facts, self.hidden_dim, device=self.device, requires_grad=True),
            torch.zeros(self.layers, num_facts, self.hidden_dim, device=self.device, requires_grad=True)
            )
        return h_f
