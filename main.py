from src.RN import RelationNetwork
from src.nlp_utils import Embeddings, read_babi_list
from src.LSTM import LSTM
import torch
import argparse
import os
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--hidden_dims_g', nargs='+', type=int, default=[256, 256, 256, 256])
parser.add_argument('--hidden_dims_f', nargs='+', type=int, default=[256, 512, 159])
parser.add_argument('--hidden_dim_lstm', type=int, default=32)
parser.add_argument('--output_dim_f', type=int, default=50)
parser.add_argument('--output_dim_g', type=int, default=256)
parser.add_argument('--lstm_layers', type=int, default=1)

parser.add_argument('--obj_dim', type=int, default=32)
parser.add_argument('--query_dim', type=int, default=32)
parser.add_argument('--emb_dim', type=int, default=50)


parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=2e-4)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--batch_size_lstm', type=int, default=1)

parser.add_argument('--cuda', action="store_true")
parser.add_argument('--load', action="store_true")
parser.add_argument('--no_save', action="store_true")
parser.add_argument('--print_every', type=int, default=100)
args = parser.parse_args()

mode = 'cpu'
if args.cuda:
    if torch.cuda.is_available():
        print('Using ', torch.cuda.device_count() ,' GPU(s)')
        mode = 'cuda'
    else:
        print("WARNING: No GPU found. Using CPUs...")
else:
    print('Using 0 GPUs')

if args.emb_dim == 50:
    embeddings_file = "glove.6B.50d.txt"
elif args.emb_dim == 100:
    embeddings_file = "glove.6B.100d.txt"
elif args.emb_dim == 200:
    embeddings_file = "glove.6B.200d.txt"
elif args.emb_dim == 300:
    embeddings_file = "glove.6B.300d.txt"
else:
    print("Wrong embedding dimension!")


babi_file_train = "qa1_single-supporting-fact_train.txt"
babi_file_test = "qa1_single-supporting-fact_test.txt"

device = torch.device(mode)

cd = os.path.dirname(os.path.abspath(__file__))

path_raw_embeddings = cd + "/glove.6B/" + embeddings_file
print("Loading embeddings")
embeddings = Embeddings(path_raw_embeddings, embeddings_file, args.emb_dim, device)
print("Embeddings loaded!")

print("Reading babi")
path_babi = cd + "/babi/en-10k/" + babi_file_train
facts, questions = read_babi_list(path_babi, embeddings)
print("Done reading babi!")


lstm_facts = LSTM(args.emb_dim, args.lstm_layers, args.hidden_dim_lstm, args.batch_size_lstm, device)
lstm_query = LSTM(args.emb_dim, args.lstm_layers, args.hidden_dim_lstm, args.batch_size_lstm, device)

rn = RelationNetwork(args.obj_dim, args.hidden_dims_g, args.output_dim_g, args.hidden_dims_f, args.output_dim_f,
                     device, args.query_dim)

optimizer = torch.optim.Adam(chain(lstm_facts.parameters(), lstm_query.parameters(), rn.parameters()), args.learning_rate, weight_decay=args.weight_decay)

criterion = torch.nn.MSELoss()

lstm_facts.train()
lstm_query.train()
rn.train()

print("Start training")
for s in range(len(facts)):
    questions = questions[s]
    for q in questions:

        lstm_facts.zero_grad()
        lstm_query.zero_grad()
        rn.zero_grad()

        lstm_facts.reset_hidden_state()
        lstm_query.reset_hidden_state()

        words = q[2]
        query_target = q[3]
        query_tensor = torch.zeros(len(words), args.emb_dim, requires_grad=True)
        for i in range(len(words)):
            query_tensor[i,:] = words[i]
        query_tensor = query_tensor.unsqueeze(0)
        query_emb = lstm_query(query_tensor).squeeze()
        query_emb = query_emb[-1,:]

        story_f = facts[s]
        facts_emb = torch.zeros(len(story_f), args.hidden_dim_lstm, requires_grad=True)

        ff = 0
        while story_f[ff][0] < q[0]:
            fact = story_f[ff]

            words = fact[1]
            fact_tensor = torch.zeros(len(words), args.emb_dim, requires_grad=False)
            for i in range(len(words)):
                fact_tensor[i,:] = words[i]
            fact_tensor = fact_tensor.unsqueeze(0)
            result = lstm_facts(fact_tensor).squeeze()
            facts_emb[ff,:] = result[-1,:]

            ff += 1

        result = rn(facts_emb[:ff,:], query_emb)

        loss = criterion(result, query_target)

        loss.backward()

        optimizer.step()

        print(loss.item())

print("End training!")
