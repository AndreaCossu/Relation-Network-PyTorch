from src.RN import RelationNetwork
from src.nlp_utils import read_babi_list, get_question_encoding, get_facts_encoding
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


babi_file_train = "qa1_single-supporting-fact_train.txt"
babi_file_test = "qa1_single-supporting-fact_test.txt"

device = torch.device(mode)

cd = os.path.dirname(os.path.abspath(__file__))

'''
if not args.train_embeddings:
    path_raw_embeddings = cd + "/glove.6B/" + embeddings_file
    print("Loading embeddings")
    embeddings = Embeddings(path_raw_embeddings, embeddings_file, args.emb_dim, device)
    print("Embeddings loaded!")
else:
    print("Train embeddings from scratch")
'''

print("Reading babi")
path_babi = cd + "/babi/en-10k/" + babi_file_train

'''
if not args.train_embeddings:
    facts, questions = read_babi_list(path_babi, embeddings)
    dict_size = len(embeddings.dictionary)
'''

facts, questions, dictionary = read_babi_list(path_babi)
dict_size = len(dictionary)
print("Dictionary size: ", dict_size)
print("Done reading babi!")

lstm = LSTM(args.hidden_dim_lstm, args.batch_size_lstm, dict_size, args.emb_dim, device)

rn = RelationNetwork(args.obj_dim, args.hidden_dims_g, args.output_dim_g, args.hidden_dims_f, dict_size,
                     device, args.query_dim)

optimizer = torch.optim.Adam(chain(lstm.parameters(), rn.parameters()), args.learning_rate, weight_decay=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss()

lstm.train()
rn.train()

losses = []
print("Start training")
for s in range(len(facts)): # for each story
    story_q = questions[s]
    for q in story_q: # for each question in the current story

        lstm.zero_grad()
        rn.zero_grad()

        h_q, h_f = lstm.reset_hidden_state(len(facts[s]))

        query_emb, h_q = get_question_encoding(q, args.emb_dim, lstm, h_q, device)

        query_target = torch.tensor([dictionary.index(q[3])], requires_grad=False, device=device).long()
        story_f = facts[s]

        facts_emb, h_f = get_facts_encoding(story_f, args.hidden_dim_lstm, args.emb_dim, q[0], lstm, h_f, device)

        rr = rn(facts_emb, query_emb)

        loss = criterion(rr.unsqueeze(0), query_target)

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

print("End training!")


import matplotlib.pyplot as plt

plt.plot(range(len(losses)), losses)
plt.show()
