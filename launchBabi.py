from src.RN import RelationNetwork
from src.nlp_utils import read_babi, vectorize_babi
from src.LSTM import LSTM
import torch
import argparse
import os
from itertools import chain

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)

parser.add_argument('--hidden_dims_g', nargs='+', type=int, default=[256, 256, 256, 256])
parser.add_argument('--hidden_dims_f', nargs='+', type=int, default=[256, 512, 159])
parser.add_argument('--hidden_dim_lstm', type=int, default=32)
parser.add_argument('--output_dim_g', type=int, default=256)
parser.add_argument('--lstm_layers', type=int, default=1)

parser.add_argument('--obj_dim', type=int, default=32)
parser.add_argument('--query_dim', type=int, default=32)
parser.add_argument('--emb_dim', type=int, default=50)


parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=2e-5)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--batch_size_lstm', type=int, default=1)

parser.add_argument('--cuda', action="store_true")
parser.add_argument('--load', action="store_true")
parser.add_argument('--no_save', action="store_true")
parser.add_argument('--print_every', type=int, default=500)
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

print("Reading babi")
path_babi = cd + "/babi/en-10k/" + babi_file_train

stories, dictionary = read_babi(path_babi)
stories = vectorize_babi(stories, dictionary, device)
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

accuracy = 0
losses = []
avg_losses = []
print("Start training")
for i in range(args.epochs):
    s = 1
    for question, answer, facts in stories: # for each story

        lstm.zero_grad()
        rn.zero_grad()

        h_q, h_f = lstm.reset_hidden_state(facts.size(0))

        question_emb, h_q = lstm.process_query(question, h_q)
        question_emb = question_emb.squeeze()[-1,:]

        facts_emb, h_f = lstm.process_facts(facts, h_f)
        facts_emb = facts_emb[:,-1,:]

        rr = rn(facts_emb, question_emb)

        loss = criterion(rr.unsqueeze(0), answer)

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            predicted = torch.argmax(torch.sigmoid(rr)).item()
            if predicted == answer:
                accuracy += 1
            #print(dictionary[predicted], ": ", q[3])

        losses.append(loss.item())

        if ( (s %  args.print_every) == 0):
            print("Epoch ", i, ": ", s, " / ", len(stories))
            avg_losses.append(sum(losses)/float(len(losses)))
            losses = []

        s += 1

print("End training!")
print("Accuracy: ", accuracy)
import matplotlib.pyplot as plt

plt.plot(range(len(avg_losses)), avg_losses)
plt.show()
