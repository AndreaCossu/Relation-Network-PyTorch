from src.RN import RelationNetwork
from src.nlp_utils import Embeddings, read_babi
import torch
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--hidden_dims_g', nargs='+', type=int, default=[60, 60])
parser.add_argument('--hidden_dims_f', nargs='+', type=int, default=[60, 60])
parser.add_argument('--hidden_dim_lstm', type=int, default=100)
parser.add_argument('--output_dim_f', type=int, default=1)
parser.add_argument('--output_dim_g', type=int, default=60)
parser.add_argument('--lstm_layers', type=int, default=1)

parser.add_argument('--obj_dim', type=int, default=8)
parser.add_argument('--query_dim', type=int, default=8)
parser.add_argument('--emb_dim', type=int, default=50)


parser.add_argument('--momentum', type=float, default=0.7)
parser.add_argument('--learning_rate', type=float, default=1e-5)
parser.add_argument('--batch_size', type=int, default=8)

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

X = torch.randn(args.batch_size, args.obj_dim) # 6 objects with dimension 8
q = torch.randn(args.query_dim)

cd = os.path.dirname(os.path.abspath(__file__))

print("Reading babi")
path_babi = cd + "/babi/en-10k/" + babi_file_train
facts, questions = read_babi(path_babi)
print("Done reading babi!")


path_raw_embeddings = cd + "/glove.6B/" + embeddings_file
print("Loading embeddings")
embeddings = Embeddings(path_raw_embeddings, embeddings_file, args.emb_dim, device)
print("Embeddings loaded!")


rn = RelationNetwork(args.obj_dim, args.hidden_dims_g, args.output_dim_g, args.hidden_dims_f, args.output_dim_f,
                     device, args.query_dim)

result = rn(X,q)

print(result)
