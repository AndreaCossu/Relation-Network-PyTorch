from src.RN import RelationNetwork
from src.nlp_utils import read_babi, vectorize_babi
from src.LSTM import LSTM
import torch
import argparse
import os
from itertools import chain
from src.utils import files_names_test, files_names_train, saving_paths_models, load_models, split_train_validation
from src.train import train_single, test


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2)

parser.add_argument('--hidden_dims_g', nargs='+', type=int, default=[256, 256, 256])
parser.add_argument('--hidden_dims_f', nargs='+', type=int, default=[256, 512])
parser.add_argument('--hidden_dim_lstm', type=int, default=32)
parser.add_argument('--output_dim_g', type=int, default=256)
parser.add_argument('--lstm_layers', type=int, default=1)

parser.add_argument('--emb_dim', type=int, default=50)

# which babi task to train and test
# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
parser.add_argument('--babi_tasks', nargs='+', type=int, default=[1])

parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--learning_rate', type=float, default=2e-4)
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

device = torch.device(mode)

cd = os.path.dirname(os.path.abspath(__file__))
path_babi_base = cd + "/babi/en-10k/"
print("Reading babi")

to_read_test = [files_names_test[i-1] for i in args.babi_tasks]
to_read_train = [files_names_train[i-1] for i in args.babi_tasks]
stories, dictionary, labels = read_babi(path_babi_base, to_read_train, args.babi_tasks)
stories = vectorize_babi(stories, dictionary, device)

train_stories, validation_stories = split_train_validation(stories, labels)

test_stories, _, _ = read_babi(path_babi_base, to_read_test, args.babi_tasks)
test_stories = vectorize_babi(test_stories, dictionary, device)

dict_size = len(dictionary)
print("Dictionary size: ", dict_size)
print("Done reading babi!")

lstm = LSTM(args.hidden_dim_lstm, args.batch_size_lstm, dict_size, args.emb_dim, device)

rn = RelationNetwork(args.hidden_dim_lstm, args.hidden_dims_g, args.output_dim_g, args.hidden_dims_f, dict_size,
                     device)

if args.load:
    load_models([lstm, rn], saving_paths_models)

optimizer = torch.optim.Adam(chain(lstm.parameters(), rn.parameters()), args.learning_rate, weight_decay=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss()

if args.epochs > 0:
    print("Start training")
    avg_train_losses, avg_train_accuracies, val_losses, val_accuracies = train_single(train_stories, validation_stories, args.epochs, lstm, rn, criterion, optimizer, args.print_every, args.no_save)
    print("End training!")

print("Testing...")
avg_test_loss, avg_test_accuracy = test(test_stories, lstm, rn, criterion)

print("Test accuracy: ", avg_test_accuracy)
print("Test loss: ", avg_test_loss)

if args.epochs > 0:
    import matplotlib

    if args.cuda:
        matplotlib.use('Agg')

        import matplotlib.pyplot as plt


    plt.figure()
    plt.plot(range(len(avg_train_losses)), avg_train_losses, 'b', label='train')
    plt.plot(range(len(val_losses)), val_losses, 'r', label='val')
    plt.legend(loc='best')

    if args.cuda:
        plt.savefig('loss.png')
    else:
        plt.show()

    plt.figure()
    plt.plot(range(len(avg_train_accuracies)), avg_train_accuracies, 'b', label='train')
    plt.plot(range(len(val_accuracies)), val_accuracies, 'r', label='val')
    plt.legend(loc='best')

    if args.cuda:
        plt.savefig('accuracy.png')
    else:
        plt.show()
