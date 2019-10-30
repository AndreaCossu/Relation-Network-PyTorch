import wandb
from src.models.RN import RelationNetwork
from src.nlp_utils import read_babi, vectorize_babi
from src.models.LSTM import LSTM
import torch
import argparse
import os
from itertools import chain
from src.utils import *
from task.babi_task.rn.train import train, test, test_separately


parser = argparse.ArgumentParser()

parser.add_argument('name', type=str, help='folder in which to store results')

parser.add_argument('--epochs', type=int, default=1, help='epochs to train.')
parser.add_argument('--hidden_dims_g', nargs='+', type=int, default=[256, 256, 256], help='layers of relation function g')
parser.add_argument('--output_dim_g', type=int, default=256, help='output dimension of relation function g')
parser.add_argument('--hidden_dims_f', nargs='+', type=int, default=[256, 512], help='layers of final network f')
parser.add_argument('--hidden_dim_lstm', type=int, default=32, help='units of LSTM')
parser.add_argument('--lstm_layers', type=int, default=1, help='layers of LSTM')

parser.add_argument('--emb_dim', type=int, default=32, help='word embedding dimension')
parser.add_argument('--batch_size', type=int, default=3, help='batch size')

parser.add_argument('--dropout', action="store_true", help='enable dropout')
parser.add_argument('--tanh_act', action="store_true", help='use tanh activation for MLP instead of relu')
parser.add_argument('--wave_penc', action="store_true", help='use sin/cos positional encoding instead of one-of-k')


# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
parser.add_argument('--babi_tasks', nargs='+', type=int, default=-1, help='which babi task to train and test. -1 to select all of them.')

parser.add_argument('--split_manually', action="store_true", help='Use en-10k folder instead of en-valid-10k folder of babi. Active only with --babi_tasks specified.')
parser.add_argument('--only_relevant', action="store_true", help='read only relevant fact from babi dataset. Active only with --split_manually')

# optimizer parameters
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer hyperparameter')
parser.add_argument('--learning_rate', type=float, default=2e-4, help='optimizer hyperparameter')

parser.add_argument('--test_on_test', action="store_true", help='final test on test set instead of validation set')
parser.add_argument('--test_jointly', action="store_true", help='final test on all tasks')
parser.add_argument('--cuda', action="store_true", help='use gpu')
parser.add_argument('--load', action="store_true", help=' load saved model')
parser.add_argument('--no_save', action="store_true", help='disable model saving')
args = parser.parse_args()

if args.batch_size == 1:
    print("Batch size must be > 1. Setting it to 2.")
    args.batch_size = 2

result_folder = get_run_folder(args.name)

wandb.init(project="relation-network-babi", name=args.name, config=args, dir=result_folder)

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

if args.babi_tasks == -1: # 20 tasks are already dumped to file
    args.babi_tasks = list(range(1,21))
    print('Loading babi')
    dictionary = load_dict(separately=False) # use always en_valid

    train_stories = load_stories(False, 'train')
    validation_stories = load_stories(False, 'valid')
    if args.test_on_test:
        test_stories = load_stories(False, 'test')

    print('Babi loaded')

else: # single combinations have to be preprocessed from scratch
    if not args.split_manually:
        path_babi_base = os.path.join(cd, os.path.join("babi", "en-valid-10k"))
        to_read_test = [files_names_test_en_valid[i-1] for i in args.babi_tasks]
        to_read_val = [files_names_val_en_valid[i-1] for i in args.babi_tasks]
        to_read_train = [files_names_train_en_valid[i-1] for i in args.babi_tasks]
    else:
        path_babi_base = os.path.join(cd, os.path.join("babi", "en-10k"))
        to_read_test = [files_names_test_en[i-1] for i in args.babi_tasks]
        to_read_train = [files_names_train_en[i-1] for i in args.babi_tasks]

    print("Reading babi")



    if args.split_manually: # When reading from en-10k and not from en-valid-10k
        stories, dictionary, labels = read_babi(path_babi_base, to_read_train, args.babi_tasks, only_relevant=args.only_relevant)
        train_stories, validation_stories = split_train_validation(stories, labels)
        train_stories = vectorize_babi(train_stories, dictionary, device)
        validation_stories = vectorize_babi(validation_stories, dictionary, device)
    else:
        train_stories, dictionary, _ = read_babi(path_babi_base, to_read_train, args.babi_tasks, only_relevant=args.only_relevant)
        train_stories = vectorize_babi(train_stories, dictionary, device)
        validation_stories, _, _ = read_babi(path_babi_base, to_read_val, args.babi_tasks, only_relevant=args.only_relevant)
        validation_stories = vectorize_babi(validation_stories, dictionary, device)
    if args.test_on_test:
        test_stories, _, _ = read_babi(path_babi_base, to_read_test, args.babi_tasks, only_relevant=args.only_relevant)
        test_stories = vectorize_babi(test_stories, dictionary, device)


# save_dict(dictionary, args.split_manually)
# save_stories(train_stories, args.split_manually, 'train')
# save_stories(validation_stories, args.split_manually, 'valid')
# save_stories(test_stories, args.split_manually, 'test')


def init_weights(m):
    # if m.dim() > 1:
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)


dict_size = len(dictionary)
print("Dictionary size: ", dict_size)

lstm = LSTM(args.hidden_dim_lstm, args.batch_size, dict_size, args.emb_dim, args.lstm_layers, device, wave_penc=args.wave_penc, dropout=args.dropout).to(device)
lstm.apply(init_weights)

rn = RelationNetwork(args.hidden_dim_lstm, args.hidden_dims_g, args.output_dim_g, args.hidden_dims_f, dict_size, args.dropout, args.tanh_act, args.batch_size, args.wave_penc, device).to(device)
rn.apply(init_weights)

wandb.watch(rn)
wandb.watch(lstm)

if args.load:
    load_models([(lstm, names_models[0]), (rn, names_models[1])], result_folder, saving_path_rn)

optimizer = torch.optim.Adam(chain(lstm.parameters(), rn.parameters()), args.learning_rate, weight_decay=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')

if args.epochs > 0:
    print("Start training")
    avg_train_losses, avg_train_accuracies, val_losses, val_accuracies = train(train_stories, validation_stories, args.epochs, lstm, rn, criterion, optimizer, args.no_save, device, result_folder, args.batch_size)
    print("End training!")

if not args.test_on_test:
    test_stories = validation_stories

if args.test_jointly:
    print("Testing jointly...")
    avg_test_loss, avg_test_accuracy = test(test_stories, lstm, rn, criterion, device, args.batch_size)

    print("Test accuracy: ", avg_test_accuracy)
    print("Test loss: ", avg_test_loss)
else:
    print("Testing separately...")
    avg_test_accuracy = test_separately(test_stories, lstm, rn, device, args.batch_size)
    avg_test_loss = None
    print("Test accuracy: ", avg_test_accuracy)

    write_test(result_folder, losses=avg_test_loss, accs=avg_test_accuracy)

if args.epochs > 0:
    plot_results(result_folder, avg_train_losses, val_losses, avg_train_accuracies, val_accuracies)
