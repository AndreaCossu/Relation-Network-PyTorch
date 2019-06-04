from src.models.RN_image import RelationNetwork
from src.nlp_utils import read_babi, vectorize_babi
from src.models.LSTM import LSTM
import torch
import argparse
import os
from itertools import chain
from src.utils import files_names_test_en, files_names_train_en, files_names_test_en_valid, files_names_train_en_valid, files_names_val_en_valid
from src.utils import saving_path_rn, names_models, load_models, split_train_validation
from src.utils import load_dict, save_dict
from task.gqa_task.rn.train_objects import train, test

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1, help='epochs to train.')

# g-mpl arguments
parser.add_argument('--hidden_dims_g', nargs='+', type=int, default=[512, 512, 512], help='layers of relation function g')
parser.add_argument('--output_dim_g', type=int, default=512, help='output dimension of relation function g')
parser.add_argument('--dropouts_g', nargs='+', type=bool, default=[False,False,False], help='witch hidden layers of function g haves dropout')
parser.add_argument('--drop_prob_g', type=float, default=0.5, help='prob dropout hidden layers of function g')

# f-mpl arguments
parser.add_argument('--hidden_dims_f', nargs='+', type=int, default=[512, 1024], help='layers of final network f')
parser.add_argument('--dropouts_f', nargs='+', type=bool, default=[False,True], help='witch hidden layers of function f haves dropout')
parser.add_argument('--drop_prob_f', type=float, default=0.02, help='prob dropout hidden layers of function f')

#lstm arguments
parser.add_argument('--hidden_dim_lstm', type=int, default=256, help='units of LSTM')
parser.add_argument('--lstm_layers', type=int, default=1, help='layers of LSTM')

#embething arguments
parser.add_argument('--emb_dim', type=int, default=32, help='word embedding dimension')
parser.add_argument('--only_relevant', action="store_true", help='read only relevant fact from babi dataset')
parser.add_argument('--batch_size_stories', type=int, default=10, help='stories batch size')

# optimizer parameters
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer hyperparameter')
#parser.add_argument('--learning_rate', type=float, default=2e-4, help='optimizer hyperparameter')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer hyperparameter')

parser.add_argument('--cuda', action="store_true", help='use gpu')
parser.add_argument('--load', action="store_true", help=' load saved model')
parser.add_argument('--no_save', action="store_true", help='disable model saving')
parser.add_argument('--print_every', type=int, default=500, help='print information every print_every steps')
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

#Set Paths
train_questions_path = "./data/miniGQA/training_question_ids.hdf5"
test_questions_path = "./data/miniGQA/testing_question_ids.hdf5"
validation_questions_path = "./data/miniGQA/new_valid.json"
features_path = "./data/miniGQA/miniGQA_imageFeatures.hdf5"

if not args.en_valid: # When reading from en-10k and not from en-valid-10k
    stories, dictionary, labels = read_babi(path_babi_base, to_read_train, args.babi_tasks, only_relevant=args.only_relevant)
    train_stories, validation_stories = split_train_validation(stories, labels)
    train_stories = vectorize_babi(train_stories, dictionary, args.batch_size_stories, device)
    validation_stories = vectorize_babi(validation_stories, dictionary, args.batch_size_stories, device)
else:
    train_stories, dictionary, labels = read_babi(path_babi_base, to_read_train, args.babi_tasks, only_relevant=args.only_relevant)
    train_stories = vectorize_babi(train_stories, dictionary, args.batch_size_stories, device)
    validation_stories, _, _ = read_babi(path_babi_base, to_read_val, args.babi_tasks, only_relevant=args.only_relevant)
    validation_stories = vectorize_babi(validation_stories, dictionary, args.batch_size_stories, device)

test_stories, _, _ = read_babi(path_babi_base, to_read_test, args.babi_tasks, only_relevant=args.only_relevant)
test_stories = vectorize_babi(test_stories, dictionary, args.batch_size_stories, device)

if not args.load:
    save_dict(dictionary)
else:
    dictionary = load_dict()

dict_size = len(dictionary)
print("Dictionary size: ", dict_size)
print("Done reading babi!")

lstm = LSTM(args.hidden_dim_lstm, args.batch_size_stories, dict_size, args.emb_dim, args.lstm_layers, device).to(device)

rn = RelationNetwork(args.object_dim, args.hidden_dim_lstm, args.hidden_dims_g, args.output_dim_g, args.dropouts_g, args.drop_prob_g, args.hidden_dims_f, dict_size, args.dropouts_f, args.drop_prob_f, args.batch_size_stories, device).to(device)
#    def __init__(self, features_dim, query_dim, hidden_dims_g, output_dim_g, drops_g, hidden_dims_f, output_dim_f, drops_f, batch_size, device)

if args.load:
    load_models([(lstm, names_models[0]), (rn, names_models[1])], saving_path_rn)

optimizer = torch.optim.Adam(chain(lstm.parameters(), rn.parameters()), args.learning_rate, weight_decay=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')

if args.epochs > 0:
    print("Start training")
    avg_train_losses, avg_train_accuracies, val_losses, val_accuracies = train(train_questions_path, validation_questions_path, features_path, args.epochs, lstm, rn, criterion, optimizer, args.no_save, args.print_every)
    print("End training!")

print("Testing...")
avg_test_loss, avg_test_accuracy = test(test_questions_path, features_path, lstm, rn, criterion, test_mode=True)

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
        plt.savefig('plots/loss.png')
    else:
        plt.show()

    plt.figure()
    plt.plot(range(len(avg_train_accuracies)), avg_train_accuracies, 'b', label='train')
    plt.plot(range(len(val_accuracies)), val_accuracies, 'r', label='val')
    plt.legend(loc='best')

    if args.cuda:
        plt.savefig('plots/accuracy.png')
    else:
        plt.show()
