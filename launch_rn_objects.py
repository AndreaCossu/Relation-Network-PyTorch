from src.models.RN_image import RelationNetwork
from src.nlp_utils import read_babi, vectorize_babi
from src.models.LSTM import LSTM
import torch
import argparse
import os
from itertools import chain
from src.utils import files_names_test_en, files_names_train_en, files_names_test_en_valid, files_names_train_en_valid, files_names_val_en_valid
from src.utils import saving_path_rn, names_models, load_models, split_train_validation
from task.gqa_task.rn.train_objects import train, test
from utils.generate_dictionary import generate_questions_dict, generate_answers_dict, load_dict

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='epochs to train.')

# g-mpl arguments
parser.add_argument('--object_dim', type=int, default=2048, help='number of features of each object')
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

#embedding arguments
parser.add_argument('--emb_dim', type=int, default=300, help='word embedding dimension') #Basado en paper GQA
parser.add_argument('--only_relevant', action="store_true", help='read only relevant fact from babi dataset')
parser.add_argument('--batch_size_stories', type=int, default=10, help='stories batch size')

# optimizer parameters
parser.add_argument('--weight_decay', type=float, default=0, help='optimizer hyperparameter')
#parser.add_argument('--learning_rate', type=float, default=2e-4, help='optimizer hyperparameter')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer hyperparameter')

parser.add_argument('--cuda', type=bool, default=True, help='use gpu')
parser.add_argument('--load', type=bool, default=False, help=' load saved model')
parser.add_argument('--no_save', type=bool, default=False, help='disable model saving')
parser.add_argument('--load_dictionary', type=bool, default=False, help='load dict from path')
parser.add_argument('--dictionary_path', action="store_true", help='load dict from path')
parser.add_argument('--print_every', type=int, default=500, help='print information every print_every steps')
args = parser.parse_args()

print(f"load dict bool: {args.load_dictionary}")
print(f"save model bool: {not args.no_save}")

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
train_questions_path = "./data/miniGQA/training_question_ids.json"
test_questions_path = "./data/miniGQA/testing_question_ids.json"
validation_questions_path = "./data/miniGQA/new_valid_filtered.json"
features_path = "./data/miniGQA/miniGQA_objectFeatures.h5"
questions_dictionary_path = "./data/miniGQA/questions_dictionary.json"
answers_dictionary_path = "./data/miniGQA/answers_dictionary.json"
MAX_QUESTION_LENGTH = 136
BATCH_SIZE = 128
isObjectFeatures = True

#torch.set_default_tensor_type(torch.FloatTensor)

if not args.load_dictionary:
    questions_dictionary = generate_questions_dict(train_questions_path, test_questions_path, validation_questions_path, questions_dictionary_path)
    answers_dictionary = generate_answers_dict(train_questions_path, test_questions_path, validation_questions_path, answers_dictionary_path)    
else:
    questions_dictionary = load_dict(questions_dictionary_path)
    answers_dictionary = load_dict(answers_dictionary_path)
questions_dict_size = len(questions_dictionary)
print(f"Questions dictionary size: {questions_dict_size}")
print(f"Answers dictionary size: {len(answers_dictionary)}")


lstm = LSTM(args.hidden_dim_lstm, BATCH_SIZE, questions_dict_size, args.emb_dim, args.lstm_layers, device).to(device)
rn = RelationNetwork(args.object_dim, args.hidden_dim_lstm, args.hidden_dims_g, args.output_dim_g, args.dropouts_g, args.drop_prob_g, args.hidden_dims_f, questions_dict_size, args.dropouts_f, args.drop_prob_f, BATCH_SIZE, device).to(device)
print("Modelos definidos.")

if args.load:
    load_models([(lstm, names_models[0]), (rn, names_models[1])], saving_path_rn)

optimizer = torch.optim.Adam(chain(lstm.parameters(), rn.parameters()), args.learning_rate, weight_decay=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss(reduction='mean')

if args.epochs > 0:
    print("Start training")
    avg_train_losses, avg_train_accuracies, val_losses, val_accuracies = train(train_questions_path, validation_questions_path, features_path, BATCH_SIZE, args.epochs, lstm, rn, criterion, optimizer, args.no_save, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH, isObjectFeatures, args.print_every)
    print("End training!")

print("Testing...")
avg_test_loss, avg_test_accuracy = test(test_questions_path, features_path, BATCH_SIZE, lstm, rn, criterion, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH, isObjectFeatures, test_mode=True)



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
