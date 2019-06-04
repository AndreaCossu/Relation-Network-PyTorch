import torch
from src.utils import save_models, saving_path_rn, get_answer, names_models
from src.utils import  random_idx_gen
from src.nlp_utils import vectorize_gqa
import os
import random
import h5py
import json
from tqdm import tqdm
from silx.io.dictdump import h5todict
from utils.generate_dictionary import load_dict
from torch.nn.utils.rnn import pad_sequence
import numpy as np

DATASET_VALIDATION_SIZE = 0

def set_train_mode(module):
    module.train()
    module.zero_grad()
        
def set_eval_mode(module):
    module.eval()
    
def load_dict_from_h5(file_path, h5_path):
    dictionary = h5todict(file_path, h5_path)
    return dictionary

def get_size(dataset_questions_path, validation=False):
    with open(dataset_questions_path, "r") as miniGQA_val:
        miniGQA = json.load(miniGQA_val)
        size = len(miniGQA.keys())
    return size
    

def get_batch(questions_path, features_path, batch_size, device):
    questions = load_dict(questions_path)
    questions_ids = questions.keys()
    
    i = 0
    question_batch = []
    answer_ground_truth_batch = []
    object_features_batch = [] #64*57*2048
    objectsNum_batch = []
    for question_id in questions_ids:
        question = questions[question_id]["question"]
        answer = questions[question_id]["answer"]
        imageId = questions[question_id]["imageId"]
        
        features_dict = load_dict_from_h5(features_path, imageId)
        features = features_dict["features"] #features_h5[imageId]["features"]
        objectNum = features_dict["objectNum"]
        # features = add_padding_features(features, objectNum)

        features = torch.FloatTensor(features, device=device).long()
        
        object_features_batch.append(features)
        answer_ground_truth_batch.append(answer)
        question_batch.append(question)
        objectsNum_batch.append(objectNum)
        
        i += 1
        
        if i == batch_size:
            print(f"Dimensions before padding: {object_features_batch[0].shape}")
            # print(f"type(object_features_batch[0]): {type(object_features_batch[0])}")
            object_features_batch = pad_sequence(object_features_batch, batch_first=True).long()
            print(f"Dimensions after padding: {object_features_batch[0].shape}")

            # objectsNum_batch = torch.FloatTensor(objectsNum_batch, device=device).long() 
            print(objectsNum_batch[0])
            print("Type antes: ", type(objectsNum_batch[0]))
            objectsNum_batch = np.array(objectsNum_batch).astype(int)
            print("Type despues: ", type(objectsNum_batch[0]))
            print(objectsNum_batch[0])
            
            yield (question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch)
            i = 0
            question_batch = []
            answer_ground_truth_batch = []
            object_features_batch = []
            objectsNum_batch = []
    

def train(train_questions_path, validation_questions_path, features_path, BATCH_SIZE, epochs, lstm, rn, criterion, optimizer, no_save, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH, print_every=1):
    global DATASET_VALIDATION_SIZE
    avg_train_accuracies = []
    train_accuracies = []
    avg_train_losses = []
    train_losses = []

    val_accuracies = []
    val_losses = [1000.]
    best_val = val_losses[0]
    
    DATASET_TRAIN_SIZE = get_size(train_questions_path)
    DATASET_VALIDATION_SIZE = get_size(validation_questions_path , validation=True)

    for i in range(epochs):
        num_batch = DATASET_TRAIN_SIZE/BATCH_SIZE
        pbar = tqdm(total=num_batch)
        
        dataset_size_remain = DATASET_TRAIN_SIZE
        
        while dataset_size_remain > 0:
            
            batch_size = min(BATCH_SIZE, dataset_size_remain)
            dataset_size_remain -= batch_size
            
            batch = get_batch(train_questions_path, features_path,  batch_size, device)
            
            question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch = next(batch)

            # Reset model
            
            set_train_mode(rn)
            set_train_mode(lstm)

            h_q = lstm.reset_hidden_state()
            
            # Train
            
            """
            def vectorize_gqa(batch_question, batch_answer, dictionary_question , dictionary_answer, batch_size, device):
    
            returns:
            * question_v: list (64) of tensor (question_length)
            * answers_v: tensor (64)
            """

            question_batch, answer_ground_truth_batch = vectorize_gqa(question_batch, answer_ground_truth_batch,
                                                                      questions_dictionary , answers_dictionary,
                                                                      BATCH_SIZE, device, MAX_QUESTION_LENGTH)
            
            ## Pass question through LSTM
            question_emb_batch, h_q = lstm.process_question(question_batch, h_q)
            question_emb_batch = question_emb_batch[:,-1]

            ## Pass question emb and object features to the Relation Network
            rr = rn(object_features_batch, question_emb_batch, objectsNum_batch)

            # Adjust weight
            loss = criterion(rr, answer_ground_truth_batch)
            loss.backward()
            optimizer.step()

            # For calculating accuracies and losses
            with torch.no_grad():
                correct, _ = get_answer(rr, answer_ground_truth_batch)
                train_accuracies.append(correct)

            train_losses.append(loss.item())
            pbar.update()
        pbar.close()

        if ( ((i+1) %  print_every) == 0):
            print("Epoch ", i+1, "/ ", epochs)
            avg_train_losses.append(sum(train_losses)/len(train_losses))
            avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))

            val_loss, val_accuracy = test(validation_questions_path, features_path,BATCH_SIZE, lstm, rn, criterion, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH)
            val_accuracies.append(val_accuracy)
            val_losses.append(val_loss)

            if not no_save:
                if val_losses[-1] < best_val:
                    save_models([(lstm, names_models[0]), (rn, names_models[1])], saving_path_rn)
                    best_val = val_losses[-1]

            print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
            print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
            train_losses = []
            train_accuracies = []

    return avg_train_losses, avg_train_accuracies, val_losses[1:], val_accuracies

def test(dataset_questions_path, features_path, BATCH_SIZE, lstm, rn, criterion, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH, test_mode=True):

    val_loss = 0.
    val_accuracy = 0.

    set_eval_mode(rn)
    set_eval_mode(lstm)

    with torch.no_grad():
         
        dataset_size_remain = get_size(dataset_questions_path)
        
        while dataset_size_remain > 0:
            
            # Get batch 
            
            batch_size = min(BATCH_SIZE, dataset_size_remain)
            dataset_size_remain -= batch_size
            
            batch = get_batch(dataset_questions_path, features_path, batch_size, device)
        
            question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch = next(batch)

            h_q = lstm.reset_hidden_state()

            question_batch, answer_ground_truth_batch = vectorize_gqa(question_batch, answer_ground_truth_batch,
                                                            questions_dictionary , answers_dictionary,
                                                            batch_size, device, MAX_QUESTION_LENGTH)

            ## Pass question through LSTM
            question_emb_batch, h_q = lstm.process_question(question_batch, h_q)
            question_emb_batch = question_emb_batch[:, -1]

            ## Pass question emb and object features to the Relation Network
            rr = rn(object_features_batch, question_emb_batch)

            loss = criterion(rr, answer_ground_truth_batch)
            val_loss += loss.item()

            correct, _ = get_answer(rr, answer_ground_truth_batch)
            val_accuracy += correct

        val_accuracy /= float(dataset_size_remain)
        val_loss /= float(dataset_size_remain)

        return val_loss, val_accuracy