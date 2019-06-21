import torch
from src.utils import save_models, saving_path_rn, get_answer, names_models, save_training_state
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
import win32api
from functools import partial
import time
import csv

DATASET_VALIDATION_SIZE = 0
csv_path = "./results"
execution_state_path = "./saved_models/last_execution_state.json"

def set_train_mode(module):
    module.train()
    module.zero_grad()
        
def set_eval_mode(module):
    module.eval()
    
def load_dict_from_h5(file_path, h5_path):
    dictionary = h5todict(file_path, h5_path)
    return dictionary

def get_size(dataset_questions_path):
    with open(dataset_questions_path, "r") as miniGQA_val:
        miniGQA = json.load(miniGQA_val)
        size = len(miniGQA.keys())
    return size
    

def get_batch(questions_path, features_path, batch_size, device, isObjectFeatures, categoryBatch=False):
    OBJECT_TRIM = 42
    
    questions = load_dict(questions_path)
    questions_ids = questions.keys()
    
    i = 0
    question_batch = []
    answer_ground_truth_batch = []
    object_features_batch = [] #64*OBJECT_TRIM*2048
    category_batch = []
    
    #objectsNum_batch = [] NO  ES USADO
    
    if isObjectFeatures:
        #add tensor of zeros of size OBJECT_TRIM*2048 to list
        aux_tensor = torch.zeros((OBJECT_TRIM, 2048), device=device)
        object_features_batch.append(aux_tensor)
    
    for question_id in questions_ids:
        question = questions[question_id]["question"]
        answer = questions[question_id]["answer"]
        imageId = questions[question_id]["imageId"]
        if categoryBatch:
            category = {"group": questions[question_id]["group"],
                        "types":questions[question_id]["types"]}
        
        features_dict = load_dict_from_h5(features_path, imageId)
        features = features_dict["features"] #features_h5[imageId]["features"]
        #objectNum = features_dict["objectNum"]
        # features = add_padding_features(features, objectNum)

        #features = torch.FloatTensor(features, device=device).long()
        if isObjectFeatures:
            if len(features)>OBJECT_TRIM:
                features= features[:OBJECT_TRIM]
        features = torch.tensor(features, device=device) # dtype=torch.long
        if not isObjectFeatures:
            features = features.view(-1,49).transpose(0,1)
        object_features_batch.append(features)
        answer_ground_truth_batch.append(answer)
        question_batch.append(question)
        if categoryBatch:
            category_batch.append(category)
        #objectsNum_batch.append(objectNum)
        
        i += 1

        if i == batch_size:
            # Pad of tensor that have less than OBJECT_TRIM objects, and join all in one tensor:
            object_features_batch = pad_sequence(object_features_batch, batch_first=True).to(device)
            if isObjectFeatures:
                #print(f"Dimensions after padding: {object_features_batch.shape}")
            
                #remove aux_tensor
                object_features_batch = object_features_batch[1:,:,:]

            
            # objectsNum_batch = torch.FloatTensor(objectsNum_batch, device=device).long() 
            #print(objectsNum_batch[0])
            #print("Type antes: ", type(objectsNum_batch[0]))
            #objectsNum_batch = np.array(objectsNum_batch).astype(int)
            #print("Type despues: ", type(objectsNum_batch[0]))
            #print(objectsNum_batch[0])
                        
            #yield (question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch)
            if categoryBatch:
                yield (question_batch, answer_ground_truth_batch, object_features_batch, category_batch)
            else:
                yield (question_batch, answer_ground_truth_batch, object_features_batch) 
            i = 0
            question_batch = []
            answer_ground_truth_batch = []
            object_features_batch = []
            category_batch = []
            #objectsNum_batch = []
            if isObjectFeatures:
                aux_tensor = torch.zeros((OBJECT_TRIM, 2048), device=device)
                object_features_batch.append(aux_tensor)

#Manejo ctrl + c
lstm_model = None
rn_model = None
def signalHandler(sig, func=None):
    print("Aborting but saving models first")
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_models([(lstm_model, names_models[0]), (rn_model, names_models[1])], f"saved_models/aborted_{timestr}.tar")
    raise KeyboardInterrupt


def train(train_questions_path, validation_questions_path, features_path, BATCH_SIZE, epochs,
          lstm, rn, criterion, optimizer, no_save, questions_dictionary, answers_dictionary, device,
          MAX_QUESTION_LENGTH, isObjectFeatures, past_lists=[[],[],[],[]], print_every=1):
    global DATASET_VALIDATION_SIZE
    global lstm_model
    global rn_model
    rn_model = rn
    lstm_model = lstm
    win32api.SetConsoleCtrlHandler(signalHandler, 1)

    avg_train_accuracies, train_accuracies, avg_train_losses, train_losses = past_lists

    val_accuracies = []
    val_losses = [1000.]
    best_val = val_losses[0]
    
    DATASET_TRAIN_SIZE = get_size(train_questions_path)
    DATASET_VALIDATION_SIZE = get_size(validation_questions_path)

    for i in range(epochs):
        num_batch = DATASET_TRAIN_SIZE/BATCH_SIZE
        pbar = tqdm(total=num_batch)
        print(f"Traning epochs {i}")

        dataset_size_remain = DATASET_TRAIN_SIZE
        
        batch = get_batch(train_questions_path, features_path,  BATCH_SIZE, device, isObjectFeatures)
        while dataset_size_remain > 0:
            
            if dataset_size_remain < BATCH_SIZE:
                break
            dataset_size_remain -= BATCH_SIZE
            # dataset_size_remain = 0 #TODO:
            
            #question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch = next(batch)
            question_batch, answer_ground_truth_batch, object_features_batch = next(batch)

            
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
            
            #question_emb_batch.size() = torch.Size([64, 256])
            # print(f"question_emb_batch.size() {question_emb_batch.size()}") 

            ## Pass question emb and object features to the Relation Network
            #rr = rn(object_features_batch, question_emb_batch, objectsNum_batch)
            rr = rn(object_features_batch, question_emb_batch)

            # Adjust weight
            loss = criterion(rr, answer_ground_truth_batch) # batch_size
            loss.backward()
            optimizer.step()

            # For calculating accuracies and losses
            with torch.no_grad():
                correct, _, _ = get_answer(rr, answer_ground_truth_batch)
                train_accuracies.append(correct)

            train_losses.append(loss.item())
            pbar.update()
            
        pbar.close()
        print("((i+1) %  print_every): ", ((i+1) %  print_every))
        # if ( ((i+1) %  print_every) == 0):
        print("Epoch ", i+1, "/ ", epochs)
        avg_train_losses.append(sum(train_losses)/len(train_losses))
        avg_train_accuracies.append(sum(train_accuracies)/len(train_accuracies))

        val_loss, val_accuracy = validation(validation_questions_path, features_path, BATCH_SIZE, lstm, rn, criterion, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH, isObjectFeatures)
        val_accuracies.append(val_accuracy)
        val_losses.append(val_loss)
        
        if not no_save:
            if val_losses[-1] < best_val:
                save_models([(lstm, names_models[0]), (rn, names_models[1])], saving_path_rn)
                save_training_state(avg_train_losses, avg_train_accuracies,
                                    val_losses, val_accuracies, execution_state_path)
                best_val = val_losses[-1]
            save_models([(lstm, names_models[0]),
                         (rn, names_models[1])], f"saved_models/epoch_{i+1}.tar")
            save_training_state(avg_train_losses, avg_train_accuracies,
                                val_losses, val_accuracies, f"./saved_models/epoch_{i+1}_execution_state.json")
            

        print("Train loss: ", avg_train_losses[-1], ". Validation loss: ", val_losses[-1])
        print("Train accuracy: ", avg_train_accuracies[-1], ". Validation accuracy: ", val_accuracies[-1])
        train_losses = []
        train_accuracies = []

    return avg_train_losses, avg_train_accuracies, val_losses[1:], val_accuracies

def test(dataset_questions_path, features_path, BATCH_SIZE, lstm, rn, criterion, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH, isObjectFeatures):

    val_loss = 0.
    val_accuracy = 0.

    set_eval_mode(rn)
    set_eval_mode(lstm)

    with torch.no_grad():
         
        dataset_size_remain = get_size(dataset_questions_path)

        print("Testing")
        batch = get_batch(dataset_questions_path, features_path, BATCH_SIZE, device, isObjectFeatures, categoryBatch=True)

        groups = {}
        groups_acc = []
        types = {"semantic":   {},
                 "detailed":   {},
                 "structural": {}
                }
        semantic_acc = []
        structural_acc = []
        detailed_acc = []
        
        #pbar = tqdm(total=num_batch)
        batch_number = 0
        while dataset_size_remain > 0:
            
            # Get batch 
            
            if dataset_size_remain < BATCH_SIZE:
                break
            dataset_size_remain -= BATCH_SIZE
            
            #question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch = next(batch)
            question_batch, answer_ground_truth_batch, object_features_batch, category_batch = next(batch)

            h_q = lstm.reset_hidden_state()

            question_batch, answer_ground_truth_batch = vectorize_gqa(question_batch, answer_ground_truth_batch,
                                                            questions_dictionary , answers_dictionary,
                                                            BATCH_SIZE, device, MAX_QUESTION_LENGTH)

            ## Pass question through LSTM
            question_emb_batch, h_q = lstm.process_question(question_batch, h_q)
            question_emb_batch = question_emb_batch[:, -1]

            ## Pass question emb and object features to the Relation Network
            rr = rn(object_features_batch, question_emb_batch)

            loss = criterion(rr, answer_ground_truth_batch)
            val_loss += loss.item()

            correct, _, correct_answers = get_answer(rr, answer_ground_truth_batch, return_answer=True)
            val_accuracy += correct
            
            """
            Structure:
                groups -> {
                    "group_name1": (100, 200), -> This means 200 question were tested and 100 were correct, giving 50% accuracy
                    "group_name2": (20, 30),
                    "group_name3": (5, 8)
                }
                
                types -> {
                    "structural": {
                        "struct1": (20, 30), -> same structure as the one on groups
                        "struct2": (10, 15),
                    },
                    "semantic": {}, -> they might be empty (so might "groups")
                    "detailed": {
                        "det1": (1, 4),
                        "det2": (30, 45)
                    }
                }
            """
            
            # Obtain results for each group and type
            for question, correct_answer in zip(category_batch, correct_answers):
                
                group = question["group"] # e.g. -> all color questions
                if group is not None:
                    group_rights, group_total = groups.get(group, (0, 0))
                    groups[group] = (group_rights + correct_answer, group_total + 1)
                else:
                    group_rights, group_total = groups.get("None", (0, 0))
                    groups["None"] = (group_rights + correct_answer, group_total + 1)
                
                for typ in question["types"]: # -> e.g. semantic, detailed, structural
                    type_category = question["types"][typ] # -> e.g. query
                    if type_category is not None:
                        category_rights, category_total = types[typ].get(type_category, (0, 0))
                        types[typ][type_category] = (category_rights + correct_answer, category_total + 1)
                    else:
                        category_rights, category_total = types[typ].get("None", (0, 0))
                        types[typ]["None"] = (category_rights + correct_answer, category_total + 1)
                        
            
            batch_number += 1
            #pbar.update() 


        print(f"Accuracy seperated by group")
        for group in groups:
            rights, total = groups[group]
            groups_acc.append([group, 100*rights/total])
            print(f"Group: {group} -> {rights}/{total} gives us {100*rights/total}% ")
        write_csv(groups_acc, "group_accuracy")
            
        print("___________________________________")
            
        print(f"Accuracy seperated by types")
        for typ in types:
            print(f"Type: {typ}")
            current_type = types[typ]
            for category in current_type:
                rights, total = current_type[category]
                print(f"Category: {category} -> {rights}/{total} gives us {100*rights/total}% ")
            print("___________________________________")
               
        for category in types["semantic"]:
            rights, total = types["semantic"][category]
            semantic_acc.append([category, 100*rights/total])
        write_csv(semantic_acc, "semantic_accuracy")
        
        for category in types["structural"]:
            rights, total = types["structural"][category]
            structural_acc.append([category, 100*rights/total])
        write_csv(structural_acc, "structural_accuracy")
        
        for category in types["detailed"]:
            rights, total = types["detailed"][category]
            detailed_acc.append([category, 100*rights/total])
        write_csv(detailed_acc, "detailed_accuracy")

        #pbar.close()

        val_accuracy /= float(batch_number)
        val_loss /= float(batch_number)

        return val_loss, val_accuracy

def write_csv(data, filename):
    with open(csv_path + "/" + filename + ".csv", 'w', newline='') as writeFile:
        header = ["Category", "Accuracy"]
        writer = csv.writer(writeFile)
        fileData = [header] + data
        writer.writerows(fileData)

def validation(dataset_questions_path, features_path, BATCH_SIZE, lstm, rn, criterion, questions_dictionary, answers_dictionary, device, MAX_QUESTION_LENGTH, isObjectFeatures, test_mode=True):

    val_loss = 0.
    val_accuracy = 0.

    set_eval_mode(rn)
    set_eval_mode(lstm)

    with torch.no_grad():
         
        dataset_size_remain = get_size(dataset_questions_path)

        print("Validation")
        batch = get_batch(dataset_questions_path, features_path,
                            BATCH_SIZE, device, isObjectFeatures)

        #pbar = tqdm(total=num_batch)
        batch_number = 0
        while dataset_size_remain > 0:
            
            # Get batch 
            
            if dataset_size_remain < BATCH_SIZE:
                break
            dataset_size_remain -= BATCH_SIZE
            #dataset_size_remain = 0 #TODO
        
            question_batch, answer_ground_truth_batch, object_features_batch = next(batch)

            h_q = lstm.reset_hidden_state()

            question_batch, answer_ground_truth_batch = vectorize_gqa(question_batch, answer_ground_truth_batch,
                                                            questions_dictionary , answers_dictionary,
                                                            BATCH_SIZE, device, MAX_QUESTION_LENGTH)

            ## Pass question through LSTM
            question_emb_batch, h_q = lstm.process_question(question_batch, h_q)
            question_emb_batch = question_emb_batch[:, -1]

            ## Pass question emb and object features to the Relation Network
            rr = rn(object_features_batch, question_emb_batch)

            loss = criterion(rr, answer_ground_truth_batch)
            val_loss += loss.item()

            correct, _, _ = get_answer(rr, answer_ground_truth_batch, return_answer=True)
            val_accuracy += correct
            
            batch_number += 1

            #pbar.update()mn 

        #pbar.close()

        val_accuracy /= float(batch_number)
        val_loss /= float(batch_number)

        return val_loss, val_accuracy
