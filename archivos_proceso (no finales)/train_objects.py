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

DATASET_VALIDATION_SIZE = 0
BATCH_SIZE = 0
features_path = "./data/miniGQA/miniGQA_objectFeatures.h5"

def set_train_mode(module):
    module.train()
    module.zero_grad()
        
def set_eval_mode(module):
    module.eval()
    
def load_dict(file_path, h5_path):
    dictionary = h5todict(file_path, h5_path)
    print("file loaded!")
    return dictionary

def get_size(dataset_questions_path, validation=False):
    if validation:
        with open(dataset_questions_path, "r") as miniGQA_val:
            miniGQA = json.load(miniGQA_val)
            size = len(miniGQA.keys())
    else:
        with h5py.File(dataset_questions_path, "r") as dataset:
            question_ids = dataset.keys()
            size = len(question_ids)
    return size

def get_train_batch(train_questions_path, features_path, batch_size):
    training_questions_paths = load_dict(train_questions_path, "questions_id")
    i = 0
    question_batch = []
    answer_ground_truth_batch = []
    object_features_batch = []
    objectsNum_batch = []
    for question_id in training_questions_paths:
        question = load_dict(train_questions_path, question_id)
        imageId = question["imageId"]
        features_dict = load_dict(features_path, imageId)
        features = features_dict["features"] #features_h5[imageId]["features"]
        objectNum = features_dict["objectNum"]

        question_batch.append(question["question"])
        answer_ground_truth_batch.append(question["answer"])
        object_features_batch.append(features)
        objectsNum_batch.append(objectNum)
        
        i += 1
        
        if i == batch_size:
            yield (question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch)
            i = 0
            question_batch = []
            answer_ground_truth_batch = []
            object_features_batch = []
            objectsNum_batch = []

def get_validation_batch(validation_questions_path, features_path, batch_size):
    with open(validation_questions_path, "r") as validation_questions_file:
        validation_questions = json.load(validation_questions_file)
        i = 0
        question_batch = []
        answer_ground_truth_batch = []
        object_features_batch = []
        objectsNum_batch = []
        for question_id in validation_questions.keys():
            question = validation_questions[question_id]
            imageId = question["imageId"]
            features_dict = load_dict(features_path, imageId)
            features = features_dict["features"] #features_h5[imageId]["features"]
            objectNum = features_dict["objectNum"]

            question_batch.append(question["question"])
            answer_ground_truth_batch.append(question["answer"])
            object_features_batch.append(features)
            objectsNum_batch.append(objectNum)
            
            i += 1
            
            if i == batch_size:
                i = 0
                yield (question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch)
                question_batch = []
                answer_ground_truth_batch = []
                object_features_batch = []
                objectsNum_batch = []

def get_test_batch(test_questions_path, features_path, batch_size):
    #TODO
    testing_questions_paths = load_dict(test_questions_path, "questions_id")
    i = 0
    question_batch = []
    answer_ground_truth_batch = []
    object_features_batch = []
    objectsNum_batch = []
    for question_id in testing_questions_paths:
        question = load_dict(test_questions_path, question_id)
        imageId = question["imageId"]
        features_dict = load_dict(features_path, imageId)
        features = features_dict["features"] #features_h5[imageId]["features"]
        objectNum = features_dict["objectNum"]

        question_batch.append(question["question"])
        answer_ground_truth_batch.append(question["answer"])
        object_features_batch.append(features)
        objectsNum_batch.append(objectNum)
        
        i += 1
        
        if i == batch_size:
            yield (question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch)
            i = 0
            question_batch = []
            answer_ground_truth_batch = []
            object_features_batch = []
            objectsNum_batch = []

def train(train_questions_path, validation_questions_path, features_path, epochs, lstm, rn, criterion, optimizer, no_save, questions_dictionary, answers_dictionary, device, print_every=1):
    global DATASET_VALIDATION_SIZE
    global BATCH_SIZE
    avg_train_accuracies = []
    train_accuracies = []
    avg_train_losses = []
    train_losses = []

    val_accuracies = []
    val_losses = [1000.]
    best_val = val_losses[0]
    
    DATASET_TRAIN_SIZE = get_size(train_questions_path)
    DATASET_VALIDATION_SIZE = get_size(validation_questions_path)
    BATCH_SIZE = 64
    
    

    for i in range(epochs):
        num_batch = DATASET_TRAIN_SIZE/BATCH_SIZE
        pbar = tqdm(total=num_batch)
        
        dataset_size_remain = DATASET_TRAIN_SIZE
        
        while dataset_size_remain > 0:
            
            batch_size = min(BATCH_SIZE, dataset_size_remain)
            dataset_size_remain -= batch_size
            
            batch = get_train_batch(train_questions_path, features_path,  batch_size)
   
            question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch = batch

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
                                                                      BATCH_SIZE, device)
            
            ## Pass question through LSTM
            question_emb_batch, h_q = lstm.process_question(question_batch, h_q)
            question_emb_batch = question_emb_batch[:,-1]

            ## Pass question emb and object features to the Relation Network
            rr = rn(object_features_batch, question_emb_batch)

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

            val_loss, val_accuracy = test(validation_questions_path, features_path, lstm, rn, criterion, questions_dictionary, answers_dictionary, device)
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

def test(dataset_questions_path, features_path, lstm, rn, criterion, questions_dictionary, answers_dictionary, device, test_mode=True):

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
            
            if test_mode:
                gen = get_test_batch
            else:
                gen = get_validation_batch
            
            batch = gen(dataset_questions_path, features_path, batch_size)
        
            question_batch, answer_ground_truth_batch, object_features_batch, objectsNum_batch = batch

            h_q = lstm.reset_hidden_state()

            question_batch, answer_ground_truth_batch = vectorize_gqa(question_batch, answer_ground_truth_batch,
                                                            questions_dictionary , answers_dictionary,
                                                            BATCH_SIZE, device)

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