import json
import random
import h5py
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import os

"""
Este código lo usamos para dividir las preguntas de miniGQA en un training set y un testing set. 
Además filtramos las preguntas para incluir solo aquellas que se refieren a una imagen en miniGQA

outputs:
    training_question_ids.json
    testing_question_ids.json
"""

def progressBar(value, endvalue, bar_length=20):
    if endvalue > 0:
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def get_indexes(images_path):
    indexes = os.listdir(images_path)
    print(f"Number of images: {len(indexes)}")
    return [index.split(".")[0] for index in indexes]
        

if __name__ == "__main__":
    new_train_json_path = "./data/miniGQA/new_train.json"
    output_training_question_ids_path = "./data/miniGQA/training_question_ids.json"
    output_testing_question_ids_path = "./data/miniGQA/testing_question_ids.json"
    id_images_in_miniGQA_path = "./data/miniGQA/id_images_in_miniGQA.json"
    RANDOM_SEED = 30
    
    # new_train_json_path = "./data/miniGQA/json_dummy.json"
    
    PERCENTAGE = 10
    
    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = json.load(f)
        id_images_in_miniGQA = set(id_images_in_miniGQA)
        print(f"number of image ids in miniGQA: {len(id_images_in_miniGQA)}")
        
    final_train_json = {}
    final_test_json = {}
    with open(new_train_json_path) as f:
        train_json = json.load(f)
        #print(f"train_json {train_json}")        
        lst = list(train_json.keys())
        random.seed(RANDOM_SEED)
        random.shuffle(lst)
        size =  len(lst)
        test_size = int(size/PERCENTAGE)
        
        print(f"full size: {size}")
        print(f"test size: {test_size}")
        
        train_questions = lst[test_size:]
        test_questions = lst[:test_size]
        
        #Save training questions
        endvalue = size
        value = 0
        progressBar(value, endvalue)
        pbar = tqdm(total=endvalue)

        for question_id in train_questions:
            imageId = train_json[question_id]["imageId"]
            if imageId in id_images_in_miniGQA: #Solo incluir preguntas relacionadas a imagenes en miniGQA
                # question = train_json[question_id]["question"]
                # answer = train_json[question_id]["answer"]
                # question_dict = {"question": train_json[question_id]["question"], "answer": train_json[question_id]["answer"], "imageId": imageId}
                final_train_json[question_id] = {"question": train_json[question_id]["question"], "answer": train_json[question_id]["answer"], "imageId": imageId}
            del train_json[question_id] #opcional: eliminar imagen del diccionario en memoria
            pbar.update()
        pbar.close()
        
        with open(output_training_question_ids_path, "w") as training_question_ids_file:
            json.dump(final_train_json, training_question_ids_file)
        
        print("Saved training questions!")
        
        #Save testing questions
        endvalue = test_size
        value = 0
        pbar = tqdm(total=endvalue)
        for question_id in test_questions:
            imageId = train_json[question_id]["imageId"]
            if imageId in id_images_in_miniGQA: #Solo incluir preguntas relacionadas a imagenes en miniGQA
                # question = train_json[question_id]["question"]
                # answer = train_json[question_id]["answer"]
                # question_dict = {"question": train_json[question_id]["question"], "answer": train_json[question_id]["answer"], "imageId": imageId}
                final_test_json[question_id] = {"question": train_json[question_id]["question"], "answer": train_json[question_id]["answer"], "imageId": imageId}
            del train_json[question_id] #opcional: eliminar imagen del diccionario en memoria
            pbar.update()
        pbar.close()
        
        with open(output_testing_question_ids_path, "w") as testing_question_ids_file:
            json.dump(final_test_json, testing_question_ids_file)
        
        print("Saved testing questions!")
    
    print("\nFinished!")