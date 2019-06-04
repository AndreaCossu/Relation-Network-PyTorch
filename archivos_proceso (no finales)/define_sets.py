import json
import random
import h5py
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from silx.io.dictdump import dicttoh5
import os

"""
Este código lo usamos para dividir las preguntas de miniGQA en un training set y un testing set. 
Además filtramos las preguntas para incluir solo aquellas que se refieren a una imagen en miniGQA

outputs:
    training_question_ids
    testing_question_ids.h5
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
        
def add_to_dict(dictionary, file_path, h5_path):
    create_ds_args = {'shuffle': False}
    #print("save to: " + h5_path)
    dicttoh5(dictionary, file_path, h5path=h5_path, mode="a",
             create_dataset_args=create_ds_args)

if __name__ == "__main__":
    new_train_json_path = "./data/miniGQA/new_train.json"
    output_training_question_ids_path = "./data/miniGQA/training_question_ids.h5"
    output_testing_question_ids_path = "./data/miniGQA/testing_question_ids.h5"
    id_images_in_miniGQA_path = "./data/miniGQA/id_images_in_miniGQA.json"
    
    # new_train_json_path = "./data/miniGQA/json_dummy.json"
    
    PERCENTAGE = 10
    
    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = json.load(f)
        id_images_in_miniGQA = set(id_images_in_miniGQA)
        print(f"number of image ids in miniGQA: {len(id_images_in_miniGQA)}")
        
    
    with open(new_train_json_path) as f:
        train_json = json.load(f)
        #print(f"train_json {train_json}")        
        lst = list(train_json.keys())
        
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
        valid_train_questions_ids = []
        for question_id in train_questions:
            imageId = train_json[question_id]["imageId"]
            if imageId in id_images_in_miniGQA: #Solo incluir preguntas relacionadas a imagenes en miniGQA
                question = train_json[question_id]["question"]
                answer = train_json[question_id]["answer"]
                question_dict = {"question": question, "answer": answer, "imageId": imageId}
                add_to_dict(question_dict, output_training_question_ids_path, question_id)
                valid_train_questions_ids.append(imageId)
            pbar.update()
        paths_dict = {"paths": valid_train_questions_ids}
        add_to_dict(paths_dict, output_training_question_ids_path, "/questions_id")
        pbar.close()
        print("Saved training questions!")
        
        #Save testing questions
        endvalue = test_size
        value = 0
        pbar = tqdm(total=endvalue)
        valid_test_questions_ids = []
        for question_id in test_questions:
            imageId = train_json[question_id]["imageId"]
            if imageId in id_images_in_miniGQA: #Solo incluir preguntas relacionadas a imagenes en miniGQA
                question = train_json[question_id]["question"]
                answer = train_json[question_id]["answer"]
                question_dict = {"question": question, "answer": answer, "imageId": imageId}
                add_to_dict(question_dict, output_testing_question_ids_path, question_id)
                valid_test_questions_ids.append(imageId)
            pbar.update()
        paths_dict = {"paths": valid_test_questions_ids}
        add_to_dict(paths_dict, output_testing_question_ids_path, "/questions_id")
        pbar.close()
        print("Saved testing questions!")
    
    print("\nFinished!")