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

if __name__ == "__main__":
    new_train_json_path = "./data/miniGQA/new_train.json"
    output_training_question_ids_path = "./data/miniGQA/training_question_ids.json"
    output_testing_question_ids_path = "./data/miniGQA/testing_question_ids.json"
    id_images_in_miniGQA_path = "./data/miniGQA/id_images_in_miniGQA.json"
    
    RANDOM_SEED = 30
    PERCENTAGE = 10
    
    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = json.load(f)
        id_images_in_miniGQA = set(id_images_in_miniGQA)
        print(f"Number of images in miniGQA: {len(id_images_in_miniGQA)}")
        
    final_train_json = {}
    final_test_json = {}
    with open(new_train_json_path) as f:
        train_json = json.load(f)
        lst = list(train_json.keys())
        random.seed(RANDOM_SEED)
        random.shuffle(lst)
        
        size =  len(lst)
        test_size = int(size/PERCENTAGE)
        print(f"Number of training questions in miniGQA: {size}\n{test_size} of which is testing ({PERCENTAGE}%)")
        
        test_questions = lst[:test_size]
        train_questions = lst[test_size:]
        
        #Save training/testing questions
        for current_set_questions, current_json, saving_path, set_name in [(train_questions, final_train_json, output_training_question_ids_path, "Training"),
                                                                           (test_questions,  final_test_json,  output_testing_question_ids_path,  "Testing")]:
            pbar = tqdm(total=len(current_set_questions))
            for question_id in current_set_questions:
                imageId = train_json[question_id]["imageId"]
                if imageId in id_images_in_miniGQA: # Just include pictures referenced in miniGQA
                    current_json[question_id] = {"question": train_json[question_id]["question"],
                                                 "answer": train_json[question_id]["answer"],
                                                 "imageId": imageId,
                                                 "group": train_json[question_id]["groups"]["global"],
                                                 "types": train_json[question_id]["types"]}
                # del current_json[question_id] #Optional: delete image from dict to save memory
                pbar.update()
            pbar.close()
        
            with open(saving_path, "w") as f:
                json.dump(current_json, f)
                
            print(f"Saved {set_name} questions!")
    
    print("\nFinished!")
