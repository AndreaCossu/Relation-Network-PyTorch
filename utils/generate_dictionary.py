import json
import sys
import h5py
from nltk.tokenize import RegexpTokenizer

"""
Este código se usa para genera un diccionario de palabras de preguntas y de respuestas a partir de las preguntas y respuestas de miniGQA
"""

def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(
            arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def generate_questions_dict(training_questions_path, testing_questions_path, validation_questions_path, output_dict_path):
    print("Generating Question dictionary")
    dictionary_set = set()
    
    tokenizer = RegexpTokenizer(r'\w+')
    longest_question_length = 0
    
    for path in [training_questions_path, testing_questions_path, validation_questions_path]:
        questions = load_dict(path)
        questions_ids = questions.keys()
        
        value = 0
        endvalue = len(questions_ids)
        progressBar(value, endvalue)
        
        for question_id in questions_ids:
            question = questions[question_id]["question"]
            question_length = len(question)
            if question_length > longest_question_length:
                longest_question_length = question_length
            words = tokenizer.tokenize(question)
            for word in words:
                dictionary_set.add(word)
            
            value += 1
            progressBar(value, endvalue)
    
    output = list(dictionary_set)
    print(f"\nSize dict: {len(output)}")
    print(f"\nLongest question length: {longest_question_length}")
    
    with open(output_dict_path, "w") as f:
        json.dump(output, f)
    
    print("\nDiccionario de preguntas generado!")
    return output
        
def generate_answers_dict(training_questions_path, testing_questions_path, validation_questions_path, output_dict_path):
    print("Generating Answers dictionary")
    dictionary_set = set()
    
    for path in [training_questions_path, testing_questions_path, validation_questions_path]:
        questions = load_dict(path)
        questions_ids = questions.keys()
        
        value = 0
        endvalue = len(questions_ids)
        progressBar(value, endvalue)
        
        for question_id in questions_ids:
            answer = questions[question_id]["answer"]
            dictionary_set.add(answer)
            
            value += 1
            progressBar(value, endvalue)
    
    output = list(dictionary_set)
    
    with open(output_dict_path, "w") as f:
        json.dump(output, f)
    
    print("\nDiccionario de respuestas generado!")
    
    return output

def load_dict(dict_path):
    # print(f"loading dict in {dict_path}")
    with open(dict_path) as f:
        dictionary = json.load(f)
    return dictionary