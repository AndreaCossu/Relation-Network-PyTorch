import json
import sys
import h5py
from nltk.tokenize import RegexpTokenizer


def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(
            arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()
        
def keys(f):
    return [key for key in f.keys()]

def generate_questions_dict(training_questions_path, testing_questions_path, validation_questions_path, output_dict_path):
    print("Generating Question dictionary")
    dictionary_set = set()
    
    tokenizer = RegexpTokenizer(r'\w+')
    #Scan training questions
    training_questions_h5 = h5py.File(training_questions_path, 'r')
    value = 0
    # print("antes de training_keys")
    slash = training_questions_h5[training_questions_h5.name]
    print(f"slash.name: {slash.name}")
    training_keys = keys(training_questions_h5)
    # print("despues de training_keys")
    # endvalue = len(training_keys)
    # print("antes del progress bar")
    # progressBar(value, endvalue)
    # print("antes del for")
    for question_id in training_keys:
        print("entro al for")
        question = training_questions_h5[question_id]["question"]
        words = tokenizer.tokenize(question)
        for word in words:
            dictionary_set.add(word)
        value += 1
        progressBar(value, endvalue)
    print("se paso el for por donde quiso")
    
    #Scan testing questions
    with h5py.File(testing_questions_path, 'r') as testing_questions_h5:
        value = 0
        endvalue = 2000000
        progressBar(value, endvalue)
        for question_id in training_questions_h5.keys():
            question = testing_questions_h5[question_id]["question"]
            words = tokenizer.tokenize(question)
            for word in words:
                dictionary_set.add(word)
            value += 1
            progressBar(value, endvalue)

    #Scan validation questions
    with open(validation_questions_path) as f:
        validation_json = json.load(f)
    value = 0
    endvalue = 2000000
    progressBar(value, endvalue)
    for question in validation_json.keys():
        question = question["question"]
        words = tokenizer.tokenize(question)
        for word in words:
            dictionary_set.add(word)
        value += 1
        progressBar(value, endvalue)
    
    output = list(dictionary_set)
    print(f"Size dict: {len(output)}")
    
    with open(output_dict_path, "w") as f:
        json.dump(output, f)
    
    print("Diccionario de preguntas generado!")
    return output
        
def generate_answers_dict(training_questions_path, testing_questions_path, validation_questions_path, output_dict_path):
    print("Generating Answers dictionary")
    dictionary_set = set()
    
    #Scan training questions
    with h5py.File(training_questions_path, 'r') as training_questions_h5:
        value = 0
        endvalue = 2000000
        progressBar(value, endvalue)
        for question_id in training_questions_h5.keys():
            answer = training_questions_h5[question_id]["answer"]
            dictionary_set.add(answer)
            value += 1
            progressBar(value, endvalue)
    
    #Scan testing questions
    with h5py.File(testing_questions_path, 'r') as testing_questions_h5:
        value = 0
        endvalue = 2000000
        progressBar(value, endvalue)
        for question_id in training_questions_h5.keys():
            answer = testing_questions_h5[question_id]["answer"]
            dictionary_set.add(answer)
            value += 1
            progressBar(value, endvalue)

    #Scan validation questions
    with open(validation_questions_path) as f:
        validation_json = json.load(f)
    value = 0
    endvalue = 2000000
    progressBar(value, endvalue)
    for question in validation_json.keys():
        answer = question["answer"]
        dictionary_set.add(answer)
        value += 1
        progressBar(value, endvalue)
    
    output = list(dictionary_set)
    
    with open(output_dict_path, "w") as f:
        json.dump(output, f)
    
    print("Diccionario de respuestas generado!")
    
    return output

def load_dict(dict_path):
    print(f"loading_dict in {dict_path}")
    with open(dict_path) as f:
        dictionary = json.load(f)
    return dictionary