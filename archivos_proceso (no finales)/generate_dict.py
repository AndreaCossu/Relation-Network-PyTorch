import json
import sys

def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(
            arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

if __name__ == "__main__":
    new_train_json_path = "./data/miniGQA/new_train.json"
    new_valid_json_path = "./data/miniGQA/new_valid.json"
    output_dict_path = "./data/miniGQA/dict.txt"
    
    with open(new_train_json_path) as f:
        train_json = json.load(f)
        dictionary_set = set()
        
        value = 0
        endvalue = len(train_json.keys())
        progressBar(value, endvalue)
        
        for question_id in train_json.keys():
            answer = train_json[question_id]["answer"]
            dictionary_set.add(answer)
            
            value += 1
            progressBar(value, endvalue)
    
    with open(new_valid_json_path) as f:
        valid_json = json.load(f)
        dictionary_set = set()

        value = 0
        endvalue = len(valid_json.keys())
        progressBar(value, endvalue)

        for question_id in valid_json.keys():
            answer = valid_json[question_id]["answer"]
            dictionary_set.add(answer)

            value += 1
            progressBar(value, endvalue)
    
    print(f"Size dict: {len(list(dictionary_set))}")

    with open(output_dict_path, "w") as f:
        f.write("\n".join(list(dictionary_set)))
