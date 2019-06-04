import json

"""
Este código lo usamos para filtrar las preguntas de GQA para obtener solo aquellas referentes a imagenes de miniGQA

output:
    new_valid_filtered.json
"""

#input
validation_questions_path = "./data/miniGQA/new_valid.json"
id_images_in_miniGQA_path = "./data/miniGQA/id_images_in_miniGQA.json"
#output
validation_questions_output_path = "./data/miniGQA/new_valid_filtered.json"


with open(id_images_in_miniGQA_path, "r") as f:
    id_images_in_miniGQA = json.load(f)
    id_images_in_miniGQA = set(id_images_in_miniGQA)
    print(f"number of ids in miniGQA: {len(id_images_in_miniGQA)}")

with open(validation_questions_path, "r") as validation_questions_file:
    validation_questions = json.load(validation_questions_file)

validation_questions_filtered = {}
kept_questions = 0
total_questions = 0
for question_id in validation_questions.keys():
    image_id = validation_questions[question_id]["imageId"]
    if image_id in id_images_in_miniGQA:
        validation_questions_filtered[question_id] = validation_questions[question_id]
        kept_questions += 1
        print(f"kept_questions: {kept_questions}/{total_questions}")
    total_questions += 1

with open(validation_questions_output_path, "w") as validation_questions_file:
    json.dump(validation_questions_filtered, validation_questions_file)
