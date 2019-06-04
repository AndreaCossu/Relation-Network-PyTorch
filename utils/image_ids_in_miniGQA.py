import os
import json

def get_indexes(images_path):
    indexes = os.listdir(images_path)
    print(f"Number of files: {len(indexes)}")
    return [index.split(".")[0] for index in indexes]


if __name__ == "__main__":
    images_path = "./images"
    output_file = "./data/miniGQA/id_images_in_miniGQA.json"

    id_images_in_miniGQA = get_indexes(images_path)
    dictionary = {"id_images_in_miniGQA":id_images_in_miniGQA}
    with open(output_file, "w") as f:
        json.dump(id_images_in_miniGQA, f)