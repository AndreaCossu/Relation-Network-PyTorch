import h5py
import json
import os
import sys
from silx.io.dictdump import dicttoh5
from silx.io.dictdump import h5todict
import numpy as np
from tqdm import tqdm


def load_dict(file_path, h5_path):
    dictionary = h5todict(file_path, h5_path)
    return dictionary

if __name__ == "__main__":

    miniGQA_objectFeatures_path = "./data/miniGQA/miniGQA_objectFeatures.h5"
    id_images_in_miniGQA_path = "./data/miniGQA/id_images_in_miniGQA.json"
    objectsNum_in_miniGQA_path = "./data/miniGQA/objectsNum_in_miniGQA_path.json"
    

    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = json.load(f)

    objects_num = []
    more_than_mu_1dv = 0
    less_than_mu_1dv = 0
    pbar = tqdm(total=len(id_images_in_miniGQA))
    for image_id in id_images_in_miniGQA:
        features_dict = load_dict(miniGQA_objectFeatures_path, image_id)
        # features = features_dict["features"]
        objectNum = features_dict["objectNum"]
        objects_num.append(objectNum)
        if objectNum > 42:
            more_than_mu_1dv += 1
        # print(f"objectNum {objectNum}")
        pbar.update()
    pbar.close()
    
    promedio = np.mean(objects_num)
    desviacion = np.std(objects_num)
    print(f"promedio: {promedio}")
    print(f"desviacion: {desviacion}")
    print(f"less_than_mu_1dv: {less_than_mu_1dv}")
    print(f"more_than_mu_1dv: {more_than_mu_1dv}")
    print(
        f"more_than_mu_1dv percentage: {more_than_mu_1dv/len(id_images_in_miniGQA)}")
    
    # with open(objectsNum_in_miniGQA_path, "w") as f:
    #     json.dump(objects_num, f)
