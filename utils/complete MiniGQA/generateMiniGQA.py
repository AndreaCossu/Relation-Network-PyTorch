import h5py
import json
import os
import sys
import numpy as np
from silx.io.dictdump import dicttoh5
from silx.io.dictdump import h5todict
from tqdm import tqdm
"""
Este código lo usamos para crear un archivo con las features de las imagenes/objetos de miniGQA a partir de los archivos con features de todo GQA

outputs:
    miniGQA_objectFeatures.h5
    miniGQA_imageFeatures.h5
"""

def get_resnet_features(image_index):
    h5_index = gqa_spatial_info[image_index]["idx"]

    h5_file = gqa_spatial_info[image_index]["file"]
    h5_path = f"{resnet_features_filepath}/gqa_spatial_{str(h5_file)}.h5"

    h5 = h5py.File(h5_path, 'r')

    features = h5["features"][h5_index]
    return features

def get_features(image_index):
    objects_num = gqa_object_info[image_index]["objectsNum"]

    h5_file = gqa_object_info[image_index]["file"]
    h5_index = gqa_object_info[image_index]["idx"]

    h5_path = f"{features_filepath}/gqa_objects_{str(h5_file)}.h5"
    h5 = h5py.File(h5_path, 'r')

    features = h5["features"][h5_index]

    return objects_num, features[:objects_num]


def get_indexes(images_path):
    indexes = os.listdir(images_path)
    print(f"Number of images: {len(indexes)}")
    return [index.split(".")[0] for index in indexes]


def add_to_dict(dictionary, file_path, h5_path):
    create_ds_args = {'shuffle': False,
                      'fletcher32': False}

    #print("save to: " + h5_path)
    dicttoh5(dictionary, file_path, h5path=h5_path, mode="a",
             create_dataset_args=create_ds_args)

def load_dict(file_path, h5_path):
    dictionary = h5todict(file_path, h5_path)
    return dictionary


if __name__ == "__main__":
    images_path = "./images"
    features_filepath = "./object_features"
    resnet_features_filepath = "./image_features"
    gqa_object_info_path = "./object_features/gqa_objects_info.json"
    gqa_spatial_info_path = "./image_features/gqa_spatial_info.json"

    miniGQA_objectFeatures_path = "./output/miniGQA_objectFeatures.h5"
    miniGQA_imageFeatures_path = "./output/miniGQA_imageFeatures.h5"

    EXTRACT_FULL_IMAGE_FEATURES = True
    EXTRACT_OBJECTS_FEATURES = False

    if EXTRACT_OBJECTS_FEATURES:
        print("Starting extraction of object features")
        with open(gqa_object_info_path) as f:
            gqa_object_info = json.load(f)

        id_images = get_indexes(images_path)

        endvalue = len(id_images)
        pbar = tqdm(total=endvalue)

        paths_dict = {"paths": id_images}
        add_to_dict(paths_dict, miniGQA_objectFeatures_path, "/images_id")

        for id_image in id_images:
            num_obj, features_obj = get_features(id_image)
            features_dict = {"features": features_obj, "objectNum": num_obj}
            add_to_dict(features_dict, miniGQA_objectFeatures_path, id_image)
            pbar.update()
       
        # Open data
        # load_feature_image = load_dict(miniGQA_objectFeatures_path, aux_id)
        # print(load_feature_image)
        # print(type(load_feature_image))
        pbar.close()




    if EXTRACT_FULL_IMAGE_FEATURES:
        print("Starting extraction of image features")

        with open(gqa_spatial_info_path) as f:
            gqa_spatial_info = json.load(f)

        id_images = get_indexes(images_path)

        endvalue = len(id_images)
        pbar = tqdm(total=endvalue)

        paths_dict = {"paths": id_images}
        add_to_dict(paths_dict, miniGQA_imageFeatures_path, "/images_id")
        
        for id_image in id_images:
            feature_img = get_resnet_features(id_image)
            features_dict = {"features": feature_img}
            add_to_dict(features_dict, miniGQA_imageFeatures_path, id_image)
            pbar.update()

        pbar.close()

        # Open data
        # load_feature_image = load_dict("./output/img_features.h5", id_image)
        # print(load_feature_image['features_resnet'])


    print("\nFinished!")
