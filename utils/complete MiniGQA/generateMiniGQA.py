import h5py
import json
import os
import sys

def progressBar(value, endvalue, bar_length=20):
        percent = float(value) / endvalue
        arrow = '-' * int(round(percent * bar_length)-1) + '>'
        spaces = ' ' * (bar_length - len(arrow))

        sys.stdout.write("\rPercent: [{0}] {1}%".format(arrow + spaces, int(round(percent * 100))))
        sys.stdout.flush()

def get_resnet_features(image_index):
    h5_index = gqa_object_info[image_index]["idx"]
    
    # TODO: assumes h5 is split
    h5_file = gqa_object_info[image_index]["file"]
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

if __name__ == "__main__":
    images_path = "./images"
    features_filepath = "./object_features"
    resnet_features_filepath = "./image_features"
    gqa_object_info_path = "./image_features/gqa_objects_info.json"
    gqa_spatial_info_path = "./object_features/gqa_spatial_info.json"

    EXTRACT_FULL_IMAGE_FEATURES = True
    EXTRACT_OBJECTS_FEATURES = False
    
    if EXTRACT_OBJECTS_FEATURES:
        with open(gqa_object_info_path) as f:
            gqa_object_info = json.load(f)
        
        with h5py.File("./output/miniGQA_objectFeatures.hdf5", "w") as miniGQA:
            id_images = get_indexes(images_path)
            
            endvalue = len(id_images)
            value = 0
            progressBar(value, endvalue)
            
            for id_image in id_images:
                image = miniGQA.create_group(id_image)
                objectNum, features = get_features(id_image)
                image["features"] = features
                image["objectNum"] = objectNum
                
                value += 1
                progressBar(value, endvalue)
    
    if EXTRACT_FULL_IMAGE_FEATURES:
        with open(gqa_spatial_info_path) as f:
            gqa_spatial_info = json.load(f)
        
        with h5py.File("./output/miniGQA_imageFeatures.hdf5", "w") as miniGQA:
            id_images = get_indexes(images_path)
            
            endvalue = len(id_images)
            value = 0
            progressBar(value, endvalue)
            
            for id_image in id_images:
                image = miniGQA.create_group(id_image)
                features = get_resnet_features(id_image)
                image["features"] = features
                
                value += 1
                progressBar(value, endvalue)

    print("\nFinished!")
