import h5py
import json
import os
import sys
import numpy as np
from tqdm import tqdm
from torch import nn
import torchvision.models as models
from skimage import io, transform
from skimage.color import gray2rgb
from skimage.util import crop, pad     
from matplotlib import pyplot as plt
import  torch


if __name__ == "__main__":

    # --- set device ---
    if torch.cuda.is_available():
        print('Using ', torch.cuda.device_count() ,' GPU(s)')
        mode = 'cuda'
    else:
        print("WARNING: No GPU found. Using CPUs...")
        mode = 'cpu'
    device = torch.device(mode)


    id_images_in_miniGQA_path = "./data/id_images_in_miniGQA.json"
    images_miniGQA_path = "./data/img/"
    train_graph_path = "./data/sceneGraphs/train_sceneGraphs.json"
    val_graph_path = "./data/sceneGraphs/val_sceneGraphs.json"

    save_features_path = "./data/GQA_objects_features/"

    resnet = models.resnet50().to(device)
    
    print("reading ids...")
    with open(id_images_in_miniGQA_path, "r") as f:
        image_ids = json.load(f)

    print("reading train graph...")    
    with open(train_graph_path, "r") as f:
        train_graph = json.load(f)

    print("reading val graph...")     
    with open(val_graph_path, "r") as f:
        val_graph = json.load(f)

    seen_id = []
    not_seen = []

    # --- GET valid image ids ---
    print("searching for valid ids...")
    #for image_id in image_ids[0:20]:
    for image_id in image_ids:
        if image_id in train_graph:
            seen_id.append(image_id)

        elif image_id in val_graph:
            seen_id.append(image_id)

        else:
            not_seen.append(image_id)
            continue

    # --- GET corp bb ---
    print("getting feature of objects of each image ...")
    pbar = tqdm(total=len(seen_id))
    for image_id in seen_id:

        if image_id in train_graph:
            scene = train_graph[image_id]

        else: #image_id in val_graph:
            scene = val_graph[image_id]

        # ---- GET and CROP Image ---
        img_name = images_miniGQA_path + image_id + ".jpg"
        image = io.imread(img_name)
        if len(image.shape) == 2:
            image = gray2rgb(image)

        objects_img = []
        for bb_number in scene['objects']:
            bb = scene['objects'][bb_number]
            object_img = crop(image, ((bb['y'], image.shape[0]-(bb['y']+bb['h'])), (bb['x'], image.shape[1]-(bb['x']+bb['w'])), (0,0) ), copy=False)
            object_resize = transform.resize(object_img, (224, 224, 3), anti_aliasing=False)
            objects_img.append(object_resize)        
        
        objects_tensor = torch.from_numpy(np.array(objects_img)).float().to(device)
        objects_tensor = objects_tensor.permute(0,3,1,2)
        
        objects_features = resnet(objects_tensor)
        # print(f"objects_features.size() -> {objects_features.size()}")
        # torch.Size([Num_objects, 1000])
        
        torch.save(objects_features, save_features_path + str(image_id) + '.pt')
        
        # --- Load tensor example ---
        #objects_features2 = torch.load( save_features_path + str(image_id) + '.pt')
        #print(f"objects_features2.size() -> {objects_features2.size()}")

        pbar.update()
    pbar.close()


