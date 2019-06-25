#%%

import h5py
import json
import os
import sys
from silx.io.dictdump import dicttoh5
from silx.io.dictdump import h5todict
import numpy as np
from tqdm import tqdm

#%%

if __name__ == "__main__":

    id_images_in_miniGQA_path = "./data/miniGQA/id_images_in_miniGQA.json"
    
    train_graph_path = "./data/miniGQA/sceneGraphs/train_sceneGraphs.json"
    val_graph_path = "./data/miniGQA/sceneGraphs/val_sceneGraphs.json"

    print("reading ids...")
    with open(id_images_in_miniGQA_path, "r") as f:
        image_ids = json.load(f)
    
    print("reading train graph...")    
    with open(train_graph_path, "r") as f:
        train_graph = json.load(f)
    
    print("reading val graph...")     
    with open(val_graph_path, "r") as f:
        val_graph = json.load(f)

    objects_num = []
    relation_num = []
    object_ids_set = set()
    
    object_mean_one_std = 25
    relation_mean_one_std = 106
    objects_over = 0 
    relations_over = 0 
    
    pbar = tqdm(total=len(image_ids))
    seen = 0
    for image_id in image_ids:
        
        if image_id in train_graph:
            scene = train_graph[image_id]
            seen += 1
            
            
        elif image_id in val_graph:
            scene = val_graph[image_id]
            seen += 1
            
        else:
            print(f"Error! image id {image_id} not found in either graph") 
            continue
        
        relations = 0
        for object in scene["objects"]:
            object_ids_set.add(object)
            relations += len(scene["objects"][object]["relations"])
        objects = len(scene["objects"])
        
        if relations >= relation_mean_one_std:
            relations_over += 1
    
        if objects >= object_mean_one_std:
            objects_over += 1
        
        objects_num.append(objects)
        relation_num.append(relations)
        
        pbar.update()
    pbar.close()
    
    objects_mean = np.mean(objects_num)
    objects_std = np.std(objects_num)
    relations_mean = np.mean(relation_num)
    relations_std = np.std(relation_num)
    print(f"Imágenes vistas: {seen}/{len(image_ids)} = {seen*100/len(image_ids)}%")
    print("OBJECTOS:")
    print(f"promedio: {objects_mean}")
    print(f"desviacion: {objects_std}")
    print(f"Objetos distintos: {len(object_ids_set)}")
    print("_______________________")
    print("RELATIONS")
    print(f"promedio: {relations_mean}")
    print(f"desviacion: {relations_std}")
    
#%%
    print(f"over relations: {100-objects_over*100/len(objects_num)}")
    print(f"over objects: {100-relations_over*100/len(relation_num)}")

#%%
