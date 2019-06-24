import json
import h5py
from skimage import io
import os
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy
from random import shuffle


def get_bboxes(image_index):
    objects_num = gqa_object_info[image_index]["objectsNum"]

    h5_file = gqa_object_info[image_index]["file"]
    h5_index = gqa_object_info[image_index]["idx"]

    h5_path = f"{features_filepath}/gqa_objects_{str(h5_file)}.h5"
    h5 = h5py.File(h5_path, 'r')

    bboxes = h5["bboxes"][h5_index]
    return bboxes[:objects_num]


def convert_points_to_box(points, color, alpha):
    # upper_left_point = (points[0], points[1])
    lower_left_point = (points[0], points[3])
    width = points[2] - points[0]
    height = points[1] - points[3]
    text_pos = (lower_left_point[0]+width/2, lower_left_point[1]+height/2)
    return Rectangle(lower_left_point, width, height, ec=(*color, 1),
                     fc=(*color, alpha)), text_pos

if __name__ == "__main__":
    images_path = "./images"
    features_filepath = "./object_features"
    gqa_object_info_path = "./object_features/gqa_objects_info.json"
    id_images_in_miniGQA_path = "./id_images_in_miniGQA.json"
    MAX_OBJECTS = 5
    
    with open(id_images_in_miniGQA_path, "r") as f:
        id_images_in_miniGQA = json.load(f)

    with open(gqa_object_info_path) as f:
        gqa_object_info = json.load(f)

    shuffle(id_images_in_miniGQA)

    for image_id in id_images_in_miniGQA:
        bboxes = get_bboxes(image_id)
        # print(f"bboxes: {bboxes}")
        image = io.imread(os.path.join(images_path, image_id + ".jpg"))
        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1 = plt.subplot(121)
        io.imshow(image)
        ax1.autoscale(enable=True)
        ax1.title.set_text(f'{MAX_OBJECTS} objects')
        ax1.set_axis_off()
        for idx, bbox in enumerate(bboxes[:MAX_OBJECTS]):
            # print(f"bbox: {bbox}")
            color = numpy.random.rand(3)
            box, text_pos = convert_points_to_box(
                bbox, color, .4)
            ax1.text(text_pos[0], text_pos[1],
                    f"obj {idx+1}", bbox=dict(facecolor=color, alpha=0.5))
            ax1.add_patch(box)
        ax2 = plt.subplot(122)
        ax2.title.set_text(f'Original image')
        ax2.set_axis_off()
        io.imshow(image)
        plt.show()

