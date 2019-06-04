import h5py
from silx.io.dictdump import dicttoh5
from silx.io.dictdump import h5todict

# Documentación: http://www.silx.org/doc/silx/0.5.0/modules/io/dictdump.html

def save_dict(dictionary, file_path, h5_path):
    create_ds_args = {'compression': "gzip",
                      'shuffle': True,
                      'fletcher32': True}

    dicttoh5(dictionary, file_path, h5path=h5_path,
             create_dataset_args=create_ds_args)
    print("file saved!")


def add_to_dict(dictionary, file_path, h5_path):
    create_ds_args = {'compression': "gzip",
                      'shuffle': True,
                      'fletcher32': True}

    dicttoh5(dictionary, file_path, h5path=h5_path, mode="a",
             create_dataset_args=create_ds_args)
    print("file saved!")

def load_dict(file_path, h5_path):
    dictionary = h5todict(file_path, h5_path)
    print("file loaded!")
    return dictionary


def save_dict_v2(dictionary, file_path):
    silx.io.dictdump.dump(dictionary, file_path, mode='w', fmat="hdf5")
    print("file saved!")


def load_dict_v2(file_path, h5_path):
    dictionary = silx.io.dictdump.load(file_path, fmat="hdf5")
    print("file loaded!")
    return dictionary


question_1 = {"imageId": "11111",
              "question": "como estas?",
              "answer": "bien"}

question_2 = {"imageId": "22222",
              "question": "vienes?",
              "answer": "si"}

question_3 = {"imageId": "33333",
              "question": "comes?",
              "answer": "si"}

question_4 = {"imageId": "44444",
              "question": "sales?",
              "answer": "no"}

dictionary = {"1":question_1, "2":question_2, "3":question_3, "4":question_4}
dictionary_with_batches = {"batch_1":{"1":question_1, "2":question_2}, "batch_2":{"3":question_3, "4":question_4}}

######## V1
#Simple dictionary
# save_dict(dictionary, "./file.h5", "/questions")
# simple_dict_v1 = load_dict("./file.h5", "/questions")
# print(f"simple_dict_v1: {simple_dict_v1}")

#Batch dictionary
# for key, value in dictionary_with_batches.items():
#     save_dict(value, "./batches_file.h5", key)
#     loaded_batch = load_dict("./batches_file.h5", key)
#     print(f"loaded_batch: {loaded_batch}")

#ESTO ES LO QUE MÁS ME TINCA. QUEDA PARSEAR LA LECTURA NOMÁS. PORQUE LOS DICCIONARIOS QUEDAN CON UNOS array CON LOS STRINGS
#Guardar cada image en un path distinto, pero tener un indice de los paths en un path conocido
for key, value in dictionary.items():
    add_to_dict(value, "./best_file.h5", key)
paths_dict = {"paths": list(dictionary.keys())}
add_to_dict(paths_dict, "./best_file.h5", "/image_ids")

loaded_paths_dict = load_dict("./best_file.h5", "image_ids")
loaded_paths = loaded_paths_dict["paths"]
print(loaded_paths)
for image_id in loaded_paths:
    image_id_path = image_id.decode("utf-8")
    image_dict = load_dict("./best_file.h5", image_id_path)
    print(image_dict)

######## V1

######## V2
#Simple dictionary v2
# save_dict(dictionary, "./file_2.h5", "/questions")
# simple_dict_v2 = load_dict("./file_2.h5", "/questions")
# print(f"simple_dict_v2: {simple_dict_v2}")
######## V2
