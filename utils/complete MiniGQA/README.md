# Completar miniGQA

El dataset [mini GQA](http://datasetvqa.ing.puc.cl:443/) cuenta con las imágenes y sus preguntas asociadas, no tiene:

- Features Resnet de cada imagen
- Features Resnet de los objetos encontrados por Faster R-CNN

Este código permite generar los archivos `miniGQA_imageFeatures.hdf5` y `miniGQA_objectFeatures.hdf5` con esta información para las imágenes de miniGQA.

### Setup

Para generar `miniGQA_imageFeatures.hdf5` se requiere:

- Resnet features de todo GQA (32 gb). Ubicar en la carpeta `./image_features`.
- Imágenes miniGQA. Ubicar en la carpeta `./images`.

Para generar `miniGQA_objectFeatures.hdf5` se requiere:

- Object features de todo GQA. Ubicar los archivos `gqa_objects_X.h5` y `gqa_objects_info.json` en `./image_features `.
- Imágenes miniGQA. Ubicar en la carpeta `./images`.