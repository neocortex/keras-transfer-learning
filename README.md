# Transfer Learning

Simple examples for pre-trained Keras deep learning models on images based on
[this blogpost](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html).

Base model used is the [VGG16 model](https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3).

# Datasets

Datasets need to be stored in a subfolder in `./data/` where images belonging
to different classes go in separate subfolders. For instance:
`./data/inria/pos/` for the
[INRIA pedestrian dataset](http://pascal.inrialpes.fr/data/human/) people
images and `./data/inria/neg/` for negative class images.
