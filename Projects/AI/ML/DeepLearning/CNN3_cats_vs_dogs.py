import os
import glob
import shutil
import random
import itertools
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
# from keras.preprocessing.image import ImageDataGenerator

#=============== Data Preparartion =====================
path = os.getcwd()+'/data/dogsvscats' #used to get current directory
os.chdir(path)
if os.path.isdir('train/dog') is False:
    os.makedirs('train/dog')
    os.makedirs('train/cat')
    os.makedirs('valid/dog')
    os.makedirs('valid/cat')
    os.makedirs('test/dog')
    os.makedirs('test/cat')
    

    for c in random.sample(glob.glob('train/cat*'),500):
        shutil.move(c,'train/cat')
    for c in random.sample(glob.glob('train/dog*'),500):
        shutil.move(c,'train/dog')
    for c in random.sample(glob.glob('train/cat*'),100):
        shutil.move(c,'valid/cat')
    for c in random.sample(glob.glob('train/dog*'),100):
        shutil.move(c,'valid/dog')
    for c in random.sample(glob.glob('train/cat*'),50):
        shutil.move(c,'test/cat')
    for c in random.sample(glob.glob('train/dog*'),50):
        shutil.move(c,'test/dog')

train_path = path + '/train'
valid_path = path + '/valid'
test_path = path + '/test'

# # Data Processing
# # Shuffle is false for test set because we want to look at unshuffled labels
# train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
# .flow_from_directory(directory=train_path,target_size=(224,224),classes=['cat', 'dog'], batch_size=10)
# test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
# .flow_from_directory(directory=test_path,target_size=(224,224),classes=['cat', 'dog'],shuffle=False, batch_size=10)
# valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
# .flow_from_directory(directory=valid_path,target_size=(224,224),classes=['cat', 'dog'], batch_size=10)

# assert train_batches.n == 1000
# assert test_batches.n == 100
# assert valid_batches.n == 200
# assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes

# # plots images with labels within jupyter notebook
# def plotImages(images,labels):
#     fig, axes = plt.subplots(2,5,figsize=(20,10))
#     axes = axes.flatten()
#     for img, ax, label in zip(images,axes,labels):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()

# imgs, labels = next(train_batches)

# plotImages(imgs,labels)
# print(labels)

