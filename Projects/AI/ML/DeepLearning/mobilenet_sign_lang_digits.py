# imports

import os
import shutil
import random
import itertools
import numpy as np
from keras import Model
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
from keras.layers import Dense,GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

# Fine-tuning mobilenet with a dataset that it was not pretrained on

# 10 classes in this dataset of sign languages

# Organize the data on disk

path = os.getcwd() +'/data/signlangdigits' #used to get current directory
os.chdir(path)
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0,10): # iterate all over the directories and 
        shutil.move(f'{i}', 'train') # move them to train directory
        os.mkdir(f'valid/{i}') # make a sub-directory corresponding to where we are at e.g valid/0, valid/1
        os.mkdir(f'test/{i}') # make a sub-directory corresponding to where we are at e.g test/0, test/1 
    
        valid_samples = random.sample(os.listdir(f'train/{i}'),30) # 30 random samples from each train/dir  
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}',f'valid/{i}') # move the random sampled from train/dir to validation/dir

        test_samples = random.sample(os.listdir(f'train/{i}'),5) # 
        for k in test_samples:
            shutil.move(f'train/{i}/{k}',f'test/{i}')

train_path = path + '/train'
valid_path = path + '/valid'
test_path = path + '/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
.flow_from_directory(directory=train_path,target_size=(224,224), batch_size=10)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
.flow_from_directory(directory=test_path,target_size=(224,224),shuffle=False, batch_size=10)

valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
.flow_from_direct

assert train_batches.n == 1712
assert test_batches.n == 50
assert valid_batches.n == 300

# fine-tune mobilenet
model = tf.keras.applications.mobilenet.MobileNet()

model.summary()

def count_params(model):
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list())
                                    for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) 
                                    for v in model.trainable_weights])
    print(non_trainable_params,trainable_params)
    return {'non_trainable_params': non_trainable_params,'trainable_params': trainable_params}

params = count_params(model)
assert params['non_trainable_params'] == 21888
assert params['trainable_params'] == 4231976


x = model.layers[-6].output # 6th to last layer
g = GlobalAveragePooling2D()(x)
output = Dense(units=10,activation='softmax')(g)

model = Model(inputs=model.input, outputs=output)


for layer in model.layers[:-23]: # freeze all except the last 23 layers
    layer.trainable = False

model.summary()

params = count_params(model)
assert params['trainable_params'] == 1873930 
assert params['non_trainable_params'] == 1365184

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(x=train_batches,validation_data=valid_batches,epochs=30,verbose=2)

test_labels = test_batches.classes

predictions = model.predict(x=test_batches, verbose=0)

cm = confusion_matrix(y_true=test_labels, y_pred=predictions.argmax(axis=1))

cm

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['0','1','2','3','4','5','6','7','8','9']
plot_confusion_matrix(cm=cm, classes=cm_plot_labels,title='Confusion Matrix')

