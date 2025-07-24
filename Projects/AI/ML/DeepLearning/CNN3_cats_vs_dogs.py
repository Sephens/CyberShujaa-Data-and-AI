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
from keras.preprocessing.image import ImageDataGenerator

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

#===============Data Processing ============
# Shuffle is false for test set because we want to look at unshuffled labels
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
.flow_from_directory(directory=train_path,target_size=(224,224),classes=['cat', 'dog'], batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
.flow_from_directory(directory=test_path,target_size=(224,224),classes=['cat', 'dog'],shuffle=False, batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
.flow_from_directory(directory=valid_path,target_size=(224,224),classes=['cat', 'dog'], batch_size=10)

assert train_batches.n == 1000
assert test_batches.n == 100
assert valid_batches.n == 200
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes

# plots images with labels within jupyter notebook
def plotImages(images,labels):
    fig, axes = plt.subplots(2,5,figsize=(20,10))
    axes = axes.flatten()
    for img, ax, label in zip(images,axes,labels):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

imgs, labels = next(train_batches)

plotImages(imgs,labels)
print(labels)

#======================= Build and Train the Model ===================
# Padding same means the dimensionality of our images doesnt reduces after applying convolution
model = Sequential([
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding= 'same',  input_shape=(224,224,3)),
        MaxPool2D(pool_size=(2,2),strides=2),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding= 'same',),
        MaxPool2D(pool_size=(2,2),strides=2),
        Flatten(),
        Dense(2, activation='softmax'),
    ])

model.summary()   

model.compile(Adam(learning_rate=.0001),loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x=train_batches, validation_data=valid_batches,
           epochs=10, verbose=2)


#=================== Prediction ==============

test_imgs, test_labels = next(test_batches)
plotImages(test_imgs, test_labels)
print(test_labels)

# show the classes
test_batches.classes

# make prediction
# verbose is set at lowest since we are not interested in seeing the output of prediction process
pred = model.predict(x=test_batches, verbose=0) 

# round off the predictions and print
np.round(pred)

# visualize the predictions with a confusion matrix

cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(pred, axis=1))

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

cm_plot_labels = ['cat', 'dog']
plot_confusion_matrix(cm=cm_plot_labels, title='Confusion Matrix')

#===============Fine tunning our model using VGG16================

vgg16_model = keras.applications.vgg16.VGG16()

vgg16_model.summary()

def count_params(model):
    non_trainable_params = np.sum([np.prod(v.get_shape().as_list())
                                    for v in model.non_trainable_weights])
    trainable_params = np.sum([np.prod(v.get_shape().as_list()) 
                                    for v in model.trainable_weights])
    print(non_trainable_params,trainable_params)
    return {'non_trainable_params': non_trainable_params,'trainable_params': trainable_params}

params = count_params(vgg16_model)
assert params['non_trainable_params'] == 0
assert params['trainable_params'] == 138357544

model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

for layer in model.layers:
    layer.trainable = False

model.add(Dense(2, activation='softmax'))

model.summary()

params = count_params(vgg16_model)
assert params['non_trainable_params'] == 134260544
assert params['trainable_params'] == 4097000

#=============Train the fine-tuned VGG16 model ============

model.compile(Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics='accuracy')

model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

#=========== Predict using the fine-tuned VGG16 model ========
predictions = model.predict(test_batches, verbose=0)

test_batches.classes


#================ Visualize the Predictions of Fined-tuned VGG16 nodel ==============
cm = confusion_matrix(y_true=test_batches.classes,y_pred = np.argmax(predictions, axis=-1))

test_batches.class_indices

test_imgs, test_labels = next(test_batches)
plotImages(test_imgs,test_labels)

cm_plot_labels = ['cat','dog']
plot_confusion_matrix(cm, cm_plot_labels, title='Confusion Matrix')


# =================== Saving the Model =========

if os.path.isfile('vgg16_catvsdog.h5') is False:
    model.save('vgg16_catvsdog.h5')
