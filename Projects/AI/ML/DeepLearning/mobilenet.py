import keras
import numpy as np
from keras import utils
from keras.applications import imagenet_utils

mobile = keras.applications.mobilenet.MobileNet()

def prepare_image(file):
    img_path = 'data/images/'
    img = utils.load_img(img_path + file, target_size=(224, 224))
    img_array = utils.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

from IPython.display import Image
Image(filename='data/images/dog.jpg', width=300,height=200) 

preprocessed_image = prepare_image('dog.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

Image(filename='data/images/cat.jpg', width=300,height=200) 

preprocessed_image = prepare_image('cat.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

Image(filename='data/images/park.jpg', width=300,height=200) 

preprocessed_image = prepare_image('park.jpg')
predictions = mobile.predict(preprocessed_image)
results = imagenet_utils.decode_predictions(predictions)
results

