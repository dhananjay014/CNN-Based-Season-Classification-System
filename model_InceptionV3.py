import os
import h5py

import matplotlib.pyplot as plt
import time, pickle, pandas

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers
from keras import applications
import pandas as pd
from keras.utils import np_utils

nb_classes = 4
class_name = {
    0: 'fall',
    1: 'rain',
    2: 'spring',
    3: 'winter'
}


# img_width, img_height = 224, 224
# img_width, img_height = 150, 150
img_width, img_height = 100, 100

train_data_dir =      '../all_images/train'
validation_data_dir = '../all_images/validation_1'
test_data_dir =       '../all_images/test'
nb_train_samples = 6368
nb_validation_samples = 1568
nb_test_samples = 2033


tensorboard_callback = TensorBoard(log_dir='./logs/', histogram_freq=0, write_graph=False, write_images=False)
# checkpoint_callback = ModelCheckpoint('./models/little_convnet_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


batch_size = 32

top_model_weights_path = 'fc_model_inceptionv3.h5'
epochs = 10


# In[8]:

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_features():
    datagen = ImageDataGenerator(rescale=1. / 255)
    print "making datagen"

    # building the InceptionV3 network from keras.applications
    model = applications.InceptionV3(include_top=False, weights='imagenet')
    print "model loaded"
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
#         class_mode=None,
        class_mode = 'categorical',
        shuffle=False)

    print "flow from directory done"
    
    # predicting the output using the training data
    features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)

    print "predict generator for train done"
    # Saving the output of the training data into .npy file
    np.save(open('features_train_inceptionv3.npy', 'w'),
            features_train)

    print "features saved"
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
#         class_mode=None,
        class_mode = 'categorical',
        shuffle=False)
    
    # predicting the output of the validation data
    features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    # saving the output of the validation data into .npy file
    np.save(open('features_validation_inceptionv3.npy', 'w'),
            features_validation)


def train_top_model():
    # getting the training data for the top model which is the output of save_features()
    train_data = np.load(open('features_train_inceptionv3.npy'))
    # getting training labels for the top model
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    
    # getting the validation data for the top model which is the output of save_features()
    validation_data = np.load(open('features_validation_inceptionv3.npy'))
    # getting validation labels for the top model
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))
    
    # Building our top model
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    
    # Compiling the top model
#     model.compile(optimizer='rmsprop',
#                   loss='binary_crossentropy', metrics=['accuracy'])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    
    train_labels = np_utils.to_categorical(train_labels, 4)
    validation_labels = np_utils.to_categorical(validation_labels, 4)
    # Running the training for 5te epochs
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
	      callbacks = [tensorboard_callback],
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


save_features()
train_top_model()

epochs = 30

from keras.layers import Input
from keras.models import Model
# input_t = Input(shape=(224,224,3))
input_t = Input(shape=(100,100,3))
# Building VGG16
Inception_model = applications.InceptionV3(weights='imagenet',include_top=False,input_tensor=input_t)
# Building top_model
top_model = Sequential()
top_model.add(Flatten(input_shape=Inception_model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(4, activation='softmax'))
#Loading to the top model weights that we learned earlier when we trained the top model
top_model.load_weights('fc_model_inceptionv3.h5')
model = Model(inputs=Inception_model.input, outputs=top_model(Inception_model.output))


len(model.layers)

#for layer in model.layers[:200]:
#    layer.trainable = False

model.compile(loss='binary_crossentropy',
#               optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              optimizer=optimizers.Adam(lr=0.00003, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks = [tensorboard_callback],
    validation_steps=nb_validation_samples // batch_size)


test_datagen_1 = ImageDataGenerator(rescale = 1. / 255)
test_generator = test_datagen_1.flow_from_directory(
        test_data_dir,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        class_mode = 'categorical')
    
model.evaluate_generator(
    test_generator,steps=nb_test_samples // batch_size)
