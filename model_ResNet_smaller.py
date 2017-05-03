
# coding: utf-8

# In[1]:

import os
import h5py

import matplotlib.pyplot as plt
import time, pickle, pandas

import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import backend
from keras import optimizers
from keras import applications
import pandas as pd
from keras.utils import np_utils

get_ipython().magic('matplotlib inline')


# In[2]:

nb_classes = 4
class_name = {
    0: 'fall',
    1: 'rain',
    2: 'spring',
    3: 'winter'
}


# In[4]:

# dimensions of our images.
# img_width, img_height = 224, 224
# img_width, img_height = 150, 150
img_width, img_height = 100, 100

train_data_dir =      '../all_images/train'
validation_data_dir = '../all_images/validation_1'
test_data_dir =       '../all_images/test'
nb_train_samples = 6370
nb_validation_samples = 1596
nb_test_samples = 2033


# In[7]:

# tensorboard_callback = TensorBoard(log_dir='./logs/little_convnet/', histogram_freq=0, write_graph=True, write_images=False)
# checkpoint_callback = ModelCheckpoint('./models/little_convnet_weights.{epoch:02d}-{val_acc:.2f}.hdf5', monitor='val_acc', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)


# In[6]:

batch_size = 32


# In[7]:

top_model_weights_path = 'fc_model.h5'
epochs = 10


# In[8]:

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# In[9]:

def save_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # building the VGG16 network from keras.applications
    model = applications.ResNet50(include_top=False, weights='imagenet')
    
    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
#         class_mode=None,
        class_mode = 'categorical',
        shuffle=False)
    
    # predicting the output using the training data
    features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    # Saving the output of the training data into .npy file
    np.save(open('features_train.npy', 'w'),
            features_train)

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
    np.save(open('features_validation.npy', 'w'),
            features_validation)


# In[10]:

def train_top_model():
    # getting the training data for the top model which is the output of save_features()
    train_data = np.load(open('features_train.npy'))
    # getting training labels for the top model
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))
    
    # getting the validation data for the top model which is the output of save_features()
    validation_data = np.load(open('features_validation.npy'))
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
    # Training the top model using training data and giving it validation data and validation labels to know the accuracy
    # Running the training for 5 epochs
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))
    model.save_weights(top_model_weights_path)


# In[ ]:

# save_features()


# In[ ]:

# train_top_model()


# In[45]:

def build_Resnet():
    # input_t = Input(shape=(224,224,3))
    input_t = Input(shape=(100,100,3))
    # Building VGG16
    ResNet_model = applications.ResNet50(weights='imagenet',include_top=False,input_tensor=input_t)
    # Building top_model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=ResNet_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(4, activation='softmax'))
    #Loading to the top model weights that we learned earlier when we trained the top model
    top_model.load_weights('fc_model.h5')
    model = Model(inputs=ResNet_model.input, outputs=top_model(ResNet_model.output))
    return model


# In[47]:

# model = build_Resnet()
# for layer in model.layers[:10]:
#     layer.trainable = False


# In[ ]:

def main():
    save_features()
    train_top_model()
    model = build_Resnet()
    for layer in model.layers[:10]:
        layer.trainable = False
    model.compile(loss='binary_crossentropy',
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
        validation_steps=nb_validation_samples // batch_size)


# In[ ]:

if __name__== '__main__':
    main()

