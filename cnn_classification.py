# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 18:53:39 2021

@author: BoosamG
"""

#Convolutional Neural Networks
#Import Libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


#Part 1 : Preprocessing the images
#Preprocessing the Train Set
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

train_generator_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


#Preprocess the Test Set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')


#Part 2: IMplementing CNN
#Initialize CNN
cnn = tf.keras.models.Sequential()

#Convolution Layer 1
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu', input_shape = [64,64,3]))

#Max Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))

#Convolution Layer 2
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3,activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2,strides=2))


#Flattening
cnn.add(tf.keras.layers.Flatten())

#Full Connection
cnn.add(tf.keras.layers.Dense(units = 128, activation = 'relu'))

#output Layer
cnn.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))


#Part 3: Training the CNN
#Compile CNN
cnn.compile(optimizer='adam', loss = 'binary_crossentropy', metrics= ['accuracy'])

#Train the CNN with training set and validation with test set
cnn.fit(x=train_generator_set , validation_data= test_generator_set, epochs= 25)


#Part 4: Predict the image
from keras.preprocessing import image
import numpy as np
image_pred =image.load_img('cat_new_1.jfif', target_size=[64,64])
image_pred =image.img_to_array(image_pred)
image_pred = np.expand_dims(image_pred, axis = 0)
result = cnn.predict(image_pred)
train_generator_set.class_indices
print(result)
if result[0][0] > 0.5:
    pred_value = 'dog'
else:
    pred_value = 'cat'

print(pred_value)


