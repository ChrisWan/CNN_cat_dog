# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:23:47 2020

@author: wanke
"""


# Building your CNN
# Import libaries
import numpy as np
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing import image
from keras.layers import Dropout

# Initialising the CNN
classifier = Sequential()

# Step 1 Convolution; when first layer parameter input_shape is necessary read help for that
classifier.add(Convolution2D(filters = 32, kernel_size = 3, input_shape = (128,128,3), activation = 'relu'))

# Step 2 Pooling; reducing the size to make it less computial intensive and keep features; 2x2 is best for not loosing any information
classifier.add(MaxPooling2D(pool_size = 2))

# Step 6 Improving the test accuracy; adding a second convolutional layer; also possible: add another hidden layer
classifier.add(Convolution2D(filters = 32, kernel_size = 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2))

# Step 6.5 Common practice add another convolutional layer with 64 kernels/filters/feature detectors
classifier.add(Convolution2D(filters = 64, kernel_size = 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = 2))

# Step 3 Flattening; transform all max pooling windows in one big vector
classifier.add(Flatten())

# Step 4 Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.6))
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dropout(0.4))
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Step 5 Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the CNN to the image; image augmentation = enrich our images = more data to train on; it rotates etc our images for more variaty
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128), # dimension expected see above
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128), # dimension expected see above
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000/32, # divide by batch_size
                         epochs = 80,
                         validation_data = test_set,
                         validation_steps = 2000/32,
                         workers = 12,
                         max_queue_size = 100)

# make a single prediction
def make_single_prediction(location):
    img_width, img_height = 128, 128
    img = image.load_img(location, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    result = classifier.predict(img)
    # find out what class corresponds to what numerical value
    training_set.class_indices
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    print(prediction)

make_single_prediction('dataset/single_prediction/cat_or_dog_1.jpg')
make_single_prediction('dataset/single_prediction/cat_or_dog_2.jpg')
make_single_prediction('dataset/single_prediction/cat_or_dog_3.jpg')
make_single_prediction('dataset/single_prediction/cat_or_dog_4.jpg')
make_single_prediction('dataset/single_prediction/cat_or_dog_5.jpg')
make_single_prediction('dataset/single_prediction/cat_or_dog_6.jpg')
make_single_prediction('dataset/single_prediction/cat_or_dog_7.jpg')