import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.utils import np_utils #print_summary
import pandas as pd
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import keras.backend as K
#Application
import cv2
from keras.models import load_model
from collections import deque #queue

data = pd.read_csv('data.csv')
data = data.iloc[0:72000, :]
dataset = np.array(data)
np.random.shuffle(dataset)
X = dataset
Y = dataset
X = X[:,0:1024]
Y = Y[:, 1024]
X_train = X[0:70000, :]
X_train = X_train / 255.
X_test = X[70000:72000, :]
X_test = X_test / 255.

#Reshape
Y = Y.reshape(Y.shape[0], 1)
Y_train = Y[0:70000, :]
Y_train = Y_train.T
Y_test = Y[70000:72000, :]
Y_test = Y_test.T

print("number of training exaples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape :" + str(Y_test.shape))

image_x = 32
image_y = 32

label_encoder = LabelEncoder()
Y_train_new = label_encoder.fit_transform(np.array(Y_train.T))
Y_test_new = label_encoder.fit_transform(np.array(Y_test.T))
train_y = np_utils.to_categorical(Y_train_new)
test_y = np_utils.to_categorical(Y_test_new)
X_train = X_train.reshape(X_train.shape[0], image_x, image_y, 1)
X_train = np.asarray(X_train).astype(np.float64)
X_test = X_test.reshape(X_test.shape[0], image_x,image_y, 1 )
X_test = np.asarray(X_test).astype(np.float64)

print("X_train_shape: " + str(X_train.shape))
print("Y_train_shape: " + str(train_y.shape))
#print(np.unique(Y_train_new))

#Building a model

def keras_model(image_x,image_y):
    num_of_classes = 36
    model = Sequential()
    model.add(Conv2D(filters = 32, kernel_size = (5,5), input_shape=(image_x, image_y, 1), activation='relu'))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size = (5, 5), strides = (5, 5), padding = 'same'))
    model.add(Flatten())
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    filepath = "devanagari.h5"
    checkpoint1 = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_beat_only=True, mode='max')
    callbacks_list = [checkpoint1]

    return model, callbacks_list

model, callbacks_list = keras_model(image_x, image_y)

model.fit(X_train, train_y, validation_data=(X_test, test_y), epochs=3, batch_size=64, callbacks=callbacks_list)
scores = model.evaluate(X_test, test_y, verbose=0)
print("CNN Error: %.2f%%" % (100 - scores[1] * 100))
model.summary()
model.save('devanagiri.h5')

