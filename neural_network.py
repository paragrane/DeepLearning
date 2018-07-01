# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils

np.random.seed(100)

NB_EPOCH = 200
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10
OPTIMIZER = SGD()
N_HIDDEN = 128
VALIDATION_SPLIT = 0.2 

(X_train, y_train), (X_test, y_test) = mnist.load_data()
RESHAPED =784
X_train = X_train.reshape(60000, RESHAPED)
X_test = X_test.reshape(10000, RESHAPED)
X_train = X_train.astype("float32")
X_test = X_test.astype("float32")


X_train = X_train / 255
X_test = X_test / 255
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "Test samples")

y_train = np_utils.to_categorical(y_train, NB_CLASSES)
y_test = np_utils.to_categorical(y_test, NB_CLASSES)


model=Sequential()
model.add(Dense(NB_CLASSES, input_shape=(RESHAPED,)))
model.add(Activation("softmax") )
model.summary()


model.compile(loss = "categorical_crossentropy", optimizer=OPTIMIZER, metrics=["accuracy"])


history = model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=NB_EPOCH, verbose=VERBOSE)

score = model.evaluate(X_test, y_test, verbose=VERBOSE)
print(score[0])
print(score[1])



