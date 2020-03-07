#!/usr/bin/env python
# coding: utf-8

#  ###  libraries 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras import models
from keras import layers


# ### Binary representation

# In[104]:


from keras.datasets import reuters
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation


# In[139]:


# load reuters dataset
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

no_features = max(y_train) + 1

max_words = 20000

# one-hot encoding
tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

y_train = keras.utils.to_categorical(y_train, no_features)
y_test = keras.utils.to_categorical(y_test, no_features)

print(x_train[0])
print(len(x_train[0]))
print(max(x_train[0]))

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(x_train,  # features
                    y_train,  # target
                    batch_size=32,   # number of observations per batch
                    epochs=3,   # number of epochs
                    verbose=1,  # output
                    validation_split=0.1)   # test data

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# count for the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# visualize the loss history
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b--")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# ### Count representation

# In[140]:


# load reuters dataset
(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=None, test_split=0.2)

no_features = max(y_train) + 1

max_words = 20000

tokenizer = Tokenizer(num_words=max_words)
x_train = tokenizer.sequences_to_matrix(x_train, mode='count')
x_test = tokenizer.sequences_to_matrix(x_test, mode='count')

y_train = keras.utils.to_categorical(y_train, no_features)
y_test = keras.utils.to_categorical(y_test, no_features)

print(x_train[0])
print(len(x_train[0]))
print(max(x_train[0]))

model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history = model.fit(x_train,  # features
                    y_train,  # target
                    batch_size=32,   # number of observations per batch
                    epochs=3,   # number of epochs
                    verbose=1,  # output
                    validation_split=0.1)   # test data
score = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# get training and test loss histories
training_loss = history.history['loss']
test_loss = history.history['val_loss']

# count for the number of epochs
epoch_count = range(1, len(training_loss) + 1)

# visualize the loss history
plt.plot(epoch_count, training_loss, "r--")
plt.plot(epoch_count, test_loss, "b--")
plt.legend(["Training Loss", "Test Loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# In[ ]:




