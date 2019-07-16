#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.metrics import confusion_matrix

from tensorflow import keras
layers = keras.layers
models = keras.models


# In[2]:


import os
os.chdir('C:\\Analytics\\Text mining\\bbc-text')


# In[3]:


data = pd.read_csv('bbc-text.csv')


# In[4]:


data.head()


# In[5]:


data['category'].value_counts()


# In[6]:


train_size = int(len(data) * 0.8)
print('Train size: %d' % train_size)
print('Test size: %d' % (len(data) - train_size))


# In[7]:


def train_test_split(data,train_size):
    train = data[:train_size]
    test = data[train_size:]
    return train,test


# In[8]:


train_cat, test_cat = train_test_split(data['category'], train_size)
train_text, test_text = train_test_split(data['text'],train_size)


# In[9]:


max_words = 1000
tokenize = keras.preprocessing.text.Tokenizer(num_words=max_words,
                                             char_level=False)


# In[10]:


tokenize.fit_on_texts(train_text) # fit tokenize to our training text data
x_train = tokenize.texts_to_matrix(train_text)
x_test = tokenize.texts_to_matrix(test_text)


# In[11]:


# use sklearn utility to convert label strings to number index
encoder = LabelEncoder()
encoder.fit(train_cat)
y_train = encoder.transform(train_cat)
y_test = encoder.transform(test_cat)


# In[12]:


# Converts the labels to a one-hot representation
num_classes = np.max(y_train) + 1
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# In[13]:


# lets check the dimension
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)


# # Train the Model

# In[14]:


batch_size = 32
epochs = 2
drop_ratio = 0.5


# In[15]:


model = models.Sequential()
model.add(layers.Dense(512, input_shape=(max_words,)))
model.add(layers.Activation('relu'))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy'])


# In[16]:


# lets fit the model
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_split=0.1)


# In[17]:


# evaluate the model
score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=1)
print('Test loss:', score[0])
print('Test accuracy', score[1])


# # Hyper parameter tuning

# In[18]:


def run_experiment(batch_size, epochs, drop_ratio):
    print('batch size: {}, epochs:{}, drop_ratio:{}'.format(batch_size, epochs,drop_ratio))
    
    model = models.Sequential()
    model.add(layers.Dense(512, input_shape=(max_words,)))
    
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(drop_ratio))
    model.add(layers.Dense(num_classes))
    model.add(layers.Activation('softmax'))
    
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size= batch_size,
                        epochs=epochs,
                        verbose=0,
                        validation_split=0.1)
    
    score = model.evaluate(x_test, y_test, batch_size=batch_size,verbose=0)
    
    print('\t Test loss:', score[0])
    print('\t Test accuracy:', score[1])
    


# In[19]:


batch_size = 16
epochs = 4
drop_ratio = 0.4
run_experiment(batch_size, epochs, drop_ratio)


# In[20]:


# lets make some prediction
text_labels = encoder.classes_

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    
    predicted_label = text_labels[np.argmax(prediction)]
    
    print(test_text.iloc[i][:50], "...")
    print('Actual label:' + test_cat.iloc[i])
    print('Predicted label:' + predicted_label + "\n")


# In[21]:


# lets visulalize the confusion matrix

y_softmax = model.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)
    
for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)


# In[22]:


def plot_confusion_matrix(cm,classes, title='Confusion matrix', cmap=plt.cm.Blues):
    
    cm = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=22)
    plt.yticks(tick_marks, classes, fontsize=22)
    
    fmt = '.2f'
    thresh = cm.max() / 2
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.ylabel('True label', fontsize=25)
        plt.xlabel('Predicted label', fontsize=25)


# In[23]:


cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(24,20))
plot_confusion_matrix(cnf_matrix, classes=text_labels, title='Confusion matrix')
plt.show()


# In[ ]:




