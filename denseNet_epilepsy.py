"""
Created on Tue Jan 31 14:33:47 2023

@author: waleed.al.haidri
"""
# ----import moduls and packages ----
import tensorflow as tf
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from utils.preprocess import min_max_norm2D
from sklearn import metrics

#============================
# read the data
#============================

row_data_folder = r'dataset\row_dataset'
data_file = os.listdir(row_data_folder)[0]
data = pd.read_csv(os.path.join(row_data_folder, data_file), skiprows=0)
data = data.sort_values('y')
data = data.iloc[:, 1:]  # skip the 1 unimportant column
x_data, y_data = data.iloc[:-4600, :-1], data.iloc[:-4600, -1] - 1
x_data_arr = np.array(x_data)
cls_names = ['seizure activity', 'No seizure activity']

# noramlization
x_norm_data = min_max_norm2D(x_data_arr)
input_data = x_norm_data
out_data = np.array(y_data)
out_data = np.ones(len(out_data))
out_data[0:2300] = 0

# dataset split to train, test
X_train, X_test, y_train, y_test = train_test_split(input_data, out_data, random_state=2023, test_size=0.20)
print(f'train subet size: \t {X_train.shape}')
print(f'test subet size: \t {X_test.shape}')
plt.hist(y_train, 2)
plt.show()
input_shape = X_train[0, :].shape

#=======================
# creat model
#=====================
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=512, input_shape=input_shape, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='sigmoid'))
print(model.summary())

#=====================
# compile model
#=====================

model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
#=====================
# train
#=====================
history = model.fit(X_train, y_train, batch_size=23, validation_split=0.2,  epochs=100)

# save model with weights
model.save(r'trained_models\epiNet.h5')

# plot training history
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(r'results\model_plots\Model training')
plt.show()


# model evalute
test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print('\nTest accuracy:', test_acc)

#model test to get f1 score
y_pred = []
for i in range(len(y_test)):
    pred = model.predict(np.expand_dims(X_test[i, :], axis=0))
    y_pred.append(np.argmax(pred[0]))
print(metrics.classification_report(y_test, y_pred))

# import keras_tuner
# from tensorflow import keras
#
# def build_model(hp):
#   model = keras.Sequential()
#   model.add(keras.layers.Dense(
#       hp.Choice('units', [8, 16, 32, 64, 128, 256, 512]),input_shape=input_shape,
#       activation='relu'))
#
#   model.add(keras.layers.Dense(5, activation='softmax'))
#   model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
#                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#                 metrics=['accuracy'])
#   return model
#
# tuner = keras_tuner.RandomSearch(
#     build_model,
#     objective='val_loss',
#     max_trials=10)
#
# tuner.search(X_train, y_train, epochs=50, validation_data=(X_test, y_test))
# best_model = tuner.get_best_models()[0]