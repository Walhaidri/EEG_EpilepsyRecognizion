# multi-class classification with Keras
import pandas as pd
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

# load dataset
from sklearn.model_selection import train_test_split
import tensorflow as tf
from utils.preprocess import min_max_norm2D
from sklearn import metrics


# read the data
row_data_folder = r'dataset\row_dataset'
data_file = os.listdir(row_data_folder)[0]
data = pd.read_csv(os.path.join(row_data_folder, data_file), skiprows=0)
data = data.sort_values('y')
data = data.iloc[:, 1:]  # skip the 1 unimportant column
x_data, y_data = data.iloc[:, :-1], data.iloc[:, -1] - 1
x_data_arr = np.array(x_data)

# noramlization
x_norm_data = min_max_norm2D(x_data_arr)
input_data = x_norm_data
out_data = np.array(y_data)

# out_data =  tf.keras.utils.to_categorical(out_data, num_classes=5)

# dataset split to train, test
X_train, X_test, y_train, y_test = train_test_split(input_data, out_data, random_state=2023, test_size=0.20)
print(f'train subet size: \t {X_train.shape}')
print(f'test subet size: \t {X_test.shape}')
# plt.hist(y_test, 5)
# plt.show()
input_shape = X_train[0, :].shape

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(out_data)
encoded_Y = encoder.transform(out_data)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
print(dummy_y[0, :])

# define baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(178*2, input_dim=178, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

with tf.device('/device:gpu:0'):
    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True)
    results = cross_val_score(estimator, input_data, dummy_y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))