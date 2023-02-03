"""
Created on Tue Jan 31 14:33:47 2023

@author: waleed.al.haidri
"""
# ----import moduls and packages ----
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.preprocess import min_max_norm1D, min_max_norm2D

# read the data
row_data_folder = r'dataset\row_dataset'
data_file = os.listdir(row_data_folder)[0]
data = pd.read_csv(os.path.join(row_data_folder, data_file), skiprows=0)
print(data.describe())
data = data.iloc[:, 1:]  # skip the 1 unimportant column
x_data, y_data = data.iloc[:, :-1], data.iloc[:, -1] - 1
x_data_arr = np.array(x_data)

# noramlization
x_norm_data = min_max_norm2D(x_data_arr)

# check normalization
print(np.min(x_norm_data), np.max(x_norm_data))
print(x_norm_data.shape, x_data_arr.shape)
plt.plot(x_norm_data[100, :])
plt.show()


