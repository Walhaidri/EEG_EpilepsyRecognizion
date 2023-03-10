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

cls_names = ['seizure activity',
             'EEG_tumor_area',
             'EEG_healthy_brain_area',
             'eyes closed',
             'eyes open']

# random data visiualizatin
eeg_ids = np.random.randint(0, len(y_data), 9)
for ind, eeg_id in enumerate(eeg_ids):
    plt.subplot(3, 3, ind+1)
    plt.plot(x_data_arr[eeg_id, :])
    plt.title(cls_names[y_data[eeg_id]])
plt.show()
