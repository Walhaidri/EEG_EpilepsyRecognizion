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
# eeg_ids = np.random.randint(0, len(y_data), 9)
# for ind, eeg_id in enumerate(eeg_ids):
#     plt.subplot(3, 3, ind+1)
#     plt.plot(x_data_arr[eeg_id, :])
#     plt.title(cls_names[y_data[eeg_id]])
# plt.show()

# data visualiziation by classes
seizure_cls = data[data['y'] == 1]
EEG_tumor_area = data[data['y'] == 2]
EEG_healthy_brain_area = data[data['y'] == 3]
closed_eyes = data[data['y'] == 4]
open_eyes = data[data['y'] == 5]

# cls_statistics
cls_statistics = [len(seizure_cls), len(EEG_tumor_area), len(EEG_healthy_brain_area), len(closed_eyes), len(open_eyes)]
print(f'Dataset contains about {x_data_arr.shape[0]} distribueted as following: ')
for i in range(5):
    print(f'examples of {cls_names[i]} = {cls_statistics[i]}')

seizure_cls = np.array(seizure_cls)
EEG_tumor_area = np.array(EEG_tumor_area)
EEG_healthy_brain_area = np.array(EEG_healthy_brain_area)
closed_eyes = np.array(closed_eyes)
open_eyes = np.array(open_eyes)

# normalization
# seizure_cls = min_max_norm2D(seizure_cls)
# EEG_tumor_area = min_max_norm2D(EEG_tumor_area)
# EEG_healthy_brain_area = min_max_norm2D(EEG_healthy_brain_area)
# closed_eyes = min_max_norm2D(closed_eyes)
# open_eyes = min_max_norm2D(open_eyes)


# classes distribution
bins = 10
# plt.subplot(3, 2, 1)
# plt.hist(np.mean(seizure_cls[:, :-1], axis=0), bins)
# plt.title(cls_names[0])
# plt.subplot(3, 2, 2)
# plt.hist(np.mean(EEG_tumor_area[:, :-1], axis=0), bins)
# plt.title(cls_names[1])
# plt.subplot(3, 2, 3)
# plt.hist(np.mean(EEG_healthy_brain_area[:, :-1], axis=0), bins)
# plt.title(cls_names[2])
# plt.subplot(3, 2, 4)
# plt.hist(np.mean(closed_eyes[:, :-1], axis=0), bins)
# plt.title(cls_names[3])
# plt.subplot(3, 2, 5)
# plt.hist(np.mean(open_eyes[:, :-1], axis=0), bins)
# plt.title(cls_names[4])
# plt.suptitle('Histogram of averaged and normalized data')
# plt.show()

# Соотношение эпиллепсии к другим классам
# plt.figure(20)
# plt.subplot(2, 2, 1)
# plt.hist(np.mean(seizure_cls[:, :-1], axis=0), bins, label='seizure_cls')
# plt.hist(np.mean(open_eyes[:, :-1], axis=0), bins, label='open_eyes')
# plt.legend()
#
# plt.subplot(2, 2, 2)
# plt.hist(np.mean(seizure_cls[:, :-1], axis=0), bins, label='seizure_cls')
# plt.hist(np.mean(closed_eyes[:, :-1], axis=0), bins, label='closed_eyes')
# plt.legend()
#
# plt.subplot(2, 2, 3)
# plt.hist(np.mean(seizure_cls[:, :-1], axis=0), bins, label='seizure_cls')
# plt.hist(np.mean(EEG_tumor_area[:, :-1], axis=0), bins, label='EEG_tumor_area')
# plt.legend()
#
# plt.subplot(2, 2, 4)
# plt.hist(np.mean(seizure_cls[:, :-1], axis=0), bins, label='seizure_cls')
# plt.hist(np.mean(EEG_healthy_brain_area[:, :-1], axis=0), bins, label='EEG_healthy_area')
# plt.legend()
#
# plt.suptitle('Соотношение эпилепсии к другим классам : данные усрдененныб НЕ нормированны')
# plt.show()


# ========
# plt.figure(10)
# plt.plot(np.mean(seizure_cls[:, :-1], axis=0), 'o-', label='seizure_cls')
# plt.plot(np.mean(closed_eyes[:, :-1], axis=0), 'o-', label='closed_eyes')
# plt.plot(np.mean(open_eyes[:, :-1], axis=0), 'o-', label='open_eyes')
# plt.plot(np.mean(EEG_tumor_area[:, :-1], axis=0), 'o-', label='EEG_tumor_area')
# plt.plot(np.mean(EEG_healthy_brain_area[:, :-1], axis=0), 'o-', label='EEG_healthy_area')
# plt.legend()
# plt.grid()
# plt.show()


# =========== corrlation
plt.figure(16)
feature_mtrx = np.zeros((5, 178))
feature_mtrx[0, :] = np.mean(seizure_cls[:, :-1], axis=0)
feature_mtrx[1, :] = np.mean(closed_eyes[:, :-1], axis=0)
feature_mtrx[2, :] = np.mean(open_eyes[:, :-1], axis=0)
feature_mtrx[3, :] = np.mean(EEG_tumor_area[:, :-1], axis=0)
feature_mtrx[4, :] = np.mean(EEG_healthy_brain_area[:, :-1], axis=0)
feature_mtrx = pd.DataFrame(feature_mtrx)
f_corr = feature_mtrx.corr()
f_corr = np.array(f_corr)
f_corr_2 = np.zeros_like(f_corr)
for i in range(f_corr.shape[0]):
    for j in range(f_corr.shape[1]):
        if f_corr[i, j] < 0:
            f_corr_2[i, j] = -1
sns.heatmap(f_corr_2)
plt.title('features correlation heatmap')
# plt.show()

# Соотношение эпиллепсии к другим классам
bins = 50
plt.figure(20)
plt.subplot(2, 2, 1)
plt.hist(np.max(np.abs(seizure_cls[:, 170:-1]), axis=1), bins, label='seizure_cls')
plt.hist(np.max(np.abs(open_eyes[:, 170:-1]), axis=1), bins, label='open_eyes')
plt.legend()
s = np.mean(open_eyes[:, :-1], axis=1)
print(s.shape)
plt.subplot(2, 2, 2)
plt.hist(np.max(np.abs(seizure_cls[:, 170:-1]), axis=1), bins, label='seizure_cls')
plt.hist(np.max(np.abs(closed_eyes[:, 170:-1]), axis=1), bins, label='closed_eyes')
plt.legend()

plt.subplot(2, 2, 3)
plt.hist(np.max(np.abs(seizure_cls[:, 170:-1]), axis=1), bins, label='seizure_cls')
plt.hist(np.max(np.abs(EEG_tumor_area[:, 170:-1]), axis=1), bins, label='EEG_tumor_area')
plt.legend()

plt.subplot(2, 2, 4)
plt.hist(np.max(np.abs(seizure_cls[:, 170:-1]), axis=1), bins, label='seizure_cls')
plt.hist(np.max(np.abs(EEG_healthy_brain_area[:, :-1]), axis=1), bins, label='EEG_healthy_area')
plt.legend()

plt.suptitle('Соотношение эпилепсии к другим классам : данные усрдененны, НЕ нормированны')
plt.show()

