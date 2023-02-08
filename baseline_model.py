"""
Created on Tue Jan 31 14:33:47 2023

@author: waleed.al.haidri
"""
# ----import moduls and packages ----
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import  metrics
from utils.preprocess import min_max_norm2D
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn import svm

# read the data
row_data_folder = r'dataset\row_dataset'
data_file = os.listdir(row_data_folder)[0]
data = pd.read_csv(os.path.join(row_data_folder, data_file), skiprows=0)
data = data.sort_values('y')
print(data.head())
data = data.iloc[:, 1:]  # skip the 1 unimportant column
x_data, y_data = data.iloc[:-4600, :-1], data.iloc[:-4600, -1] - 1
x_data_arr = np.array(x_data)

cls_names = ['seizure activity',
             'EEG_tumor_area',
             'EEG_healthy_brain_area',
             'eyes closed',
             'eyes open']

# noramlization
x_norm_data = min_max_norm2D(x_data_arr)
input_data = x_norm_data
out_data = np.array(y_data)
out_data = np.zeros(len(out_data))
out_data[2300:] = 1
print(input_data.shape, out_data)
# check normalization
print(np.min(x_norm_data), np.max(x_norm_data))
plt.plot(x_norm_data[100, :])
# plt.show()

# dataset split to train, test
X_train, X_test, y_train, y_test = train_test_split(input_data, out_data, random_state=2023, test_size=0.2)
print(f'train subet size: \t {X_train.shape[0]}')
print(f'test subet size: \t {X_test.shape[0]}')

# train LogisticRegression
log_reg_clf = LogisticRegression(multi_class='ovr')
log_reg_clf.fit(X_train, y_train)
predicted_logreg = log_reg_clf.predict(X_test)
clf_report = metrics.classification_report(y_test, predicted_logreg, output_dict=True)
macro_averaged_f1 = round(clf_report['macro avg']['f1-score'], 2)
print('macro_averaged_f1 for logistic regression : ', macro_averaged_f1)

# SVM OneVsRestClassifier
svm_model = OneVsRestClassifier(svm.SVC(random_state=42))
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

clf_report = metrics.classification_report(y_test, svm_pred, output_dict=False)
print(clf_report)
# macro_averaged_f1 = round(clf_report['macro avg']['f1-score'], 2)
# print('macro_averaged_f1 for SVM', macro_averaged_f1)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf = clf.fit(input_data, out_data)
tree_y_preds = clf.predict(X_test)
clf_report = metrics.classification_report(y_test, tree_y_preds, output_dict=True)
macro_averaged_f1 = round(clf_report['macro avg']['f1-score'], 2)
print('macro_averaged_f1 for DecisionTreeClassifier', macro_averaged_f1)

#======================================================
#train on truncated features
#===================================================
ftrs = np.loadtxt('features_2.txt')
non_corr_features = np.zeros((x_data_arr.shape[0], len(ftrs)))
for k, l in enumerate(ftrs):
    non_corr_features[:, k] = x_data_arr[:, int(l)]

processed_data_norm_data = min_max_norm2D(non_corr_features)
# dataset split to train, test
X_train, X_test, y_train, y_test = train_test_split(processed_data_norm_data, out_data, random_state=2023, test_size=0.2)
print(f'train subet size: \t {X_train.shape}')
print(f'test subet size: \t {X_test.shape}')

# SVM OneVsRestClassifier
svm_model = OneVsRestClassifier(svm.SVC(random_state=42))
svm_model.fit(X_train, y_train)
svm_pred = svm_model.predict(X_test)

clf_report = metrics.classification_report(y_test, svm_pred, output_dict=False)
print(clf_report)
# macro_averaged_f1 = round(clf_report['macro avg']['f1-score'], 2)
# print('macro_averaged_f1 for SVM', macro_averaged_f1)