import numpy as np


def min_max_norm1D(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def min_max_norm2D(mtrx):
    norm_mtrx = []
    mtrx = np.array(mtrx)
    for i in range(mtrx.shape[0]):
        norm_mtrx .append(min_max_norm1D(mtrx[i, :]))
    return np.array(norm_mtrx)

