import json
import os
import re
import threading
from os import makedirs, path, remove

import numpy as np
import pandas as pd
TIME_SERIES_LENGTH = 7500


def make_npz(datasets_path, output_filename, x_train_filename, y_train_filename):
    """
    Creates the npz file and deletes the x_train and y_train files
    """
    x_train_path = datasets_path + (x_train_filename)
    y_train_path = datasets_path + (y_train_filename)
    print('Creating %s.npz...' % output_filename, end='\r')
    csv2npz(x_train_path, y_train_path, datasets_path, output_filename)
    clear_line_str = '\033[K'
    print(clear_line_str+'Create '+output_filename+'.npz\tDone')
    # there is no more need to keep x_train and y_train files
    remove(x_train_path)
    remove(y_train_path)

def csv2npz(dataset_x_path, dataset_y_path, output_path, filename, labels_path='labels.json'):
    """Load input dataset from csv and create x_train tensor."""
    # Load dataset as csv
    x = pd.read_csv(dataset_x_path)
    y = pd.read_csv(dataset_y_path)

    # Load labels, file can be found in challenge description
    with open(labels_path, "r") as stream_json:
        labels = json.load(stream_json)

    m = x.shape[0]
    K = TIME_SERIES_LENGTH  # Can be found through csv

    # Create R and Z
    R = x[labels["R"]].values
    R = R.astype(np.float32)

    X = y[[f"{var_name}_{i}" for var_name in labels["X"]
           for i in range(K)]]
    X = X.values.reshape((m, -1, K))
    X = X.astype(np.float32)

    Z = x[[f"{var_name}_{i}" for var_name in labels["Z"]
           for i in range(K)]]
    Z = Z.values.reshape((m, -1, K))
#     Z = Z.transpose((0, 2, 1))
    Z = Z.astype(np.float32)

    np.savez(path.join(output_path, filename), R=R, X=X, Z=Z)


if __name__ == '__main__':
    make_npz('./', 'training', 'x_train.csv', 'y_train.csv')