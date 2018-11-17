#!/usr/bin/env python

import os
import urllib
import zipfile
import numpy as np
import pandas as pd

try:
    from tensorflow.keras.datasets import cifar10
except ImportError:
    import tensorflow as tf
    cifar10 = tensorflow.keras.datasets.cifar10

def load_cifar4():
    dir_path = os.path.join('data','cifar10_4')

    if not os.path.exists(os.path.join(dir_path, 'y_val.npy')):
        os.makedirs(dir_path)
        # car, truck, frog, cat
        x_train, y_train, x_val, y_val = load_cifar10(classes=[1,9,3,6])
        np.save(os.path.join(dir_path, 'x_train.npy'), x_train)
        np.save(os.path.join(dir_path, 'y_train.npy'), y_train)
        np.save(os.path.join(dir_path, 'x_val.npy'), x_val)
        np.save(os.path.join(dir_path, 'y_val.npy'), y_val)
    else:
        x_train = np.load(os.path.join(dir_path, 'x_train.npy'))
        x_val = np.load(os.path.join(dir_path, 'x_val.npy'))
        y_train = np.load(os.path.join(dir_path, 'y_train.npy'))
        y_val = np.load(os.path.join(dir_path, 'y_val.npy'))

    return x_train, y_train, x_val, y_val

def load_cifar10(classes=None):
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()

    if classes is not None:
        train_idx = np.array([i in classes for i in y_train])
        x_train = x_train[train_idx]
        y_train = y_train[train_idx]
        val_idx = np.array([i in classes for i in y_val])
        x_val = x_val[val_idx]
        y_val = y_val[val_idx]

    return x_train, y_train, x_val, y_val

def load_sgemm():
    csv_path = os.path.join('data','sgemm_product.csv')

    if not os.path.exists(csv_path):
        print('Downloading dataset ...')
        url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00440/'\
            'sgemm_product_dataset.zip'

        os.makedirs('data', exist_ok=True)
        path = os.path.join('data','sgemm_product_dataset.zip')
        urllib.request.urlretrieve(url, path)

        print('Extracting files ...')
        with zipfile.ZipFile(path) as f:
            f.extractall('data')

    df = pd.read_csv(csv_path)
    df['target'] = df.iloc[:,-4:].mean(axis=1)
    X = df.iloc[:,:-5].values
    y = df.target.values

    return X.astype(float), y