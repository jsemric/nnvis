#!/usr/bin/env python

import os
import numpy as np
from tensorflow.keras.datasets import cifar10

def get_cifar10_4():
    dir_path = os.path.join('data','cifar10_4')

    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        # car, truck, frog, cat
        x_train, y_train, x_val, y_val = prepare_cifar10(classes=[1,9,3,6])
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

def prepare_cifar10(classes=None):
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    if classes is not None:
        train_idx = np.array([i in classes for i in y_train])
        x_train = x_train[train_idx]
        y_train = y_train[train_idx]
        val_idx = np.array([i in classes for i in y_val])
        x_val = x_val[val_idx]
        y_val = y_val[val_idx]

    return x_train, y_train, x_val, y_val