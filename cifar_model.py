#!/usr/bin/env python
# TODO
#   subclass Model
#   use tf.Data

import argparse
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

from prepare_datasets import get_cifar10_4
from collector import Collector

SHAPE = (32,32,3)

def inspect_data(x_train, y_train, x_val, y_val, n_images=3):
    print('Train shape {} test shape {}'.format(x_train.shape, x_val.shape))
    fig, axes = plt.subplots(1,n_images)
    for i, n in enumerate(np.random.randint(0, x_train.shape[0], n_images)):
        axes[i].imshow(x_train[n])
        axes[i].set_title('label {}'.format(y_train[n]))
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
        help='number of epochs, default 10')
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
        help='batch size, default 64')
    parser.add_argument('-o', '--out', type=str, help='output name',
        default='cifar4')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s','--summary', action='store_true',
        help='print the model summary and exit')
    group.add_argument('--inspect', type=int, metavar='N',
        help='inspect N random images and exit')
    args = parser.parse_args()

    n_epochs = args.epochs
    batch_size = args.batch_size
    n_show = 10

    x_train, y_train, x_val, y_val = get_cifar10_4()
    # relabel
    y_train[y_train == 9] = 2
    y_train[y_train == 6] = 1
    y_val[y_val == 9] = 2
    y_val[y_val == 6] = 1

    if args.inspect is not None:
        inspect_data(x_train, y_train, x_val, y_val, args.inspect)
        return

    x_train = x_train / 255.
    x_val = x_val / 255.

    model = keras.models.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',
            input_shape=SHAPE),
        keras.layers.MaxPool2D(),
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation='softmax')
    ])

    if args.summary:
        print(model.summary())
        return
   
    logs_keys = ['acc','val_acc','val_loss']
    clt = Collector(logfile=args.out, logs_keys=logs_keys,
        input_data=x_val[:5])

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
        optimizer='adam')
    collector_data = x_val[:n_show], y_val[:n_show]
    model.fit(x_train, y_train, validation_data=[x_val, y_val],
        batch_size=batch_size, epochs=n_epochs, callbacks=[clt])
    model.save('cifar4.h5')

if __name__ == '__main__':
    main()