#!/usr/bin/env python

import argparse
import numpy as np
import tensorflow as tf
import os
import shutil

from tensorflow import keras
import matplotlib.pyplot as plt

from prepare_datasets import load_cifar4
from collector import Collector

SHAPE = (32,32,3)

def inspect_data(x_train, y_train, x_val, y_val, n_images=3):
    print('Train shape {} test shape {}'.format(x_train.shape, x_val.shape))
    fig, axes = plt.subplots(1, n_images)

    for i, n in enumerate(np.random.randint(0, x_train.shape[0], n_images)):
        axes[i].imshow(x_train[n])
        axes[i].set_title('label {}'.format(y_train[n]))

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
        help='number of epochs (default 10)')
    parser.add_argument('-s', '--steps', type=int, default=64, metavar='N',
        help='steps per epoch (default 64)')
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
        help='batch size (default 64)')
    parser.add_argument('-x', '--xenc', action='store_true',
        help='put X as a placeholder for arrays')
    parser.add_argument('-o', '--out', type=str,
        help='output name, default cifar4', default='cifar4')
    parser.add_argument('-t','--tb', action='store_true',
        help='collect TensorBoard data')
    parser.add_argument('--viz', type=int, default=10, metavar='N',
        help='number of validation instances used for visualization '
        '(default 10)')
    parser.add_argument('--sgd', action='store_true',
        help='use SGD optimizer instead of Adam')
    parser.add_argument('-v','--verbose', type=int, default=1,
        help='control verbosity of training (default 1)')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--summary', action='store_true',
        help='print the model summary and exit')
    group.add_argument('--inspect', type=int, metavar='N',
        help='inspect N random images and exit')

    args = parser.parse_args()

    n_epochs = args.epochs
    n_steps = args.steps
    batch_size = args.batch_size

    x_train, y_train, x_val, y_val = load_cifar4()
    # x_val, y_val = x_val[:100], y_val[:100]

    print('training instances:  ', x_train.shape[0])
    print('validation instances:', x_val.shape[0])

    # relabel
    y_train[y_train == 9] = 2
    y_train[y_train == 6] = 0
    y_val[y_val == 9] = 2
    y_val[y_val == 6] = 0

    # print(y_val.dtype)
    if args.inspect is not None:
        inspect_data(x_train, y_train, x_val, y_val, args.inspect)
        return

    # normalize data
    x_train = x_train / 255.
    x_val = x_val / 255.

    model = keras.models.Sequential([
        keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu',
            padding='same', input_shape=SHAPE),
        keras.layers.MaxPool2D(),
        keras.layers.Dropout(0.5),
        keras.layers.Conv2D(96, kernel_size=(3,3), activation='relu',
            padding='same'),
        keras.layers.MaxPool2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(4, activation='softmax')
    ])

    if args.summary:
        print(model.summary())
        keras.utils.plot_model(model, to_file='cifar4-model.png',
            show_shapes=True, show_layer_names=False)
        return

    if args.viz > x_val.shape[0]:
        raise ValueError('Value of --viz option is too high')

    logs_keys = ['acc','val_acc','val_loss']
    enc = 'X' if args.xenc else 'base64'

    # callback collecting the data during training
    clt = Collector(logfile=args.out, logs_keys=logs_keys,
        image_data=x_val[:args.viz], array_encoding=enc,
        validation_data=(x_val,y_val))
    tb = keras.callbacks.TensorBoard(log_dir='tbGraph', histogram_freq=1,  
        write_graph=True, write_images=True)

    callbacks = [clt]
    
    if args.tb:
        # remove dir
        if os.path.exists('tbGraph'):
            shutil.rmtree('tbGraph')
        callbacks = [tb]

    # image transformations
    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30, width_shift_range=0.2,
        height_shift_range=0.2, horizontal_flip=True)
    gen = datagen.flow(x_train, y_train, batch_size=batch_size)
    
    opt = 'adam'
    if args.sgd:
        opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)

    model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'],
        optimizer=opt)
    model.fit_generator(gen, steps_per_epoch=n_steps, epochs=n_epochs,
        validation_data=[x_val, y_val], callbacks=callbacks,
        verbose=args.verbose)

if __name__ == '__main__':
    main()
