#!/usr/bin/env python
# data aug
# autoencoder layers 

from tensorflow import keras

import argparse
import numpy as np
import os
from scipy.misc import imread
from skimage.transform import resize

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

SHAPE = (128,128)

def create_layers(inputs, n_classes):
    conv_args = dict(padding='same', activation='relu')
    x = keras.layers.Conv2D(96, (3,3), strides=(2,2), **conv_args)(inputs)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(32, (1,1), **conv_args)(x)
    x = keras.layers.Conv2D(128, (3,3), **conv_args)(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(32, (1,1), **conv_args)(x)
    x = keras.layers.Conv2D(256, (3,3), **conv_args)(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(32, (1,1), **conv_args)(x)
    x = keras.layers.Conv2D(316, (3,3), **conv_args)(x)
    x = keras.layers.MaxPool2D()(x)
    x = keras.layers.Conv2D(32, (1,1), padding='same')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    return keras.layers.Dense(n_classes, activation='softmax')(x)

def chunker(length, chunk_size):
    ''' Split a sequence into batches'''
    return (np.arange(pos, min(pos + chunk_size, length)) \
        for pos in range(0, length, chunk_size))

def load_images(paths):
    ''' Read and resize images.'''
    # X = np.array([resize(imread(p), SHAPE, preserve_range=True) for p in paths])
    return (resize(imread(p), SHAPE, preserve_range=True) / 255 for p in paths)
    # return X / 255

def image_loader(paths, y, batch_size=32, only_images=False):
    ''' Generate batches of images and corresponding labels'''
    gen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20, width_shift_range=0.12, height_shift_range=0.12,
        shear_range=20)
    while True:
        paths, y = shuffle(paths, y)
        for chunk in chunker(len(paths), batch_size):
            X = np.array([gen.random_transform(x) \
                for x in load_images(paths[chunk])])
            if only_images:
                yield (X, X)
            else:
                yield (X, y[chunk])

def prepare_data(classes=None):
    ''' Load paths and create labels'''
    codings = {}
    paths, labels = [], []
    i = 0

    for c in os.listdir('data'):
        if classes is not None:
            if c not in classes:
                continue
        print('class {}: {}'.format(i,c))
        codings[i] = c
        folder = os.path.join('data',c)
        fnames = [os.path.join(folder,f) for f in os.listdir(folder)]
        paths += fnames
        labels += [i] * len(fnames)
        i += 1

    return np.array(paths), np.array(labels), codings

def inspect_images(paths, labels, codings, n=5):
    paths, labels = shuffle(paths, labels)
    print(paths[:n])
    imgs = load_images(paths[:n])
    fig = plt.figure()
    for i,im in enumerate(imgs):
        ax = fig.add_subplot(1,n,i+1)
        ax.imshow(im)
        ax.set_title(labels[i])
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=3, metavar='N',
        help='number of epochs, default 3')
    parser.add_argument('-b', '--batch_size', type=int, default=64, metavar='N',
        help='batch size, default 64')
    parser.add_argument('-s','--summary', action='store_true',
        help='print the model layers')
    parser.add_argument('-n', '--name', type=str, help='collection name')
    parser.add_argument('-c', '--ckpt', type=str, help='checkpoint name',
        default='my_model')
    parser.add_argument('-a', '--auto', action='store_true')
    parser.add_argument('--inspect', type=int, metavar='N',
        help='inspect N images')
    args = parser.parse_args()

    ckpt_fname = '{}.ckpt'.format(args.ckpt)
    classes = ['tulip','sunflower','rose']
    n_classes = len(classes)
    paths, labels, codings = prepare_data(classes)
    if args.inspect is not None:
        inspect_images(paths, labels, codings, args.inspect)
        return

    inputs = keras.layers.Input((*SHAPE,3))
    outputs = create_layers(inputs, n_classes)
    model = keras.Model(inputs=inputs, outputs=outputs)

    if args.summary:
        print(model.summary())
        return

    X_train, X_valid, y_train, y_valid = train_test_split(paths, labels,
        test_size=0.15)
    # X_valid = load_images(X_valid)
    X_valid = np.array(list(load_images(X_valid)))
    batch_size = args.batch_size
    gen = image_loader(X_train, y_train, batch_size, only_images=args.auto)

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_acc', 
        mode='max', patience=10, verbose=1)
    # checkpoint = keras.callbacks.ModelCheckpoint(ckpt_fname, monitor=monitor,
    #     mode=mode, save_best_only=True, verbose=1)

    n_steps = 2*int(np.ceil(len(X_train) / batch_size))
    model.fit_generator(gen, steps_per_epoch=n_steps, epochs=args.epochs,
        validation_data=[X_valid, y_valid], callbacks=[early_stopping])

if __name__ == '__main__':
    main()