#!/usr/bin/env python

import argparse
import os

from tensorflow import keras
from sklearn.model_selection import train_test_split

from prepare_datasets import load_sgemm
from collector import Collector

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-e', '--epochs', type=int, default=10, metavar='N',
        help='number of epochs (default 10)')
    parser.add_argument('-b', '--batch-size', type=int, default=128,
        metavar='N', help='batch size (default 128)')
    parser.add_argument('-o', '--out', type=str,
        help='output name, default sgemm', default='sgemm')
    parser.add_argument('-s','--summary', action='store_true',
        help='print the model summary and exit')
    parser.add_argument('-v','--verbose', type=int, default=1,
        help='control verbosity of training (default 1)')
    args = parser.parse_args()

    # load datasets
    X, y = load_sgemm()

    # scale
    X -= X.min(axis=0)
    X /= X.max(axis=0)

    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=.2)

    # build and train model
    model = keras.models.Sequential([
        keras.layers.Dense(64, activation='relu',
            input_shape=(x_train.shape[1],)),
        keras.layers.Dropout(.5),
        keras.layers.Dense(48, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])

    if args.summary:
        print(model.summary())
        keras.utils.plot_model(model, to_file='sgemm-model.png',
            show_shapes=True, show_layer_names=False)
        return
   
    logs_keys = ['mean_absolute_error', 'val_mean_absolute_error', 'val_loss']
    clt = Collector(logfile=args.out, logs_keys=logs_keys,
        validation_data=(x_val,y_val))
    callbacks = [clt]

    model.compile(loss='mse', metrics=['mae'], optimizer='adam')
    model.fit(x_train, y_train, validation_data=[x_val, y_val],
        batch_size=args.batch_size, epochs=args.epochs, callbacks=callbacks,
        verbose=args.verbose)

if __name__ == '__main__':
    main()