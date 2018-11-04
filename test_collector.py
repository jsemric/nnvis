#!/usr/bin/env python
# TODO
# projection + add predictions
# more images/histograms
# weight dynamics (new feature)

import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64

def get_image_data(train_end, nrow=2, ncol=2):
    input_data = train_end['image_data']['input_data']
    shape = input_data['shape']
    data = input_data['data']
    a = np.frombuffer(base64.b64decode(data), np.float32)
    a.shape = shape
    fig, axes = plt.subplots(nrow,ncol)
    fig.suptitle('input images')
    for i in range(nrow*ncol):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(a[i])
        ax.axis('off')

    os.makedirs('figures', exist_ok=True)
    fig.savefig(os.path.join('figures','input_images.png'))

def get_filtered_image(train_end, layer, n_image, nrow=5, ncol=5):
    outputs = train_end['image_data']['outputs'][layer]
    shape = outputs['shape']
    data = outputs['data']
    a = np.frombuffer(base64.b64decode(data), np.float32)
    a.shape = shape # fancy reshape
    first_filter =  a[n_image,:,:,0]

    fig, axes = plt.subplots(nrow,ncol)
    fig.suptitle(f'{layer} outputs')
    for i in range(nrow*ncol):
        ax = axes[i // ncol][i % ncol]
        ax.imshow(a[n_image,:,:,i])
        ax.axis('off')

    os.makedirs('figures', exist_ok=True)
    fig.savefig(os.path.join('figures',f'{layer}_outputs_{n_image}.png'))    

def show_hist(epochs, n_epoch, layer, varname):
    w = epochs[n_epoch]['weights'][layer][varname]
    hist = w['hist']
    bins = w['bin_edges']
    h_data, h_shape = hist['data'], hist['shape']
    hist = np.frombuffer(base64.b64decode(h_data), np.float32)
    b_data, b_shape = bins['data'], bins['shape']
    bins = np.frombuffer(base64.b64decode(b_data), np.float32)
    plt.title(f'{layer} weights')
    plt.fill_between(bins[:-1], hist)
    os.makedirs('figures', exist_ok=True)
    plt.savefig(os.path.join('figures','hist.png'))

def main():
    with open('cifar4.json') as f:
        j = json.load(f)

    epochs = j['training']
    train_end = j['train_end']
    show_hist(epochs, n_epoch=0, layer='dense', varname='dense/kernel:0')
    get_filtered_image(train_end, layer='conv2d', n_image=2)
    get_filtered_image(train_end, layer='conv2d_1', n_image=2)
    get_image_data(train_end)

if __name__ == '__main__':
    main()