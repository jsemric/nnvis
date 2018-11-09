#!/usr/bin/env python

import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64

from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

def get_array(d):
    shape = d['shape']
    data = d['data']
    a = np.frombuffer(base64.b64decode(data), np.float32)
    return a.reshape(shape)

def plot_val_data(train_end):
    data = train_end['validation_data']
    labels = get_array(data['labels'])
    predictions = get_array(data['predictions'])
    val_data = get_array(data['val_data'])
    print(val_data.shape, labels.shape, predictions.shape)

    fig = plt.figure()
    X = PCA(3).fit_transform(val_data.reshape(val_data.shape[0],-1))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], c=labels.squeeze(), alpha=0.6)
    ax.set_xlabel('pca 1')
    ax.set_ylabel('pca 2')
    ax.set_zlabel('pca 3')
    ax.set_title('Projection 3D')
    fig.savefig(os.path.join('figures','projection3d.png'))
    # plt.show()

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

    fig.savefig(os.path.join('figures',f'{layer}_outputs_{n_image}.png'))

def get_hist(epochs, n_epoch, layer, varname):
    w = epochs[n_epoch]['weights'][layer][varname]
    hist = w['hist']
    bins = w['bin_edges']
    h_data, h_shape = hist['data'], hist['shape']
    hist = np.frombuffer(base64.b64decode(h_data), np.float32)
    b_data, b_shape = bins['data'], bins['shape']
    bins = np.frombuffer(base64.b64decode(b_data), np.float32)

    return hist, bins

def plot_hist(epochs, layer='dense', vname='dense/kernel:0'):
    plt.figure()
    for e in range(0,7):
        h, b = get_hist(epochs, n_epoch=e, layer=layer, varname=vname)
        # print(np.sum(h))
        plt.plot(b[:-1], h/np.sum(h), alpha=0.8)
        # plt.fill_between(b[:-1], h, alpha=0.5)

    plt.title(f'dense weights') 
    plt.savefig(os.path.join('figures','hist.png'))

def plot_weights(epochs, layer='dense', vname='dense/kernel:0', nrow=5,
    ncol=5):
    l = []
    for e in range(0, nrow * ncol + 1):
        w = epochs[e]['weights'][layer][vname]
        data = w['data']
        data = np.frombuffer(base64.b64decode(data), np.float32)
        data.shape = w['shape']
        l.append(data)

    a = [np.mean(np.abs(l[i + 1] - l[i])) for i in range(len(l)-1)]
    plt.fill_between(range(len(a)),a)
    plt.show()

    fig, axes = plt.subplots(nrow, ncol)
    fig.suptitle(f'{layer} outputs')
    for i in range(nrow * ncol):
        print(np.mean(np.abs(l[i + 1] - l[i])))
        a = np.abs(l[i + 1] - l[i])
        ax = axes[i // ncol][i % ncol]
        ax.axis('off')
        ax.imshow(a)

    plt.show()

def plot_ldiff(epochs, layer='dense', vname='dense/kernel:0'):
    diffs = []
    for e in epochs:
        diffs.append(e['weights'][layer][vname]['diff'])

    plt.figure()
    plt.fill_between(range(len(diffs)),diffs)
    vname = vname.replace('/','_')
    plt.savefig(os.path.join('figures',f'{layer}_diff.png'))    

def main():
    # with open('full_weights.json') as f:
    # with open('sgemm.json') as f:
    with open('cifar4.json') as f:
        j = json.load(f)

    epochs = j['training']
    train_end = j['train_end']
    os.makedirs('figures', exist_ok=True)

    plot_val_data(train_end)
    plot_ldiff(epochs)
    plot_hist(epochs)
    get_filtered_image(train_end, layer='conv2d', n_image=2)
    get_filtered_image(train_end, layer='conv2d_1', n_image=2)
    get_image_data(train_end)

if __name__ == '__main__':
    main()