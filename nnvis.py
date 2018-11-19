#!/usr/bin/env python
#
# outdir
#        \- learning_curve
#        \- histograms
#        \- filters
#                   \- images
#                   \- layer0 - {img}_{id}
#                   \- layer1 - {img}_{id}
#                   \- ...
#        \- projection
#        \- mean_abs_diff

import argparse
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import re

from matplotlib import cm
from random import sample
from collections import defaultdict
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from utils import get_array

def plot_metrics(epochs, outdir):
    keys = [k for k,v in epochs[0].items() if type(v) == float]
    d = defaultdict(list)

    for e in epochs:
        for k in keys:
            d[k].append(e[k])

    path = os.path.join(outdir, 'learning_curve')
    os.makedirs(path, exist_ok=True)

    for k in keys:
        plt.plot(d[k])
        plt.title(k)
        plt.xlabel('epochs')
        plt.ylabel(k)
        plt.grid()
        plt.savefig(os.path.join(path,f'{k}.png'))
        plt.clf()

def plot_weights_distributions(epochs, layers, outdir):
    d = defaultdict(list)

    for e in epochs:
        for l in layers:
            for k,v in e['weights'][l].items():
                if 'bias' not in k:
                    d[k].append(v)

    path = os.path.join(outdir, 'histograms')
    os.makedirs(path, exist_ok=True)

    # cmap = cm.get_cmap('Greens')
    cmap = cm.get_cmap('PRGn')
    
    for k,v in d.items():
        fig = plt.figure()
        x = get_array(v[0]['bin_edges'])[:-1]
        c = np.linspace(1,0.75,len(v))
        displacement = np.linspace(150,0,len(v))
        max_ = get_array(v[0]['hist']).max()

        for j,i in enumerate(v):
            y = get_array(i['hist'])
            y = y / max_ * 60
            d = displacement[j]
            plt.fill_between(x, y + d, d, facecolor=cmap(c[j]), 
                linewidth=0.2, edgecolor='k')

        # https://stackoverflow.com/questions/14908576/how-to-remove-frame-from-
        # matplotlib-pyplot-figure-vs-matplotlib-figure-frame
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

        plt.axes().get_yaxis().set_ticks([])
        plt.ylabel('epochs')
        plt.xlabel('weights values')
        plt.title(k)
        fname = re.sub('[/:]', '_',k)
        plt.savefig(os.path.join(path, f'hist-{fname}.png'))
        # plt.show()

def plot_filters(train_end, outdir, nrow=6, ncol=6):
    outputs = train_end['image_data']['outputs']
    inputs = train_end['image_data']['input_data']

    path = os.path.join(outdir, 'images')
    os.makedirs(path, exist_ok=True)
    for i,a in enumerate(get_array(inputs)):
        plt.imshow(a)
        plt.axis('off')
        plt.savefig(os.path.join(path,'img{}.png'.format(i)))

    outdir = os.path.join(outdir, 'outputs')
    fig, axes = plt.subplots(nrow, ncol, figsize=(10,10))

    for k,v in outputs.items():
        imgs = get_array(v)
        imgs = np.einsum('ijkl->iljk', imgs)

        n = nrow * ncol
        path = os.path.join(outdir, '{}-outs'.format(k))
        os.makedirs(path, exist_ok=True)

        for img_id, ims in enumerate(imgs):
            seq = 0

            for i,a in enumerate(ims):
                if i % n == 0:
                    if i > 0:
                        # print(os.path.join(path,
                        #     'outs-img{}-{}.png'.format(img_id, seq)))
                        fig.savefig(os.path.join(path,
                            'outs-img{}-{}.png'.format(img_id, seq)))
                        seq += 1

                    for ax in axes.flatten():
                        ax.clear()
                        ax.axis('off')

                ax = axes[(i % n) // ncol][(i % n) % ncol]
                ax.imshow(a)
                ax.axis('off')

            if i % n:
                # print(os.path.join(path,
                #             'outs-img{}-{}.png'.format(img_id, seq)))
                fig.savefig(os.path.join(path, 
                    'outs-img{}-{}.png'.format(img_id, seq)))

def plot_projection(train_end, outdir, dim3=False):
    data = train_end['validation_data']
    labels = get_array(data['labels'])
    predictions = get_array(data['predictions'])
    val_data = get_array(data['val_data'])

    fig = plt.figure()
    n = 3 if dim3 else 2
    X = PCA(n).fit_transform(val_data.reshape(val_data.shape[0],-1))
    n = X.shape[0]

    # take only 1000 samples
    if n > 1000:
        ix = sample(range(n), 1000)
        X = X[ix]
        labels = labels[ix]
    
    if dim3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:,0], X[:,1], X[:,2], c=labels.squeeze(), alpha=0.6)
        ax.set_zlabel('pca 3')
    else:
        ax = fig.add_subplot(111)
        ax.scatter(X[:,0], X[:,1], c=labels.squeeze(), alpha=0.6)
        ax.grid()

    ax.set_xlabel('pca 1')
    ax.set_ylabel('pca 2')
    
    ax.set_title(f'Projection {n}D')
    path = os.path.join(outdir,'projection')
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, 'projection.png'))

def plot_weight_dynamics(epochs, layers, outdir):
    d = defaultdict(list)

    for e in epochs:
        for l in layers:
            for k,v in e['weights'][l].items():
                d[k].append(v)

    path = os.path.join(outdir, 'mean_abs_diff')
    os.makedirs(path, exist_ok=True)

    for k,v in d.items():
        fig = plt.figure()
        x = []
        for i in v:
            x.append(i['diff'])

        plt.fill_between(range(len(x)), x)

        plt.xlabel('epochs')
        plt.ylabel('mean absolute difference')
        plt.title(k)
        fname = re.sub('[/:]', '_',k)
        plt.savefig(os.path.join(path, f'mad-{fname}.png'))

def main():
    parser = argparse.ArgumentParser(description='Create images from json.')
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('-o','--output', type=str, help='output directory',
        metavar='FILE', default='out')
    parser.add_argument('-p','--p3d', action='store_true',
        help='use 3D projection instead of 2D')
    args = parser.parse_args()

    with open(args.input) as f:
        j = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    # plot_metrics(j['training'], args.output)
    plot_weights_distributions(j['training'], j['train_end']['layers'],
        args.output)

    if 'image_data' in j['train_end']:
        plot_filters(j['train_end'], args.output)

    # plot_projection(j['train_end'], args.output, args.p3d)
    # plot_weight_dynamics(j['training'], j['train_end']['layers'], args.output)

if __name__ == '__main__':
    main()