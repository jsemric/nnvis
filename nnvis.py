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
from pprint import pprint

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
    preds = get_array(data['predictions'])
    val_data = get_array(data['val_data'])

    if len(preds.shape) > 1 and preds.shape[1] > 1:
        # turn probabilities into labels
        preds = np.argmax(preds,axis=1)

    fig = plt.figure()
    d = 3 if dim3 else 2
    X = PCA(d).fit_transform(val_data.reshape(val_data.shape[0],-1))
    n = X.shape[0]

    # take only 1000 samples
    if n > 1000:
        ix = sample(range(n), 1000)
        X = X[ix]
        labels = labels[ix]
        preds = preds[ix]
    
    if dim3:
        ax1 = fig.add_subplot(121, projection='3d')
        ax1.scatter(X[:,0], X[:,1], X[:,2], c=labels.squeeze(), alpha=0.6)
        ax1.set_zlabel('pca 3')

        ax2 = fig.add_subplot(122, projection='3d')
        ax2.scatter(X[:,0], X[:,1], X[:,2], c=preds.squeeze(), alpha=0.6)
        ax2.set_zlabel('pca 3')
    else:
        ax1 = fig.add_subplot(121)
        ax1.scatter(X[:,0], X[:,1], c=labels.squeeze(), alpha=0.6)
        ax1.grid()

        ax2 = fig.add_subplot(122)
        ax2.scatter(X[:,0], X[:,1], c=preds.squeeze(), alpha=0.6)
        ax2.grid()

    ax1.set_title('original')
    ax1.set_xlabel('pca 1')
    ax1.set_ylabel('pca 2')
    ax2.set_title('model')
    ax2.set_xlabel('pca 1')
    ax2.set_ylabel('pca 2')
    # fig.suptitle(f'Projection {d}D')
    plt.tight_layout()
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

def compare_metrics(inputs, outdir):
    dd = {}
    for i in inputs:
        with open(i) as f:
            j = json.load(f)

        epochs = j['training']
        keys = [k for k,v in epochs[0].items() if type(v) == float]

        d = defaultdict(list)

        for e in epochs:
            for k in keys:
                d[k].append(e[k])

        dd[i] = d

    path = os.path.join(outdir, 'model_comparison')
    os.makedirs(path, exist_ok=True)

    for key in keys:
        i = 0
        for k,v in dd.items():
            if key in v:
                plt.plot(v[key], label=k)

        plt.title(key)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(path,f'{key}.png'))
        plt.clf()

def main():
    parser = argparse.ArgumentParser(description='Create images from json.')
    parser.add_argument('input', type=str, nargs='+', help='input file')
    parser.add_argument('-o','--output', type=str, help='output directory',
        metavar='FILE', default='out')
    parser.add_argument('-p','--p3d', action='store_true',
        help='use 3D projection instead of 2D')
    parser.add_argument('-c','--cmp', action='store_true',
        help='compare multiple models')
    args = parser.parse_args()


    os.makedirs(args.output, exist_ok=True)

    try:
        if len(args.input) > 1 and args.cmp:
            compare_metrics(args.input, args.output)
        else:
            if len(args.input) > 1:
                print('to compare multiple model run with --cmp option')
            dump = args.input[0]

            with open(dump) as f:
                j = json.load(f)

            plot_metrics(j['training'], args.output)
            plot_weights_distributions(j['training'], j['train_end']['layers'],
                args.output)

            if 'image_data' in j['train_end']:
                plot_filters(j['train_end'], args.output)

            plot_projection(j['train_end'], args.output, args.p3d)
            plot_weight_dynamics(j['training'], j['train_end']['layers'],
                args.output)

    except(Exception) as e:
        raise RuntimeError("corrupted input file") from e

if __name__ == '__main__':
    main()