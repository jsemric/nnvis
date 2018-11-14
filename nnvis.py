#!/usr/bin/env python
#
# outdir
#        \- metrics
#        \- histograms
#        \- filters
#                   \- images
#                   \- layer0 - {img}_{id}
#                   \- layer1 - {img}_{id}
#                   \- ...
#        \- projection
#        \- weights dynamics

import argparse
import json
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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

    for k in keys:
        plt.plot(d[k])
        plt.title(k)
        plt.xlabel('epochs')
        plt.ylabel(k)
        plt.grid()
        plt.show()

def plot_weights_distributions(epochs, layers, outdir):
    d = defaultdict(list)

    for e in epochs:
        for l in layers:
            for k,v in e['weights'][l].items():
                d[k].append(v)

    for k,v in d.items():
        fig = plt.figure()
        x = get_array(v[0]['bin_edges'])[:-1]
        for j,i in enumerate(v):
            y = get_array(i['hist'])
            plt.fill_between(x, y/y.max() + 1.1*j, 1.1*j, color='C0', alpha=.7)

        plt.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False) 
        plt.ylabel('epochs')
        plt.xlabel('weights values')
        plt.title(k)
        plt.show()

def plot_filters(train_end, outdir, nrow=6, ncol=6):
    outputs = train_end['image_data']['outputs']
    inputs = train_end['image_data']['input_data']

    path = os.path.join(outdir, 'images')
    os.makedirs(path, exist_ok=True)
    for i,a in enumerate(get_array(inputs)):
        plt.imshow(a)
        plt.axis('off')
        plt.savefig(os.path.join(path,'img{}.png'.format(i)))

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

def main():
    parser = argparse.ArgumentParser(description='Create images from json.')
    parser.add_argument('input', type=str, help='input file')
    parser.add_argument('-o','--output', type=str, help='output directory',
        metavar='FILE', default='out')
    args = parser.parse_args()

    with open(args.input) as f:
        j = json.load(f)

    os.makedirs(args.output, exist_ok=True)

    # plot_metrics(j['training'], args.output)
    # plot_weights_distributions(j['training'], j['train_end']['layers'],
    #     args.output)
    plot_filters(j['train_end'], args.output)


if __name__ == '__main__':
    main()