#!/usr/bin/env python

import argparse
import json
import os, sys
import numpy as np
import matplotlib.pyplot as plt

from functools import wraps
from random import sample
from collections import defaultdict
from sklearn.decomposition import PCA
from utils import get_array

from bokeh.plotting import figure
from bokeh.palettes import Dark2_5, Magma256
from bokeh.io import output_file, show, save
from bokeh.layouts import column, gridplot, row
from bokeh.models.widgets import Panel, Tabs

class Extractor:

    def __init__(self, dump):
        self.epochs = dump['training']
        self.train_end = dump['train_end']

    def get_metrics(self):
        keys = [k for k,v in self.epochs[0].items() if type(v) == float]
        d = defaultdict(list)

        for e in self.epochs:
            for k in keys:
                d[k].append(e[k])

        return d

    def get_variables(self, ignore_bias=True):
        d = defaultdict(list)
        layers = self.train_end['layers']

        for e in self.epochs:
            for l in layers:
                for k,v in e['weights'][l].items():
                    if not ignore_bias or 'bias' not in k:
                        d[k].append(v)
        return d

    def get_val_data(self):
        data = self.train_end['validation_data']
        labels = get_array(data['labels'])
        preds = get_array(data['predictions'])
        val_data = get_array(data['val_data'])
        return val_data, labels, preds

    @staticmethod
    def load_from_json(fname):
        return Extractor(json.load(fname))

class BokenApp:

    def __init__(self, inputs, output='output-nnvis'):
        self.inputs = []
        self.dumps = {}
        self.output = output

        for i in inputs:
            try:
                with open(i) as f: 
                    self.dumps[i] = Extractor.load_from_json(f)
                self.inputs.append(i)
            except:
                print(f'[WARNING] cannot open `{i}` skipping to next')
                raise

        if len(self.inputs) == 0:
            raise RuntimeError("at least one valid input required")
        elif len(self.inputs) > 5:
            print(f'[WARNING] too many arguments keeping only first 5')
            self.inputs = self.inputs[:5]
            self.dumps = {k: v for k,v in self.dumps if k in self.inputs}

        self.colors = {f: Dark2_5[i] for i,f in enumerate(self.inputs)}

    @property
    def empty(self):
        return figure()

    def plot_or_empty(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception:
                raise # TODO remove
                return self.empty
        return wrapper

    def render(self):
        output_file(self.output)
        tab1 = Panel(title='Metrics & Losses', child=self.metrics())
        tab2 = Panel(title='Histograms', child=self.histograms())
        tab3 = Panel(title='Mean Absolute Difference',
            child=self.mean_abs_diff())
        tab4 = Panel(title='Projection', child=self.projection())
        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
        show(tabs)
        # save(tabs)

    @staticmethod
    def normalize(a, max_val=1.0):
        a = a - a.min() * 1.
        return a * max_val / a.max()

    @plot_or_empty
    def projection(self):
        rows = []
        cmap = Magma256

        for i,fname in enumerate(self.inputs):
            val_data, labels, preds = self.dumps[fname].get_val_data()
            X = PCA(2).fit_transform(val_data.reshape(val_data.shape[0],-1))
            n = X.shape[0]

            if len(preds.shape) > 1 and preds.shape[1] > 1:
                # turn probabilities into labels
                preds = np.argmax(preds, axis=1)

            if n > 1000:
                ix = sample(range(n), 1000)
                X = X[ix]
                labels = labels[ix]
                preds = preds[ix]

            labels = BokenApp.normalize(labels, 255).astype(int)
            preds = BokenApp.normalize(preds, 255).astype(int)
            color = [cmap[i] for i in labels.squeeze()]

            f1 = figure(title=f'{fname} original')
            f1.circle(X[:,0], X[:,1], color=color)
            color = [cmap[i] for i in preds.squeeze()]
            f2 = figure(title=f'{fname} predictions')
            f2.circle(X[:,0], X[:,1], color=color)
            # rows.append(column(f1,f2))
            rows.append(f1)
            rows.append(f2)

        return gridplot(rows, ncols=2)

    @plot_or_empty
    def mean_abs_diff(self):
        # iterate over all inputs
        diffs = defaultdict(list)

        for i,fname in enumerate(self.inputs):
            data = self.dumps[fname].get_variables(ignore_bias=False)

            # iterate over all variables
            for k,v in data.items():
                diffs[k].append(([i['diff'] for i in v], fname))

        fs = []
        for k,v in diffs.items():
            fig = figure(title=k, x_axis_label='epochs',
                y_axis_label='mean absolute difference')
            for i in v:
                y, f = i
                fig.line(range(len(y)), y, color=self.colors[f], legend=f,
                    line_width=3, line_alpha=0.7)

            fs.append(fig)

        return gridplot(fs, ncols=2)

    @plot_or_empty
    def histograms(self):
        cmap = Magma256
        cols = []

        # iterate over all inputs
        for fname in self.inputs:
            data = self.dumps[fname].get_variables()
            hs = []
            # iterate over all variables
            for k,v in data.items():
                fig = figure(x_axis_label='weights', title=f'{fname}: {k}')
                x = get_array(v[0]['bin_edges'])[:-1]
                c = np.linspace(255,1,len(v)).astype(int)
                displacement = np.linspace(150, 0, len(v))

                # iterate over all epochs
                for j,i in enumerate(v):
                    y = get_array(i['hist'])
                    d = displacement[j]
                    assert (y + d >= 0).all()
                    fig.patch(x,y + d, fill_color=cmap[c[j]], line_width=0.5,
                        line_color=cmap[256-c[j]])

                hs.append(fig)

            cols.append(column(hs))
        return row(cols)

    @plot_or_empty
    def metrics(self):
        dd = {}
        for i,inp in enumerate(self.inputs):
            dd[inp] = self.dumps[inp].get_metrics()

        # just keep metrics from the last dump
        metrics = dd[inp].keys()

        fs = []
        for metric in metrics:
            i = 0
            f = figure(x_axis_label='epochs', y_axis_label=metric)
            for k,v in dd.items():
                if metric in v:
                    y = v[metric]
                    f.line(x=range(len(y)), y=y, legend=k, color=self.colors[k],
                        line_width=3, line_alpha=0.7)

            fs.append(f)

        return gridplot(fs, ncols=2)

def plot_outputs(train_end, outdir, nrow=6, ncol=6):
    outputs = train_end['image_data']['outputs']
    inputs = train_end['image_data']['input_data']

    path = os.path.join(outdir,'input_images')
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
        path = os.path.join(outdir, '{}-outputs'.format(k))
        os.makedirs(path, exist_ok=True)

        for img_id, ims in enumerate(imgs):
            seq = 0

            for i,a in enumerate(ims):
                if i % n == 0:
                    if i > 0:
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
                fname = os.path.join(path, f'outs-img{img_id}-{seq}.png')
                print(f'Creating {fname}')
                fig.savefig(fname)

def main():
    parser = argparse.ArgumentParser(description='Create images from json.')
    parser.add_argument('input', type=str, nargs='+', help='input file')
    parser.add_argument('-o','--output', type=str, help='output directory',
        metavar='FILE', default='output')
    parser.add_argument('-i','--img',action='store_true',
        help='output only outputs of convolutional layers')
    args = parser.parse_args()

    if args.img:
        if len(args.input) > 1:
            print('Producing output for the first argument only ...',
                file=sys.stderr)

        with open(args.input[0]) as f: 
            train_end = Extractor.load_from_json(f).train_end

            if 'image_data' in train_end:
                plot_outputs(train_end, args.output)
            else:
                print('[ERROR] No image data found', file=sys.stderr)
    else:
        output = args.output
        if not args.output.endswith('.html'):
            output += '.html'
        bokap = BokenApp(args.input, output)
        bokap.render()

if __name__ == '__main__':
    main()