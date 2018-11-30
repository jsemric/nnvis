#!/usr/bin/env python
# TODO decorator
#   try: ... except: empty
#   images - just one model
#   refactor - extract general terms
#   histograms - scale (fix collector)

import argparse
import json
import os
import numpy as np
import re

from random import sample
from collections import defaultdict
from sklearn.decomposition import PCA
from utils import get_array

from bokeh.plotting import figure
from bokeh.palettes import Dark2_5, Magma256
from bokeh.io import output_file, show, save
from bokeh.layouts import column, gridplot, row
from bokeh.layouts import layout
from bokeh.models.widgets import Panel, Tabs, Dropdown

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
                    if ignore_bias and 'bias' not in k:
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
                    self.dumps.append(Extractor.load_from_json(f))
                self.inputs.append(i)
            except:
                print(f'[WARNING] cannot open `{i}` skipping to next')

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

    def render(self):
        output_file(self.output)
        tab1 = Panel(title='Metrics & Losses', child=self.metrics())
        tab2 = Panel(title='Histograms', child=self.histograms())
        tab3 = Panel(title='Mean Absolute Difference',
            child=self.mean_abs_diff())
        tab4 = Panel(title='Projection', child=self.projection())
        tabs = Tabs(tabs=[tab1, tab2, tab3, tab4])
        # show(tabs)
        save(tabs)

    @staticmethod
    def normalize(a, max_val=1):
        a = a - a.min() * 1.
        return a * max_val / a.max()

    def projection(self):
        rows = []
        cmap = Magma256

        for i,fname in enumerate(self.inputs):
            val_data, labels, preds = self.dumps[fname].get_val_data()
            X = PCA(2).fit_transform(val_data.reshape(val_data.shape[0],-1))
            n = X.shape[0]

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

    def mean_abs_diff(self):
        # iterate over all inputs
        diffs = defaultdict(list)

        for i,fname in enumerate(self.inputs):
            data = self.dumps[fname].get_variables()

            # iterate over all variables
            for k,v in data.items():
                diffs[k].append(([i['diff'] for i in v], fname))

        fs = []
        for k,v in diffs.items():
            fig = figure(title=k, x_axis_label='epochs',
                y_axis_label='mean absolute difference')
            for i in v:
                y, f = i
                fig.line(range(len(y)), y, color=self.colors[f], legend=f)

            fs.append(fig)

        return gridplot(fs, ncols=2)

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
                    f.line(x=range(len(y)), y=y, legend=k, color=self.colors[k])

            fs.append(f)

        return gridplot(fs, ncols=2)

def main():
    parser = argparse.ArgumentParser(description='Create images from json.')
    parser.add_argument('input', type=str, nargs='+', help='input file')
    parser.add_argument('-o','--output', type=str, help='output directory',
        metavar='FILE', default='output.html')
    args = parser.parse_args()

    bokap = BokenApp(args.input, args.output)
    bokap.render()

if __name__ == '__main__':
    main()