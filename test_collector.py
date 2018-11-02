#!/usr/bin/env python
# TODO
# projection + add predictions
# more images/histograms
# weight dynamics (new feature)

import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import base64

def get_filtered_image(train_end, layer, n_image, n_filter):
    outputs = train_end['image_data']['outputs'][layer]
    shape = outputs['shape']
    data = outputs['data']
    a = np.frombuffer(base64.b64decode(data), np.float32)
    a.shape = shape # fancy reshape
    return a[n_image,:,:,n_filter]

def show_hist(epochs, n_epoch, layer, varname):
    w = epochs[n_epoch]['weights'][layer][varname]
    hist = w['hist']
    bins = w['bin_edges']
    h_data, h_shape = hist['data'], hist['shape']
    hist = np.frombuffer(base64.b64decode(h_data), np.int64)
    b_data, b_shape = bins['data'], bins['shape']
    bins = np.frombuffer(base64.b64decode(b_data), np.float32)
    return hist, bins

def main():
    with open('cifar4.json') as f:
        j = json.load(f)

    epochs = j['training']
    train_end = j['train_end']
    hist, bins = show_hist(epochs, n_epoch=0, layer='dense',
        varname='dense/kernel:0')
    plt.fill_between(range(len(hist)),hist)
    # plt.set_xticks(bins)
    plt.show()

    img = get_filtered_image(train_end, layer='conv2d', n_image=2,
        n_filter=0)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()