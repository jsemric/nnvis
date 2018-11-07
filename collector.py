# TODO
# weights dynamic
# regression or classification

import json
import base64
import numpy as np
import warnings
import tensorflow as tf

from tensorflow import keras
from abc import ABC

try:
    from pymongo import MongoClient
except ImportError:
    MongoClient = None

try:
    from tensorflow.keras import backend as K
except ImportError:
    K = keras.backend

class CollectorError(Exception):
    '''Collector exception class.'''
    def __init__(self, string):
        super(CollectorError, self).__init__(string)

class CollectorBase(keras.callbacks.Callback, ABC):
    '''
    WARNING do not use this class directly, it is an abstract class.

    Description
    -----------
    Collect model data during training.

    Data collected after or before training:
        1.) image data
        2.) a sequence of network layers
        3.) validation data and predictions

    The following figures are collected during training after each epoch:
        1.) weights distributions (binning)
        2.) outputs of each layer for the input data
        3.) loss and metrics for training and validation data
    
    Attributes
    ----------
    log_keys: dict
        metrics to save, e.g., val_roc

    nbins: int
        number of bins when computing histogram of weights/outputs

    serialize_array: callback
        function for serializing arrays

    image_data: ndarray at least 2D or list of ndarrays
        data used for the layers output visualization

    val_data: tuple (ndarray, labels)
        data used for prediction projection/residuals visualization
    '''

    def __init__(self, logs_keys=None, nbins=100, image_data=None,
        validation_data=None, array_encoding='base64', labels=None):
        super(CollectorBase,self).__init__()
        self.logs_keys = {'loss'}

        if logs_keys is not None:
            type_ = type(logs_keys)

            if type_ == list:
                self.logs_keys |= set(logs_keys)
            elif type_ == str:
                logs_keys.add(logs_keys)
            else:
                raise ValueError('log_keys is of type {} but str or list '
                    ' expected'.format(type_))
            
        if type(nbins) != int:
            raise ValueError('nbins is of type {} but int expected'.
                format(type(nbins)))

        if nbins < 0:
            raise ValueError('{} invalid value for nbins'.format(nbins))

        self.nbins = nbins
        
        if array_encoding == 'X':
            self.serialize_array = lambda _: 'X'
        elif array_encoding == 'base64':
            self.serialize_array = lambda w: {
                'shape': w.shape,
                'data': base64.b64encode(
                    np.ascontiguousarray(w, dtype=np.float32)).decode('ascii')
            }
        elif array_encoding == 'list':
            # not memory efficient, avoid
            self.serialize_array = lambda w: w.tolist()
        else:
            raise ValueError('Invalid value for the parameter array_encoding. '
                'Expected one of the following {} but found `{}`'.
                format(['X','list','base64'], array_encoding))

        self.image_data = image_data
        self.validation_data = validation_data
        
        # TODO check validation data
        if image_data is not None:
            pass

        self._functor = None
        self.ranges = {}
        self.weights = {}

    def _get_input_data_and_layers(self):
        ret = {'layers': [l.name for l in self.model.layers]}

        if self.image_data is not None:
            ret['image_data'] = {
                'input_data': self.serialize_array(self.image_data),
                'outputs': {
                    l.name: self.serialize_array(o) \
                    for l,o in self._get_outputs(self.image_data)}
            }

        if self.validation_data is not None:
            X = self.validation_data[0]
            y = self.validation_data[1]

            ret['validation_data'] = {
                'val_data': self.serialize_array(X),
                'labels': self.serialize_array(y),
                'predictions': self.serialize_array(self.model.predict(X))
            }

        return ret

    def on_train_begin(self,logs=None):
        # https://stackoverflow.com/questions/41711190/keras-how-to-get-the-
        # output-of-each-layer
        inp = self.model.input
        outputs = [l.output for l in self.model.layers]
        self._functor = K.function([inp], outputs)
        ls = self.model.layers

        for l in ls:
            ws = l.get_weights()
            vs = l.variables

            self.weights[l.name] = {}
            for v,w in zip(vs,ws):
                vn = v.name
                self.weights[l.name][vn] = w

    def _get_outputs(self, input_data):
        outs = self._functor([input_data])
        ls = self.model.layers
        return zip(ls,outs)

    def _handle_weights(self):
        '''Make histograms from weights.'''
        ls = self.model.layers
        ret = {}

        for l in ls:
            ws = l.get_weights()
            vs = l.variables
            assert len(ws) == len(vs)
            ret[l.name] = {}

            for v,w in zip(vs,ws):
                vn = v.name

                if vn not in self.ranges:
                    std = w.std()
                    self.ranges[vn] = (w.min() - std, w.max() + std)

                
                # compute absolute differences
                diff = np.mean(np.abs(w - self.weights[l.name][vn]))
                # update weights
                self.weights[l.name][vn] = w
                # get histograms
                tmp = self._bin_array(w, self.ranges[vn])
                # recast because float32 is not serializable
                tmp['diff'] = float(diff) 
                ret[l.name][vn] = tmp

                # ret[l.name][vn] = self.serialize_array(w)

        return ret

    def _bin_array(self, a, rng=None):
        '''Binning an array and returning it as a dictionary.'''
        hist, bin_edges = np.histogram(a, self.nbins, range=rng)
        ret = {
            'hist': self.serialize_array(hist),
            'bin_edges': self.serialize_array(bin_edges)
        }

        return ret

    def on_epoch_end(self, epoch, logs=None):
        '''Dump variables, outputs, loss and metrics.'''
        dump = {'epoch': epoch}
        logs_keys = set(logs.keys())

        if not logs_keys >= self.logs_keys:
            warnings.warn('removing invalid logging keys', RuntimeWarning)
            self.logs_keys &= logs_keys

        for k in self.logs_keys:
            dump[k] = logs[k]

        dump['weights'] = self._handle_weights()

        # if self.validation_data is not None:
        #     dump['outputs'] = {l.name: self._bin_array(o) \
        #         for l, o in self._get_outputs(self.validation_data[0])}

        self._write_epoch(dump, epoch)

    def _write_epoch(self, dump, epoch):
        pass

    def on_train_end(self, logs=None):
        pass

class Collector(CollectorBase):
    '''Collect model data during training and store the result to a JSON file.
    Attributes
    ----------
    logfile: str
        filename where the result will be stored

    file: fd
        file descriptor
    '''
    def __init__(self, logfile, add_ext=True, logs_keys=None, nbins=50,
        image_data=None, validation_data=None, array_encoding='base64'):

        super(Collector,self).__init__(logs_keys=logs_keys, nbins=nbins,
            image_data=image_data, array_encoding=array_encoding,
            validation_data=validation_data)

        self.logfile = logfile

        if add_ext and not self.logfile.endswith('.json'):
            self.logfile += '.json'

        self.file = None

    def _write_epoch(self, dump, epoch):
        if epoch > 0:
            self.file.write(',')

        json.dump(dump, self.file)

    def on_train_begin(self, logs=None):
        super(self.__class__,self).on_train_begin(logs)
        '''Open file.'''
        try:
            self.file = open(self.logfile,'w')
        except Exception as e:
            msg = "cannot open file {}".format(self.logfile)
            raise CollectorError(msg) from e

        # begin training collection
        self.file.write('{"training": [')

    def on_train_end(self, logs=None):
        '''Dump layers close file.'''
        # enclose training
        self.file.write(']')
        # store predictions and validation data
        dump = self._get_input_data_and_layers()

        if dump is not None:
            self.file.write(',"train_end":')
            json_str = json.dumps(dump)
            self.file.write(json_str)

        # end of the collection and JSON file
        self.file.write('}')
        self.file.close()

class MongoCollector(Collector):
    '''Collect model data during training and store the result to a MongoDB
    database.

    Attributes
    ----------
    db_name: str
        database name

    collection_name: str
        collection name in the Mongo database

    client: MongoClient
        mongo client

    collection: object
        mongo collection
    
    collection structure
    ------------
    "input_data" : [data ...]
    "layers" : [layers ...],
    "training" : [epochs ...]
    "validation" : {...}
    '''
    def __init__(self, collection_name, db_name='nnviz_db', logs_keys=None,
        nbins=50, image_data=None, validation_data=None,
        array_encoding='base64'):

        if MongoClient is None:
            raise ImportError('In order to use, install pymongo')

        super(MongoCollector,self).__init__(logs_keys=logs_keys, nbins=nbins,
            image_data=image_data, array_encoding=array_encoding,
            validation_data=validation_data)
        
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def on_train_begin(self, logs=None):
        super(self.__class__,self).on_train_begin(logs)
        self.client = MongoClient()
        db = self.client[self.db_name]
        self.collection = db[self.collection_name]

    def on_train_end(self, logs=None):
        dump = self._train_begin()
        self.collection.insert(dump)
        dump = self._get_input_data_and_layers()

        if dump is not None:
            self.collection.insert(dump)

        self.client.close()

    def _write_epoch(self, dump, epoch):
        self.collection.insert(dump)