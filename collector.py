# TODO
# val data - store projection
# confusion matrix
# labels
# comments
# type checking & error handling

import json
import base64
import numpy as np
import warnings
from pymongo import MongoClient
from tensorflow import keras
from tensorflow.keras import backend as K
from abc import ABC

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

    Data collected before training:
        1.) image data
        2.) a sequence of network layers

    The following figures are collected during training after each epoch:
        1.) weights distributions (binning)
        2.) outputs of each layer for image data
        3.) loss and metrics for training and validation data
    
    Attributes
    ----------
    log_keys: dict
        metrics to save, e.g., val_roc

    nbins: int
        number of bins when computing histogram from weights

    serialize_array: callback
        function for serializing arrays

    input_data: ndarray at least 2D or list of ndarrays
        data used for the layers output visualization

    val_data: tuple (ndarray, labels)
        data used for prediction projection/residuals visualization
    '''

    def __init__(self, logs_keys=None, nbins=50, input_data=None, val_data=None,
        array_srl='base64'):
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
        
        if array_srl == 'X':
            self.serialize_array = lambda _: 'X'
        elif array_srl == 'base64':
            self.serialize_array = lambda w: {'shape': w.shape,
                'data': base64.b64encode(w).decode('ascii')}
        elif array_srl == 'list':
            self.serialize_array = lambda w: w.tolist()
        else:
            raise ValueError('Invalid value for the parameter array_srl. '
                'Expected one of the following {} but found `{}`'.
                format(['X','list','base64'], array_srl))

        self.input_data = None
        if input_data is not None:
            self.input_data = input_data

        self._functor = None

    def _train_begin(self):
        ret = {'layers': [l.name for l in self.model.layers]}
        if self.input_data is not None:
            ret['input_data'] = self.serialize_array(self.input_data)

        # https://stackoverflow.com/questions/41711190/keras-how-to-get-the-
        # output-of-each-layer
        inp = self.model.input
        outputs = [l.output for l in self.model.layers]
        self._functor = K.function([inp], outputs)

        return ret

    def _get_outputs(self):
        outs = self._functor([self.input_data])
        ls = self.model.layers
        return {l.name: self.serialize_array(o) for l,o in zip(ls,outs)}

    def _bin_weights(self):
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
                ret[l.name][vn] = self._bin_array(w)

        return ret

    def _bin_array(self, a):
        '''Binning an array and returning it as a dictionary.'''
        hist, bin_edges = np.histogram(a, self.nbins)
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

        dump['weights'] = self._bin_weights()
        if self.input_data is not None:
            dump['outputs'] = self._get_outputs()

        self._write_epoch(dump, epoch)

    def _write_epoch(self, dump, epoch):
        pass

    def on_train_end(self, logs=None):
        pass

    def on_train_begin(self, logs=None):
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
        input_data=None, array_srl='base64'):
        super(Collector,self).__init__(logs_keys, nbins, input_data, array_srl)
        self.logfile = logfile
        if add_ext and not self.logfile.endswith('.json'):
            self.logfile += '.json'
        self.file = None

    def _write_epoch(self, dump, epoch):
        if epoch > 0:
            self.file.write(',')

        json.dump(dump, self.file)

    def on_train_begin(self, logs=None):
        '''Open file.'''
        try:
            self.file = open(self.logfile,'w')
        except Exception as e:
            msg = "cannot open file {}".format(self.logfile)
            raise CollectorError(msg) from e

        # dump validation data and layers
        json_str = json.dumps(self._train_begin())
        # begin JSON file
        self.file.write(json_str[:-1])
        # begin training collection
        self.file.write(',"training" : [')

    def on_train_end(self, logs=None):
        '''Dump layers close file.'''
        # end of the collection and JSON file
        self.file.write(']}')
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
    '''
    def __init__(self, collection_name, db_name='nnviz_db', logs_keys=None,
        nbins=50, input_data=None, array_srl='base64'):
        super(MongoCollector,self).__init__(logs_keys, nbins, input_data,
            array_srl)
        
        self.db_name = db_name
        self.collection_name = collection_name
        self.client = None
        self.collection = None

    def on_train_begin(self, logs=None):
        self.client = MongoClient()
        db = self.client[self.db_name]
        self.collection = db[self.collection_name]

    def on_train_end(self, logs=None):
        dump = self._train_begin()
        self.collection.insert(dump)
        self.client.close()

    def _write_epoch(self, dump, epoch):
        self.collection.insert(dump)