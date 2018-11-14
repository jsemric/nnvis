import numpy as np
import base64

def get_array(d):
    shape = d['shape']
    data = d['data']
    a = np.frombuffer(base64.b64decode(data), np.float32)
    return a.reshape(shape)