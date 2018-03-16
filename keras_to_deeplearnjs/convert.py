import keras
from keras import backend as K
from types import SimpleNamespace
import json


class LayerConverter:
    "Base class for all layer converters"
    def __init__(self, layer):
        self.layer = layer
        self.subs = {}
        self.subs['input'] = "layers['{name}']".format(name=layer.input.name)
        for attr in getattr(self.__class__, 'attrs', []):
            self.subs[attr] = json.dumps(getattr(layer, attr))

    def format_op(self, op):
        return op.format(**self.subs)

    def get_output_name(self):
        return self.layer.output.name

    def _save_weight_json(self, values):
        vals =[round(x, 3) for x in  values.flatten().tolist()]
        return "dl.tensor({values}, {shape})".format(values=json.dumps(vals), shape=json.dumps(values.shape))

    def _save_weight_bytearray(self, values, byteArray):
        begin = len(byteArray)
        byteArray += values.flatten().astype('float32').tobytes()
        end = len(byteArray)
        return "dl.tensor(new Float32Array(weightBuf.slice({begin}, {end})), {shape})".format(begin=begin, end=end, shape=json.dumps(values.shape))

    def get_deeplearn_weights(self, byteArray = None):
        "Save the weights for deeplearn.js.  If byteArray is passed it is assumed that we are not inlining the weights"

        weight_attrs = getattr(self.__class__, 'weights', [])

        if getattr(self.layer, 'bias', None) is not None and not 'bias' in weight_attrs:
            weight_attrs.append('bias')

        weights = {}

        for attr in weight_attrs:
            weight = getattr(self.layer, attr)
            name = weight.name
            values=K.eval(weight)
            if byteArray is not None: weights[name] = self._save_weight_bytearray(values, byteArray)
            else: weights[name] = self._save_weight_json(values)
            self.subs[attr] = "weights['{name}']".format(name=name)
        return weights


    def get_deeplearn_op(self):
        "Default implementation of get_deeplearn_op.  Automatically adds activation and bias if relevant"
        cls = self.__class__
        result = self.format_op(cls.op)
        use_bias =getattr(self.layer, 'use_bias', False)
        if use_bias: result = "dl.add({},{})".format(result, self.subs['bias'])
        activation = getattr(self.layer, 'activation', False)
        if activation == keras.activations.relu: result = "dl.relu({})".format(result)
        elif activation == keras.activations.linear: pass
        elif activation == keras.activations.softmax: result = "dl.softmax({})".format(result)
        elif activation is not False: raise Exception("Unknown activation function: " + str(self.layer.activation))

        return result

class Input(LayerConverter):
    kerasLayer = keras.engine.topology.InputLayer
    op = "input"

class Dense(LayerConverter):
    kerasLayer =  keras.layers.core.Dense
    weights = ['kernel']
    op = "dl.matMul(dl.reshape({input}, [1,-1]), {kernel})"

class Activation(LayerConverter):
    kerasLayer = keras.layers.core.Activation
    op = '{input}' # Activation gets added automatically

class Conv2D(LayerConverter):
    kerasLayer = keras.layers.convolutional.Conv2D
    weights = ['kernel']
    attrs = ['strides', 'padding']
    op = "dl.conv2d({input}, {kernel}, {strides}, {padding})"

class MaxPooling2D(LayerConverter):
    kerasLayer = keras.layers.pooling.MaxPooling2D
    attrs = ['pool_size', 'padding', 'strides']
    op = "dl.maxPool({input}, {pool_size}, {strides}, {padding})"

class Dropout(LayerConverter):
    kerasLayer = keras.layers.core.Dropout
    op = "{input}" # no op for dropout in inference mode

class Reshape(LayerConverter):
    kerasLayer = keras.layers.core.Reshape
    attrs = ['target_shape']
    op = "dl.reshape({input}, {target_shape})"

class Flatten(LayerConverter):
    kerasLayer = keras.layers.core.Flatten
    op = "dl.reshape({input}, [-1])"

class BatchNormalization(LayerConverter):
    kerasLayer = keras.layers.normalization.BatchNormalization
    weights = ['moving_mean', 'moving_variance']
    op = "dl.batchNormalization({input}, {moving_mean}, {moving_variance})"


def get_converter(layer):
    for converter in LayerConverter.__subclasses__():
        if converter.kerasLayer == layer.__class__:
            return converter(layer)
    raise Exception("Unsupported layer: " + str(layer));

