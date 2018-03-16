import keras
from keras import backend as K
from types import SimpleNamespace
import json

class LayerConverter:
    def __init__(self, layer):
        global math_ops
        self.layer = layer
        self.subs = {}
        self.subs['input'] = "layers['{name}']".format(name=layer.input.name)
        if hasattr(self.__class__, 'attrs'):
            for attr in self.__class__.attrs:
                self.subs[attr] = json.dumps(getattr(layer, attr))


    def format_op(self, op):
        return op.format(**self.subs)

    def create_op(self, op):
        return lambda *args: op.format(*[arg.format(**self.subs) for arg in args])

    def get_output_name(self):
        return self.layer.output.name

    def get_deeplearn_weights(self):
        weight_attrs = getattr(self.__class__, 'weights', [])

        if getattr(self.layer, 'bias', None) is not None and not 'bias' in weight_attrs:
            weight_attrs.append('bias')

        weights = {}

        for attr in weight_attrs:
            weight = getattr(self.layer, attr)
            name = weight.name
            values=K.eval(weight)
            vals =[round(x, 3) for x in  values.flatten().tolist()]
            weights[name] = "dl.tensor({values}, {shape})".format(values=json.dumps(vals), shape=json.dumps(values.shape))
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


def get_converter(layer):
    for converter in LayerConverter.__subclasses__():
        if converter.kerasLayer == layer.__class__:
            return converter(layer)
    raise Exception("Unsupported layer: " + str(layer));

