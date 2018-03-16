import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="keras model to convert in keras's h5py format")
parser.add_argument("output", help="javascript output filename")
args = parser.parse_args()

from keras.models import load_model
import pdb
from convert import get_converter

model = load_model(args.input)
out_file = open(args.output, 'w')

layers = [ get_converter(layer) for layer in model.layers ]

def write_preamble(outf):
    outf.write(\
"""
let dl = require('deeplearn');
const math = dl.ENV.math;
let weights = {}
""")

def write_weights(outf, layers):
    for layer in layers:
        for key,value in layer.get_deeplearn_weights().items():
            outf.write("weights['{key}'] = {value}\n".format(key=key, value=value))

def write_infer(outf, model, layers):
    outf.write(\
"""
module.exports = function infer(input) {{
    layers = {{}}
    layers['{input_name}'] = input;
""".format(input_name = model.input.name))

    for layer in layers:
        outf.write("    layers['{key}'] = {op};\n".format(key = layer.get_output_name(), op = layer.get_deeplearn_op()))

    outf.write(\
"""
    return layers['{output_name}'];
}} // end infer
""".format(output_name = model.output.name))
write_preamble(out_file)
write_weights(out_file, layers)
write_infer(out_file, model, layers)
out_file.close()

