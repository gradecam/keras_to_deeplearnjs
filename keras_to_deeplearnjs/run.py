import argparse
parser = argparse.ArgumentParser()
parser.add_argument("input", help="keras model to convert in keras's h5py format")
parser.add_argument("output", help="output filename of javascript module")
parser.add_argument("--weights", help="output filename of weight file, if not present the weights will be inlined")

args = parser.parse_args()

from keras.models import load_model
import pdb
from convert import get_converter

model = load_model(args.input)
out_file = open(args.output, 'w')

layers = [ get_converter(layer) for layer in model.layers ]

def write_preamble(outf):
    global args
    outf.write(\
"""
let dl = require('deeplearn');
module.exports = {};
let weights = {};
""")
    if args.weights:
        outf.write("""\
console.log("Fetching Weights");
weightPromise = fetch('weights').then(function(response) { return response.arrayBuffer(); }).catch((err) => console.log("Error loading weights: ", err));
module.exports.load = function() { return weightPromise; }
""")
    else:
        outf.write("module.exports.load = function() { return Promise.resolve(); }\n")


def write_weights(outf, layers):
    global args
    data = bytearray() if args.weights else None

    if args.weights: outf.write("weightPromise.then(function(weightBuf) {\n");
    for layer in layers:
        for key,value in layer.get_deeplearn_weights(data).items():
            outf.write("weights['{key}'] = {value}\n".format(key=key, value=value))
    outf.write("console.log('weights loaded');")
    if args.weights: outf.write("});\n")

    if data:
        weight_file = open(args.weights, 'wb')
        weight_file.write(data)
        weight_file.close()

def write_infer(outf, model, layers):
    outf.write(\
"""
module.exports.infer = function infer(input) {{
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

