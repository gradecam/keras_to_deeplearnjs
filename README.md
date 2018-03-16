# keras_to_deeplearnjs: Tool for converting keras models to deeplearnjs

## Dependencies
To run this tool, you will need:

- python3 (python2 not currently supported)
- keras
- h5py (required by keras to read/write models to file)

In addition, if you want to use the demo projects:

- npm


## Basic Usage
```
python3 keras_to_deeplearnjs/run.py <input keras model.h5> <output commonjs module.js>
```

## Supported Layers
Currently the following layers are supported:

- Dense
- Activation
- Conv2D
- MaxPooling2D
- Dropout
- Reshape
- Flatten
- BatchNormalization

It is quite easy to add support for additional layers, so more layers will be coming soon, or feel free to submit a PR.

## Demos
Demos are contained in the `demo` directory.  To install the demo dependencies:
```
cd demo
npm install
```

To build the demos:
```
npm run build
```

To start a webserver for testing the demos:
```
npm run serve
```

