"use strict";
/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 *
 * Originally obtained from deeplearnjs (https://github.com/PAIR-code/deeplearnjs/blob/master/demos/mnist/mnist.ts)
 * Modified by Robert Porter (3/15/2018) for keras_to_deeplearnjs example
 *
 */
var dl = require("deeplearn");
var infer = require('./model');

// Get sample data.
const xhr = new XMLHttpRequest();
xhr.open('GET', 'sample_data.json');
xhr.onload = async () => {
    const data = JSON.parse(xhr.responseText);
    console.log(`Evaluation set: n=${data.images.length}.`);
    let numCorrect = 0;
    for (let i = 0; i < data.images.length; i++) {
        const inferred = dl.tidy(() => {
            const x = dl.tensor(data.images[i], [28,28,1]);
            return infer(x).argMax();
        });
        const predictedLabel = Math.round((await inferred.data())[0]);
        inferred.dispose();
        console.log(`Item ${i}, predicted label ${predictedLabel}.`);
       // Aggregate correctness to show accuracy.
        const label = data.labels[i];
        if (label === predictedLabel) { numCorrect++; }

        // Show the image.
        dl.tidy(() => {
            const result = renderResults(dl.tensor1d(data.images[i]), label, predictedLabel);
            document.body.appendChild(result);
        });
    }

    // Compute final accuracy.
    const accuracy = numCorrect * 100 / data.images.length;
    document.getElementById('accuracy').innerHTML = `${accuracy}%`;
};
xhr.onerror = (err) => console.error(err);
xhr.send();

function renderMnistImage(array) {
    var width = 28;
    var height = 28;
    var canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    var ctx = canvas.getContext('2d');
    var float32Array = array.dataSync();
    var imageData = ctx.createImageData(width, height);
    for (var i = 0; i < float32Array.length; ++i) {
        var j = i * 4;
        var value = Math.round(float32Array[i] * 255);
        imageData.data[j + 0] = value;
        imageData.data[j + 1] = value;
        imageData.data[j + 2] = value;
        imageData.data[j + 3] = 255;
    }
    ctx.putImageData(imageData, 0, 0);
    return canvas;
}
function renderResults(array, label, predictedLabel) {
    var root = document.createElement('div');
    root.appendChild(renderMnistImage(array));
    var actual = document.createElement('div');
    actual.innerHTML = "Actual: " + label;
    root.appendChild(actual);
    var predicted = document.createElement('div');
    predicted.innerHTML = "Predicted: " + predictedLabel;
    root.appendChild(predicted);
    if (label !== predictedLabel) {
        root.classList.add('error');
    }
    root.classList.add('result');
    return root;
}

