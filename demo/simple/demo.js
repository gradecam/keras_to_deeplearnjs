let model = require('./model')
let dl = require('deeplearn')
model.load().then(function() {
    model.infer(dl.tensor2d([[3,5]])).data().then((x) => console.log("3,5", x) )
    model.infer(dl.tensor2d([[8,5]])).data().then((x) => console.log("8,5", x) )
    model.infer(dl.tensor2d([[3,1]])).data().then((x) => console.log("3,1", x) )
});

