let infer = require('./model')
let dl = require('deeplearn')
infer(dl.tensor2d([[3,5]])).data().then((x) => console.log("3,5", x) )
infer(dl.tensor2d([[8,5]])).data().then((x) => console.log("8,5", x) )
infer(dl.tensor2d([[3,1]])).data().then((x) => console.log("3,1", x) )

