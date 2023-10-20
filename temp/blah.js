var demo = {};
var x = {id:1};
y = {id:2};
demo[x] = "alpha";
demo[y] = "beta";
console.log(demo[x]); // "beta"
console.log(demo[y]); // "beta"