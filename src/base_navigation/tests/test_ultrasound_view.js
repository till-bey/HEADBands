const assert = require('node:assert/strict');
const {classify, groupReadings} = require('../ultrasound_view.js');
const reading = (topic, side, distance_m, status = 'valid', fresh = true) =>
  ({topic, side, distance_m, status, fresh});

assert.equal(classify(reading('/front', 'front', 0.30)), 'danger');
assert.equal(classify(reading('/front', 'front', 0.60)), 'warning');
assert.equal(classify(reading('/front', 'front', 0.61)), 'clear');
assert.equal(classify(reading('/front', 'front', 0.1, 'valid', false)), 'unknown');
assert.equal(classify(reading('/front', 'front', 0.05, 'too_close')), 'danger');
assert.equal(classify(reading('/front', 'front', 4, 'beyond_range')), 'clear');
assert.equal(classify(reading('/front', 'front', 0.1, 'beyond_range')), 'unknown');
const data = [reading('/front1', 'front', 0.9), reading('/front2', 'front', 0.2),
  reading('/sensor3', null, 0.45), reading('/left', 'left', null, 'invalid')];
let groups = groupReadings(data);
assert.equal(groups.front.level, 'danger');
assert.equal(groups.front.label, '0.20 m');
assert.equal(groups.rear.level, 'unknown');
assert.equal(groups.left.level, 'unknown');
groups = groupReadings(data, {rear: ['/sensor3']});
assert.equal(groups.rear.level, 'warning');
groups = groupReadings(data, {front: ['/front1', '/missing']});
assert.equal(groups.front.level, 'unknown'); // Incomplete coverage cannot be green.
groups = groupReadings(data, {front: ['/front2', '/missing']});
assert.equal(groups.front.level, 'danger'); // A known close obstacle still wins.
console.log('Ultrasound color, bounds, grouping and missing-data checks passed.');
