#!/usr/bin/env python3
"""Interactive 3D viewer for curated backhand and flick trajectories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Curated throw trajectories</title><style>
:root{color-scheme:dark;--bg:#0e1518;--panel:#172226;--line:#35464b;--text:#edf3ef;--muted:#91a5a1;--orange:#f3ad53;--cyan:#73d0c0;--pink:#ef8f75}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% -10%,#30423d,#0e1518 48%);color:var(--text);font:14px system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:20px}.eyebrow{color:var(--orange);font-size:12px;font-weight:700;letter-spacing:.14em;text-transform:uppercase}.title{font:700 35px Georgia,serif;margin:5px 0}.summary{color:var(--muted);text-align:right}.layout{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:16px}.panel{background:#172226e8;border:1px solid var(--line);border-radius:8px;padding:18px;box-shadow:0 15px 40px #0004}.scene{height:min(70vh,680px);min-height:440px;background:linear-gradient(145deg,#132126,#0b1114);border:1px solid #2d4045;border-radius:6px;position:relative;overflow:hidden}.scene canvas{width:100%;height:100%;display:block;cursor:grab}.scene canvas:active{cursor:grabbing}.scene-label{position:absolute;left:16px;top:14px;color:var(--muted);font-size:12px;pointer-events:none}.toolbar{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:14px}.button,.select{border:1px solid var(--line);border-radius:5px;background:#24353a;color:var(--text);padding:9px 12px;font:inherit}.button{cursor:pointer;font-weight:700}.button:hover,.button.active{background:var(--orange);border-color:var(--orange);color:#17120a}.select{width:100%;margin-bottom:18px}.side-title{font-size:12px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 10px}.stat{display:flex;justify-content:space-between;border-top:1px solid #2b3a3f;padding:10px 0;color:var(--muted)}.stat strong{color:var(--text);font-variant-numeric:tabular-nums}.legend{display:flex;gap:18px;flex-wrap:wrap;color:var(--muted);margin-top:12px;font-size:13px}.swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:5px}.note{color:var(--muted);font-size:12px;line-height:1.5;margin-top:18px}@media(max-width:850px){main{padding:16px 12px}.top{align-items:start}.title{font-size:28px}.layout{grid-template-columns:1fr}.scene{height:60vh;min-height:360px}.summary{text-align:right}}
</style></head><body><main><header class="top"><div><div class="eyebrow">IMU trajectory atlas</div><h1 class="title">Curated throws in 3D</h1></div><div class="summary" id="summary"></div></header><div class="layout"><section class="panel"><div class="toolbar"><button class="button active" data-mode="all">All throws</button><button class="button" data-mode="backhand">Backhand</button><button class="button" data-mode="flick">Flick</button></div><div class="scene"><canvas id="scene"></canvas><div class="scene-label">drag to orbit · scroll to zoom</div></div><div class="legend"><span><i class="swatch" style="background:#73d0c0"></i>backhand</span><span><i class="swatch" style="background:#ef8f75"></i>flick</span><span>bright line = selected throw</span></div></section><aside class="panel"><h2 class="side-title">Inspect a throw</h2><select class="select" id="throwSelect"></select><div class="stat"><span>Technique</span><strong id="technique">-</strong></div><div class="stat"><span>Samples</span><strong id="samples">-</strong></div><div class="stat"><span>Duration</span><strong id="duration">-</strong></div><div class="stat"><span>Path length</span><strong id="pathLength">-</strong></div><p class="note">The position is reconstructed by double-integrating calibrated acceleration after removing the average acceleration of the first stationary samples. It is relative sensor-frame position, so long-term drift is expected.</p></aside></div></main><script>
let data=null,mode='all',selected=0,yaw=-.65,pitch=.38,zoom=1,dragging=false,lastX=0,lastY=0;const $=id=>document.getElementById(id);const canvas=$('scene');
const colors={backhand:'#73d0c0',flick:'#c59cf2'};
async function init(){data=await(await fetch('/api/data')).json();$('summary').textContent=`${data.recordings.length} throws · ${data.counts.backhand} backhand · ${data.counts.flick} flick`;populate();draw()}
function visible(){return data.recordings.map((item,i)=>({item,i})).filter(pair=>mode==='all'||pair.item.kind===mode)}
function populate(){let options=visible();if(!options.length){$('throwSelect').innerHTML='';return}if(!options.some(pair=>pair.i===selected))selected=options[0].i;$('throwSelect').innerHTML=options.map(pair=>`<option value="${pair.i}">${pair.item.label}</option>`).join('');$('throwSelect').value=selected;updateStats()}
function updateStats(){let item=data.recordings[selected];if(!item)return;$('technique').textContent=item.kind;$('samples').textContent=item.time.length;$('duration').textContent=(item.time[item.time.length-1]-item.time[0]).toFixed(3)+' s';$('pathLength').textContent=pathLength(item.position).toFixed(3)+' m'}
function pathLength(pos){let total=0;for(let i=1;i<pos.x.length;i++)total+=Math.hypot(pos.x[i]-pos.x[i-1],pos.y[i]-pos.y[i-1],pos.z[i]-pos.z[i-1]);return total}
function project(point,center,scale,w,h){let [x,y,z]=point;let cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);let rx=cy*x-sy*z,rz=sy*x+cy*z,ry=cp*y-sp*rz,depth=sp*y+cp*rz;let perspective=1/(1+depth*.18);return [w/2+(rx-center[0])*scale*perspective,h/2-(ry-center[1])*scale*perspective,depth]}
function draw(){if(!data)return;let rect=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=rect.width*d;canvas.height=rect.height*d;let c=canvas.getContext('2d');c.scale(d,d);let w=rect.width,h=rect.height;let paths=visible(),points=paths.flatMap(pair=>pair.item.position.x.map((_,i)=>[pair.item.position.x[i],pair.item.position.y[i],pair.item.position.z[i]]));if(!points.length)return;let center=[0,0,0];for(const point of points)for(let axis=0;axis<3;axis++)center[axis]+=point[axis]/points.length;let extent=Math.max(...points.map(point=>Math.max(...point.map((value,axis)=>Math.abs(value-center[axis])))),.001),scale=Math.min(w,h)*.36/extent*zoom;
c.fillStyle='#0b1114';c.fillRect(0,0,w,h);drawGrid(c,center,scale,w,h);let projected=[];for(const pair of paths){let item=pair.item,trail=[];for(let i=0;i<item.position.x.length;i++)trail.push(project([item.position.x[i],item.position.y[i],item.position.z[i]],center,scale,w,h));projected.push({pair,trail})}projected.sort((a,b)=>a.trail.reduce((s,p)=>s+p[2],0)/a.trail.length-b.trail.reduce((s,p)=>s+p[2],0)/b.trail.length);for(const entry of projected){let isSelected=entry.pair.i===selected;c.strokeStyle=isSelected?'#ffffff':colors[entry.pair.item.kind];c.globalAlpha=isSelected?1:.48;c.lineWidth=isSelected?3:1.2;c.beginPath();entry.trail.forEach((point,i)=>(i?c.lineTo:c.moveTo).call(c,point[0],point[1]));c.stroke();if(isSelected){let first=entry.trail[0],last=entry.trail[entry.trail.length-1];c.fillStyle='#fff';c.beginPath();c.arc(first[0],first[1],4,0,Math.PI*2);c.fill();c.fillStyle=colors[entry.pair.item.kind];c.beginPath();c.arc(last[0],last[1],5,0,Math.PI*2);c.fill()}}c.globalAlpha=1}
function drawGrid(c,center,scale,w,h){let axes=[[[0,0,0],[1,0,0]],[[0,0,0],[0,1,0]],[[0,0,0],[0,0,1]]],axisColors=['#ef8f75','#8fc7f2','#c59cf2'];axes.forEach((axis,i)=>{let a=project(axis[0],center,scale,w,h),b=project(axis[1].map(value=>value*center.map(()=>1)[i]),center,scale,w,h);c.strokeStyle=axisColors[i]+'88';c.lineWidth=1;c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.stroke()});c.strokeStyle='#29383d';c.lineWidth=1;for(let i=-2;i<=2;i++){let a=project([i*center[0]+i/2,0,0],center,scale,w,h),b=project([i*center[0]+i/2,0,1],center,scale,w,h);c.beginPath();c.moveTo(a[0],a[1]);c.lineTo(b[0],b[1]);c.stroke()}}
document.querySelectorAll('[data-mode]').forEach(button=>button.onclick=()=>{mode=button.dataset.mode;document.querySelectorAll('[data-mode]').forEach(other=>other.classList.toggle('active',other===button));populate();draw()});$('throwSelect').onchange=event=>{selected=+event.target.value;updateStats();draw()};canvas.onpointerdown=event=>{dragging=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId)};canvas.onpointermove=event=>{if(!dragging)return;yaw+=(event.clientX-lastX)*.008;pitch=Math.max(-1.35,Math.min(1.35,pitch+(event.clientY-lastY)*.008));lastX=event.clientX;lastY=event.clientY;draw()};canvas.onpointerup=()=>dragging=false;canvas.onwheel=event=>{event.preventDefault();zoom=Math.max(.35,Math.min(4,zoom*(event.deltaY<0?1.1:.9)));draw()};window.onresize=draw;init();
</script></body></html>'''


TRIM_MARKUP = '''<div class="trim-box"><h2 class="side-title">Shorten selected throw</h2><div class="trim-values"><label>Start <output id="trimStartValue">-</output></label><label>End <output id="trimEndValue">-</output></label></div><input id="trimStart" type="range"><input id="trimEnd" type="range"><button class="button primary" id="saveShortened">Save shortened trajectory</button><div class="trim-status" id="trimStatus"></div></div>'''
TRIM_STYLE = '''<style>.trim-box{border-top:1px solid #2b3a3f;margin-top:18px;padding-top:16px}.trim-values{display:flex;justify-content:space-between;color:var(--muted);font-size:12px;margin-bottom:8px}.trim-values output{color:var(--text);font-variant-numeric:tabular-nums}.trim-box input{width:100%;accent-color:var(--orange);display:block;margin:10px 0}.trim-box .button{width:100%;margin-top:8px}.trim-status{color:var(--cyan);font-size:12px;line-height:1.4;margin-top:9px;min-height:17px}</style>'''
TRIM_SCRIPT = r"""
const originalPopulate = populate;
const originalDraw = draw;
let trimStart = 0;
let trimEnd = 0;
function selectedTimes() {
	const item = data.recordings[selected];
	return item ? item.time : [];
}
function configureTrim() {
	const times = selectedTimes();
	if (!times.length) return;
	const start = $('trimStart');
	const end = $('trimEnd');
	start.min = times[0]; start.max = times[times.length - 1]; start.step = (times[1] - times[0]) || .001;
	end.min = start.min; end.max = start.max; end.step = start.step;
	trimStart = +start.value || +start.min;
	trimEnd = +end.value || +end.max;
	start.value = trimStart; end.value = trimEnd;
	updateTrimLabels();
}
function updateTrimLabels() {
	$('trimStartValue').textContent = (+$('trimStart').value).toFixed(3) + ' s';
	$('trimEndValue').textContent = (+$('trimEnd').value).toFixed(3) + ' s';
}
function trimIndices(item) {
	const start = +$('trimStart').value;
	const end = +$('trimEnd').value;
	let first = item.time.findIndex(time => time >= start);
	let last = item.time.findIndex(time => time > end);
	if (first < 0) first = 0;
	if (last < 0) last = item.time.length;
	return [first, Math.max(first + 1, last)];
}
function clippedPosition(item) {
	const [first, last] = trimIndices(item);
	return {x: item.position.x.slice(first, last), y: item.position.y.slice(first, last), z: item.position.z.slice(first, last)};
}
populate = function() {
	originalPopulate();
	configureTrim();
};
draw = function() {
	if (!data) return;
	const item = data.recordings[selected];
	const originalPosition = item && item.position;
	if (item && $('trimStart')) item.position = clippedPosition(item);
	originalDraw();
	if (item) item.position = originalPosition;
};
function updateTrim() {
	let start = +$('trimStart').value;
	let end = +$('trimEnd').value;
	if (start > end) {
		if (this.id === 'trimStart') end = start;
		else start = end;
	}
	$('trimStart').value = start; $('trimEnd').value = end;
	updateTrimLabels(); draw();
}
$('trimStart').oninput = updateTrim;
$('trimEnd').oninput = updateTrim;
$('throwSelect').onchange = event => { selected = +event.target.value; updateStats(); configureTrim(); draw(); };
$('saveShortened').onclick = async function() {
		const status = $('trimStatus');
		status.textContent = 'Saving...';
		const response = await fetch('/api/save-shortened', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({index: selected, start: +$('trimStart').value, end: +$('trimEnd').value})});
		const result = await response.json();
		status.textContent = response.ok ? `Saved ${result.path}` : `Save failed: ${result.error}`;
};
const trimBootstrap = setInterval(() => { if (data) { configureTrim(); draw(); clearInterval(trimBootstrap); } }, 25);
"""
HTML = HTML.replace("</aside>", TRIM_MARKUP + "</aside>", 1)
HTML = HTML.replace("</head>", TRIM_STYLE + "</head>", 1)
HTML = HTML.replace("</script>", TRIM_SCRIPT + "</script>", 1)
HTML = HTML.replace("drag to orbit · scroll to zoom", "drag to orbit · scroll to zoom · peak at origin", 1)
HTML = HTML.replace("relative sensor-frame position", "reoriented throw-frame position (X forward, Y left, Z up)", 1)
HTML = HTML.replace("background:#ef8f75", "background:#c59cf2", 1)
FLATTEN_MARKUP = '''<div class="trim-box"><h2 class="side-title">Rotate around X</h2><div class="trim-values"><label>Angle <output id="rotateAngleValue">0.0°</output></label><span>-180° to +180°</span></div><input id="rotateAngle" type="range" min="-180" max="180" value="0" step="0.1"><button class="button primary" id="saveFlattened">Save to flatten/</button><div class="trim-status" id="flattenStatus"></div></div>'''
FLATTEN_SCRIPT = r"""
const rotateAngles = {};
function currentRotateAngle() { return rotateAngles[selected] || 0; }
function syncRotateControl() { $('rotateAngle').value = currentRotateAngle(); $('rotateAngleValue').textContent = currentRotateAngle().toFixed(1) + '°'; }
function rotatePosition(item, index) {
	const angle = (rotateAngles[index] || 0) * Math.PI / 180;
	const cosine = Math.cos(angle), sine = Math.sin(angle);
	return {x: item.position.x.slice(), y: item.position.y.map((value, i) => cosine * value - sine * item.position.z[i]), z: item.position.y.map((value, i) => sine * value + cosine * item.position.z[i])};
}
const trimDraw = draw;
draw = function() {
	if (!data) return;
	const originals = data.recordings.map(item => item.position);
	for (const pair of visible()) pair.item.position = rotatePosition(pair.item, pair.i);
	trimDraw();
	data.recordings.forEach((item, index) => { item.position = originals[index]; });
};
$('rotateAngle').oninput = event => { rotateAngles[selected] = +event.target.value; syncRotateControl(); draw(); };
$('throwSelect').onchange = event => { selected = +event.target.value; updateStats(); configureTrim(); syncRotateControl(); draw(); };
$('saveFlattened').onclick = async function() {
	const status = $('flattenStatus');
	status.textContent = 'Saving...';
	const response = await fetch('/api/save-flattened', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({index: selected, start: +$('trimStart').value, end: +$('trimEnd').value, angle: currentRotateAngle()})});
	const result = await response.json();
	status.textContent = response.ok ? `Saved ${result.path}` : `Save failed: ${result.error}`;
};
const trimPopulate = populate;
populate = function() { trimPopulate(); syncRotateControl(); };
syncRotateControl();
"""
HTML = HTML.replace("</aside>", FLATTEN_MARKUP + "</aside>", 1)
HTML = HTML.replace("</script>", FLATTEN_SCRIPT + "</script>", 1)


def dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
	return sum(left * right for left, right in zip(first, second))


def cross(first: tuple[float, float, float], second: tuple[float, float, float]) -> tuple[float, float, float]:
	return (
		first[1] * second[2] - first[2] * second[1],
		first[2] * second[0] - first[0] * second[2],
		first[0] * second[1] - first[1] * second[0],
	)


def unit(vector: tuple[float, float, float]) -> tuple[float, float, float]:
	length = math.sqrt(dot(vector, vector))
	if length < 1e-12:
		return (0.0, 0.0, 0.0)
	return tuple(value / length for value in vector)


def principal_axes(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
	centroid = tuple(sum(point[axis] for point in points) / len(points) for axis in range(3))
	covariance = [[0.0] * 3 for _ in range(3)]
	for point in points:
		centered = [point[axis] - centroid[axis] for axis in range(3)]
		for row in range(3):
			for column in range(3):
				covariance[row][column] += centered[row] * centered[column]
		for row in range(3):
			for column in range(3):
				covariance[row][column] /= len(points)
	# Jacobi rotations diagonalize this small symmetric covariance matrix.
	basis = [[1.0 if row == column else 0.0 for column in range(3)] for row in range(3)]
	for _ in range(50):
		row, column = max(((row, column) for row in range(3) for column in range(row + 1, 3)), key=lambda pair: abs(covariance[pair[0]][pair[1]]))
		if abs(covariance[row][column]) < 1e-10:
			break
		angle = 0.5 * math.atan2(2.0 * covariance[row][column], covariance[row][row] - covariance[column][column])
		cosine, sine = math.cos(angle), math.sin(angle)
		for index in range(3):
			left, right = covariance[index][row], covariance[index][column]
			covariance[index][row] = cosine * left - sine * right
			covariance[index][column] = sine * left + cosine * right
		for index in range(3):
			upper, lower = covariance[row][index], covariance[column][index]
			covariance[row][index] = cosine * upper - sine * lower
			covariance[column][index] = sine * upper + cosine * lower
		for index in range(3):
			left, right = basis[index][row], basis[index][column]
			basis[index][row] = cosine * left - sine * right
			basis[index][column] = sine * left + cosine * right
	values = [covariance[index][index] for index in range(3)]
	return [unit(tuple(basis[row][column] for row in range(3))) for column in sorted(range(3), key=values.__getitem__)]


def parse_accelerometer(path: Path) -> dict[str, Any]:
	with path.open(newline="", encoding="utf-8-sig") as stream:
		rows = list(csv.DictReader(stream))
	times: list[float] = []
	acceleration: list[tuple[float, float, float]] = []
	for row in rows:
		try:
			times.append(float(row["seconds_elapsed"]))
			acceleration.append(tuple(float(row[axis]) for axis in ("x", "y", "z")))
		except (KeyError, TypeError, ValueError):
			continue
	if not acceleration:
		return {"time": [], "acceleration": [], "peak_index": 0, "x": [], "y": [], "z": []}
	magnitude = [sum(value * value for value in sample) ** 0.5 for sample in acceleration]
	peak_index = max(range(len(magnitude)), key=magnitude.__getitem__)
	baseline_count = min(10, len(acceleration))
	baseline = tuple(sum(sample[axis] for sample in acceleration[:baseline_count]) / baseline_count for axis in range(3))
	velocity = [0.0, 0.0, 0.0]
	position = [0.0, 0.0, 0.0]
	positions = {"time": [times[0]], "acceleration": magnitude, "peak_index": peak_index, "baseline": baseline, "x": [0.0], "y": [0.0], "z": [0.0]}
	for previous_time, current_time, previous, current in zip(times, times[1:], acceleration, acceleration[1:]):
		delta = max(0.0, current_time - previous_time)
		for axis, name in enumerate(("x", "y", "z")):
			previous_corrected = previous[axis] - baseline[axis]
			current_corrected = current[axis] - baseline[axis]
			position[axis] += velocity[axis] * delta + 0.25 * (previous_corrected + current_corrected) * delta * delta
			velocity[axis] += 0.5 * (previous_corrected + current_corrected) * delta
			positions[name].append(position[axis])
		positions["time"].append(current_time)
	return positions


def fit_throw_frame(points: list[tuple[float, float, float]]) -> dict[str, tuple[float, float, float]]:
	if len(points) < 3:
		return {"forward": (1.0, 0.0, 0.0), "left": (0.0, 1.0, 0.0), "up": (0.0, 0.0, 1.0)}
	principal = principal_axes(points)
	normal = principal[0]
	throw_vector = unit(tuple(points[-1][axis] - points[0][axis] for axis in range(3)))
	forward = unit(tuple(throw_vector[axis] - dot(throw_vector, normal) * normal[axis] for axis in range(3)))
	if dot(forward, throw_vector) < 0:
		forward = tuple(-value for value in forward)
	if dot(forward, forward) < 1e-12:
		forward = principal[2]
	left = unit(cross(normal, forward))
	# Old +X is left. Select the plane-normal sign that preserves that cue.
	if left[0] < 0:
		normal = tuple(-value for value in normal)
		left = tuple(-value for value in left)
	return {"forward": forward, "left": left, "up": normal}


def reorient_positions(trajectory: dict[str, Any], peak_position: dict[str, float]) -> dict[str, list[float]]:
	points = [
		(tuple(trajectory[axis][index] - peak_position[axis] for axis in ("x", "y", "z")))
		for index in range(len(trajectory["time"]))
	]
	frame = fit_throw_frame(points)
	return {
		"x": [dot(point, frame["forward"]) for point in points],
		"y": [dot(point, frame["left"]) for point in points],
		"z": [dot(point, frame["up"]) for point in points],
	}


def load_recordings(data_root: Path) -> list[dict[str, Any]]:
	recordings = []
	for kind in ("backhand", "flick"):
		for directory in sorted((data_root / kind).glob("*")):
			if directory.name.endswith("_shortended"):
				continue
			accelerometer = directory / "Accelerometer.csv"
			if not accelerometer.exists():
				continue
			trajectory = parse_accelerometer(accelerometer)
			peak_index = trajectory["peak_index"]
			peak_position = {axis: trajectory[axis][peak_index] for axis in ("x", "y", "z")}
			points = [tuple(trajectory[axis][index] - peak_position[axis] for axis in ("x", "y", "z")) for index in range(len(trajectory["time"]))]
			frame = fit_throw_frame(points)
			reoriented = reorient_positions(trajectory, peak_position)
			recordings.append({
				"kind": kind,
				"label": f"{kind}/{directory.name}",
				"time": trajectory["time"],
				"acceleration": trajectory["acceleration"],
				"peakIndex": peak_index,
				"peakTime": trajectory["time"][peak_index],
				"frame": frame,
				"position": reoriented,
			})
	return recordings


def trim_csv(text: str, start: float, end: float) -> str:
	from io import StringIO

	reader = csv.DictReader(StringIO(text))
	fields = reader.fieldnames or []
	if "seconds_elapsed" not in fields:
		return text
	rows = []
	for row in reader:
		try:
			if start <= float(row["seconds_elapsed"]) <= end:
				rows.append(row)
		except (KeyError, TypeError, ValueError):
			continue
	output = StringIO(newline="")
	writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
	writer.writeheader()
	writer.writerows(rows)
	return output.getvalue()


def save_shortened(source: Path, start: float, end: float) -> Path:
	destination = source.with_name(f"{source.name}_shortended")
	destination.mkdir(parents=True, exist_ok=True)
	for csv_path in source.glob("*.csv"):
		text = csv_path.read_text(encoding="utf-8-sig")
		(destination / csv_path.name).write_text(trim_csv(text, start, end), encoding="utf-8")
	return destination


def matrix_multiply(first: list[list[float]], second: list[list[float]]) -> list[list[float]]:
	return [[sum(first[row][index] * second[index][column] for index in range(3)) for column in range(3)] for row in range(3)]


def matrix_transpose(matrix: list[list[float]]) -> list[list[float]]:
	return [[matrix[column][row] for column in range(3)] for row in range(3)]


def quaternion_to_matrix(qx: float, qy: float, qz: float, qw: float) -> list[list[float]]:
	return [
		[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
		[2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
		[2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
	]


def matrix_to_quaternion(matrix: list[list[float]]) -> tuple[float, float, float, float]:
	trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
	if trace > 0:
		scale = math.sqrt(trace + 1.0) * 2
		qw = 0.25 * scale
		qx = (matrix[2][1] - matrix[1][2]) / scale
		qy = (matrix[0][2] - matrix[2][0]) / scale
		qz = (matrix[1][0] - matrix[0][1]) / scale
	elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
		scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2
		qw = (matrix[2][1] - matrix[1][2]) / scale
		qx = 0.25 * scale
		qy = (matrix[0][1] + matrix[1][0]) / scale
		qz = (matrix[0][2] + matrix[2][0]) / scale
	elif matrix[1][1] > matrix[2][2]:
		scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2
		qw = (matrix[0][2] - matrix[2][0]) / scale
		qx = (matrix[0][1] + matrix[1][0]) / scale
		qy = 0.25 * scale
		qz = (matrix[1][2] + matrix[2][1]) / scale
	else:
		scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2
		qw = (matrix[1][0] - matrix[0][1]) / scale
		qx = (matrix[0][2] + matrix[2][0]) / scale
		qy = (matrix[1][2] + matrix[2][1]) / scale
		qz = 0.25 * scale
	return qx, qy, qz, qw


def transform_orientation_csv(text: str, start: float, end: float, frame: dict[str, tuple[float, float, float]], angle: float) -> str:
	from io import StringIO

	reader = csv.DictReader(StringIO(text))
	fields = reader.fieldnames or []
	quaternion_fields = ("qx", "qy", "qz", "qw")
	if not all(field in fields for field in quaternion_fields):
		return trim_csv(text, start, end)
	basis = [list(frame[name]) for name in ("forward", "left", "up")]
	angle_radians = math.radians(angle)
	cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
	rotation_x = [[1.0, 0.0, 0.0], [0.0, cosine, -sine], [0.0, sine, cosine]]
	change = matrix_multiply(rotation_x, basis)
	change_inverse = matrix_transpose(change)
	rows = []
	for row in reader:
		try:
			timestamp = float(row["seconds_elapsed"])
			quaternion = tuple(float(row[field]) for field in quaternion_fields)
		except (KeyError, TypeError, ValueError):
			continue
		if not start <= timestamp <= end:
			continue
		old_rotation = quaternion_to_matrix(*quaternion)
		new_rotation = matrix_multiply(matrix_multiply(change, old_rotation), change_inverse)
		qx, qy, qz, qw = matrix_to_quaternion(new_rotation)
		row.update({"qx": qx, "qy": qy, "qz": qz, "qw": qw})
		row["roll"] = math.atan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
		row["pitch"] = math.asin(max(-1.0, min(1.0, 2 * (qw * qy - qz * qx))))
		row["yaw"] = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
		rows.append(row)
	output = StringIO(newline="")
	writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
	writer.writeheader()
	writer.writerows(rows)
	return output.getvalue()


def save_flattened(source: Path, recording: dict[str, Any], start: float, end: float, angle: float, data_root: Path) -> Path:
	destination = data_root / "flatten" / recording["kind"] / f"{source.name}_flattened"
	destination.mkdir(parents=True, exist_ok=True)
	for csv_path in source.glob("*.csv"):
		text = csv_path.read_text(encoding="utf-8-sig")
		if csv_path.name == "Orientation.csv":
			text = transform_orientation_csv(text, start, end, recording["frame"], angle)
		else:
			text = trim_csv(text, start, end)
		(destination / csv_path.name).write_text(trim_csv(text, start, end), encoding="utf-8")
	angle_radians = math.radians(angle)
	cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
	rows = []
	for index, timestamp in enumerate(recording["time"]):
		if start <= timestamp <= end:
			x = recording["position"]["x"][index]
			y = recording["position"]["y"][index]
			z = recording["position"]["z"][index]
			rows.append({
				"seconds_elapsed": timestamp,
				"x": x,
				"y": cosine * y - sine * z,
				"z": sine * y + cosine * z,
				"acceleration_magnitude": recording["acceleration"][index],
			})
	with (destination / "Trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
		writer = csv.DictWriter(stream, fieldnames=("seconds_elapsed", "x", "y", "z", "acceleration_magnitude"))
		writer.writeheader()
		writer.writerows(rows)
	return destination


def make_handler(payload: dict[str, Any], data_root: Path):
	class Handler(BaseHTTPRequestHandler):
		def send_json(self, value: dict[str, Any], status: int = 200) -> None:
			body = json.dumps(value).encode()
			self.send_response(status)
			self.send_header("Content-Type", "application/json")
			self.send_header("Content-Length", str(len(body)))
			self.end_headers()
			self.wfile.write(body)

		def do_GET(self) -> None:  # noqa: N802
			if self.path == "/api/data":
				body = json.dumps(payload).encode()
				self.send_response(200)
				self.send_header("Content-Type", "application/json")
				self.send_header("Content-Length", str(len(body)))
				self.end_headers()
				self.wfile.write(body)
				return
			body = HTML.encode()
			self.send_response(200)
			self.send_header("Content-Type", "text/html; charset=utf-8")
			self.send_header("Content-Length", str(len(body)))
			self.end_headers()
			self.wfile.write(body)

		def do_POST(self) -> None:  # noqa: N802
			if self.path not in ("/api/save-shortened", "/api/save-flattened"):
				self.send_error(404)
				return
			try:
				length = int(self.headers.get("Content-Length", 0))
				request = json.loads(self.rfile.read(length))
				index = int(request["index"])
				start = float(request["start"])
				end = float(request["end"])
				recording = payload["recordings"][index]
				if not start < end:
					raise ValueError("The start time must be before the end time")
				source = data_root / recording["kind"] / recording["label"].split("/", 1)[1]
				if self.path == "/api/save-flattened":
					destination = save_flattened(source, recording, start, end, float(request["angle"]), data_root)
				else:
					destination = save_shortened(source, start, end)
			except (KeyError, IndexError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
				self.send_json({"error": str(error)}, 400)
				return
			self.send_json({"path": str(destination)})

		def log_message(self, format: str, *args: Any) -> None:
			return

	return Handler


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "curated")
	parser.add_argument("--port", type=int, default=8766)
	parser.add_argument("--no-browser", action="store_true")
	args = parser.parse_args()
	recordings = load_recordings(args.data_root)
	if not recordings:
		parser.error(f"No curated Accelerometer.csv files found under {args.data_root}")
	payload = {"recordings": recordings, "counts": {kind: sum(item["kind"] == kind for item in recordings) for kind in ("backhand", "flick")}}
	server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(payload, args.data_root))
	url = f"http://127.0.0.1:{args.port}/"
	print(f"Viewing {len(recordings)} curated throws at {url}")
	if not args.no_browser:
		threading.Timer(0.25, lambda: webbrowser.open(url)).start()
	try:
		server.serve_forever()
	except KeyboardInterrupt:
		print("\nStopped.")
	finally:
		server.server_close()


if __name__ == "__main__":
	main()
