#!/usr/bin/env python3
"""Review and trim sensor-logger recordings in a small local web app."""

from __future__ import annotations

import argparse
import csv
import io
import json
import threading
import webbrowser
import zipfile
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


CSV_NAMES = (
	"Accelerometer.csv", "AccelerometerUncalibrated.csv", "Annotation.csv",
	"Gyroscope.csv", "GyroscopeUncalibrated.csv", "Metadata.csv", "Orientation.csv",
)
TIME_COLUMN = "seconds_elapsed"


@dataclass
class Recording:
	kind: str
	name: str
	source: Path
	is_zip: bool

	@property
	def label(self) -> str:
		return f"{self.kind}/{self.name}"


def discover_recordings(data_root: Path) -> list[Recording]:
	recordings: list[Recording] = []
	for kind_path in sorted(data_root.iterdir()):
		if not kind_path.is_dir():
			continue
		for source in sorted(kind_path.iterdir()):
			if source.is_file() and source.suffix.lower() == ".zip":
				recordings.append(Recording(kind_path.name, source.stem, source, True))
			elif source.is_dir() and any(source.glob("*.csv")):
				recordings.append(Recording(kind_path.name, source.name, source, False))
	return recordings


def read_sources(recording: Recording) -> dict[str, str]:
	if recording.is_zip:
		with zipfile.ZipFile(recording.source) as archive:
			by_name = {Path(name).name: name for name in archive.namelist()}
			return {name: archive.read(by_name[name]).decode("utf-8-sig") for name in CSV_NAMES if name in by_name}
	return {name: (recording.source / name).read_text(encoding="utf-8-sig") for name in CSV_NAMES if (recording.source / name).exists()}


def parse_csv(text: str) -> tuple[list[str], list[dict[str, str]]]:
	reader = csv.DictReader(io.StringIO(text))
	return reader.fieldnames or [], list(reader)


def numeric_series(text: str) -> dict[str, Any]:
	fields, rows = parse_csv(text)
	if TIME_COLUMN not in fields:
		return {"fields": fields, "time": [], "magnitude": []}
	time: list[float] = []
	magnitude: list[float] = []
	for row in rows:
		try:
			time.append(float(row[TIME_COLUMN]))
			magnitude.append(sum(float(row[name]) ** 2 for name in ("x", "y", "z")) ** 0.5)
		except (KeyError, TypeError, ValueError):
			time.append(float("nan"))
			magnitude.append(float("nan"))
	return {"fields": fields, "time": time, "magnitude": magnitude}


def integrated_position(text: str) -> dict[str, list[float]]:
	"""Estimate relative position by removing the initial stationary acceleration."""
	fields, rows = parse_csv(text)
	if TIME_COLUMN not in fields or not all(axis in fields for axis in ("x", "y", "z")):
		return {"x": [], "y": [], "z": []}
	try:
		times = [float(row[TIME_COLUMN]) for row in rows]
		acceleration = [tuple(float(row[axis]) for axis in ("x", "y", "z")) for row in rows]
	except (KeyError, TypeError, ValueError):
		return {"x": [], "y": [], "z": []}
	if not acceleration:
		return {"x": [], "y": [], "z": []}
	baseline_count = min(20, len(acceleration))
	baseline = tuple(sum(sample[axis] for sample in acceleration[:baseline_count]) / baseline_count for axis in range(3))
	velocity = [0.0, 0.0, 0.0]
	position = [0.0, 0.0, 0.0]
	positions = {"x": [0.0], "y": [0.0], "z": [0.0]}
	for previous, current, previous_acceleration, current_acceleration in zip(times, times[1:], acceleration, acceleration[1:]):
		delta = max(0.0, current - previous)
		for axis in range(3):
			previous_corrected = previous_acceleration[axis] - baseline[axis]
			current_corrected = current_acceleration[axis] - baseline[axis]
			position[axis] += velocity[axis] * delta + 0.25 * (previous_corrected + current_corrected) * delta * delta
			velocity[axis] += 0.5 * (previous_corrected + current_corrected) * delta
			positions["xyz"[axis]].append(position[axis])
	return positions


def trim_csv(text: str, start: float, end: float) -> str:
	fields, rows = parse_csv(text)
	if TIME_COLUMN not in fields:
		return text
	kept = []
	for row in rows:
		try:
			if start <= float(row[TIME_COLUMN]) <= end:
				kept.append(row)
		except (KeyError, TypeError, ValueError):
			pass
	output = io.StringIO(newline="")
	writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
	writer.writeheader()
	writer.writerows(kept)
	return output.getvalue()


def write_recording(recording: Recording, sources: dict[str, str], output_root: Path, start: float, end: float) -> Path:
	destination = output_root / recording.kind / recording.name
	destination.mkdir(parents=True, exist_ok=True)
	for name, text in sources.items():
		(destination / name).write_text(trim_csv(text, start, end), encoding="utf-8")
	return destination


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Throw curator</title><style>
:root{color-scheme:dark;--bg:#101418;--panel:#192126;--line:#34434a;--text:#e9f0ed;--muted:#91a19e;--accent:#f2b84b;--green:#73d0a6}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 75% -10%,#2c3936 0,#101418 46%);color:var(--text);font:15px system-ui,sans-serif}main{max-width:1180px;margin:0 auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:end;gap:20px;margin-bottom:22px}.eyebrow{color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.13em;text-transform:uppercase}.title{font:700 34px Georgia,serif;margin:5px 0 0}.counter{color:var(--muted);text-align:right}.current-time{color:var(--accent);font-variant-numeric:tabular-nums;margin-top:4px}.panel{background:rgba(25,33,38,.9);border:1px solid var(--line);border-radius:8px;padding:20px;box-shadow:0 12px 35px #0003}.toolbar{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-bottom:18px}.select{background:#0d1215;border:1px solid var(--line);border-radius:5px;color:var(--text);padding:10px;min-width:280px;font:inherit}.button{border:0;border-radius:5px;padding:10px 15px;background:#314047;color:var(--text);font-weight:700;cursor:pointer}.button:hover{background:#43565e}.primary{background:var(--accent);color:#17130a}.primary:hover{background:#ffd06d}.plot{height:360px;background:#0d1215;border-radius:5px;border:1px solid #29363b}.position-plot{height:240px;margin-top:14px}.plot canvas{width:100%;height:100%}.legend{display:flex;gap:18px;color:var(--muted);font-size:13px;margin:10px 2px 18px;flex-wrap:wrap}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.controls{display:grid;grid-template-columns:130px 1fr 130px;align-items:center;gap:12px}.number{background:#0d1215;border:1px solid var(--line);color:var(--text);padding:9px;border-radius:4px;width:100%;font-variant-numeric:tabular-nums}.range{grid-column:2;position:relative;height:34px}.range input{position:absolute;inset:0;width:100%;pointer-events:none;appearance:none;background:none}.range input::-webkit-slider-thumb{pointer-events:auto;appearance:none;width:18px;height:18px;border:3px solid var(--accent);border-radius:50%;background:#101418;cursor:grab}.range input::-moz-range-thumb{pointer-events:auto;width:14px;height:14px;border:3px solid var(--accent);border-radius:50%;background:#101418;cursor:grab}.track{position:absolute;left:0;right:0;top:15px;height:4px;border-radius:5px;background:#3a4b50;pointer-events:none}.selection{position:absolute;top:15px;height:4px;background:var(--accent);pointer-events:none}.status{min-height:22px;color:var(--green);margin-top:16px}.help{color:var(--muted);font-size:13px;margin-top:8px}.footer{display:flex;justify-content:space-between;align-items:center;margin-top:18px;gap:12px}.footer .button{min-width:110px}@media(max-width:700px){main{padding:18px 12px}.title{font-size:28px}.top{align-items:start}.controls{grid-template-columns:1fr 1fr}.range{grid-column:1/-1;grid-row:1}.controls .number:first-child{grid-column:1;grid-row:2}.controls .number:last-child{grid-column:2;grid-row:2}.plot{height:290px}.position-plot{height:200px}.select{min-width:0;flex:1}}
</style></head><body><main><header class="top"><div><div class="eyebrow">IMU throw curation</div><h1 class="title">Find the throw window</h1></div><div class="counter"><div id="counter"></div><div class="current-time" id="currentTime">Current time: -- s</div></div></header><section class="panel"><div class="toolbar"><select class="select" id="recording"></select><button class="button" id="previous">Previous</button><button class="button" id="next">Next</button></div><div class="plot"><canvas id="plot"></canvas></div><div class="legend"><span><i class="dot" style="background:#73d0a6"></i>acceleration magnitude</span><span><i class="dot" style="background:#f2b84b"></i>gyroscope magnitude</span><span>shaded area = saved interval</span></div><div class="plot position-plot"><canvas id="positionPlot"></canvas></div><div class="legend"><span><i class="dot" style="background:#ef8f75"></i>X position</span><span><i class="dot" style="background:#8fc7f2"></i>Y position</span><span><i class="dot" style="background:#c59cf2"></i>Z position</span><span>relative position from integrated acceleration</span></div><div class="controls"><input class="number" id="startNumber" type="number" step="0.001"><div class="range"><div class="track"></div><div class="selection" id="selection"></div><input id="start" type="range"><input id="end" type="range"></div><input class="number" id="endNumber" type="number" step="0.001"></div><div class="help">Drag the two handles until the motion begins and ends. Values are seconds elapsed in the original logger recording.</div><div class="status" id="status"></div><div class="footer"><button class="button" id="saveNext">Save &amp; next</button><button class="button primary" id="save">Save selected interval</button></div></section></main><script>
let recordings=[],index=0,current=null;const $=id=>document.getElementById(id);async function loadList(){recordings=await(await fetch('/api/recordings')).json();$('recording').innerHTML=recordings.map((r,i)=>`<option value="${i}">${r.label}</option>`).join('');await load(index)}async function load(i){index=Math.max(0,Math.min(recordings.length-1,i));$('recording').value=index;current=await(await fetch('/api/recording?index='+index)).json();let t=current.time.filter(Number.isFinite),lo=Math.min(...t),hi=Math.max(...t);for(const id of ['start','end']){$(id).min=lo;$(''+id).max=hi;$(''+id).step=(hi-lo)/1000||.001}$('start').value=lo;$('end').value=hi;$('startNumber').value=lo.toFixed(3);$('endNumber').value=hi.toFixed(3);$('counter').textContent=`${index+1} / ${recordings.length} · ${current.label}`;$('status').textContent='';draw()}function setValue(which,value){let lo=+$('start').min,hi=+$('start').max;value=Math.max(lo,Math.min(hi,+value));if(which==='start')value=Math.min(value,+$('end').value);else value=Math.max(value,+$('start').value);$(''+which).value=value;$(''+which+'Number').value=value.toFixed(3);draw()}function draw(){if(!current)return;let canvas=$('plot'),rect=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=rect.width*d;canvas.height=rect.height*d;let c=canvas.getContext('2d');c.scale(d,d);let w=rect.width,h=rect.height,p={l:46,r:18,t:18,b:30},times=current.time,all=[...current.accelerometer,...current.gyro].filter(Number.isFinite),max=Math.max(...all,1),lo=+$('start').min,hi=+$('start').max,s=+$('start').value,e=+$('end').value,x=t=>p.l+(t-lo)/(hi-lo||1)*(w-p.l-p.r),y=v=>h-p.b-v/max*(h-p.t-p.b);c.strokeStyle='#29363b';c.lineWidth=1;for(let j=0;j<5;j++){let yy=p.t+j*(h-p.t-p.b)/4;c.beginPath();c.moveTo(p.l,yy);c.lineTo(w-p.r,yy);c.stroke()}c.fillStyle='#f2b84b22';c.fillRect(x(s),p.t,x(e)-x(s),h-p.t-p.b);for(const [data,color] of [[current.accelerometer,'#73d0a6'],[current.gyro,'#f2b84b']]){c.strokeStyle=color;c.lineWidth=1.5;c.beginPath();data.forEach((v,i)=>{if(!Number.isFinite(times[i])||!Number.isFinite(v))return;(i?c.lineTo:c.moveTo).call(c,x(times[i]),y(v))});c.stroke()}c.fillStyle='#91a19e';c.font='12px system-ui';c.fillText(lo.toFixed(2)+'s',p.l,h-8);c.fillText(hi.toFixed(2)+'s',w-p.r-35,h-8);$('selection').style.left=((s-lo)/(hi-lo)*100)+'%';$('selection').style.width=((e-s)/(hi-lo)*100)+'%'}for(const id of ['start','end'])$(id).addEventListener('input',()=>setValue(id,$(id).value));$('startNumber').addEventListener('change',e=>setValue('start',e.target.value));$('endNumber').addEventListener('change',e=>setValue('end',e.target.value));$('recording').addEventListener('change',e=>load(+e.target.value));$('previous').onclick=()=>load(index-1);$('next').onclick=()=>load(index+1);window.onresize=draw;async function save(next){let body=JSON.stringify({index,start:+$('start').value,end:+$('end').value}),result=await(await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body})).json();$('status').textContent=`Saved ${result.rows} rows per timed file to ${result.path}`;if(next&&index<recordings.length-1)setTimeout(()=>load(index+1),350)}$('save').onclick=()=>save(false);$('saveNext').onclick=()=>save(true);loadList();
</script></body></html>'''


POSITION_SCRIPT = r"""
const originalDraw = draw;
function drawPosition() {
	const canvas = $('positionPlot');
	if (!current || !current.position || !canvas) return;
	const rect = canvas.getBoundingClientRect();
	const scale = devicePixelRatio || 1;
	canvas.width = rect.width * scale;
	canvas.height = rect.height * scale;
	const context = canvas.getContext('2d');
	context.scale(scale, scale);
	const width = rect.width;
	const height = rect.height;
	const padding = {left: 46, right: 18, top: 18, bottom: 30};
	const times = current.time;
	const values = current.position;
	const allValues = Object.values(values).flat().filter(Number.isFinite);
	const maximum = Math.max(...allValues.map(Math.abs), 0.001);
	const low = +$('start').min;
	const high = +$('start').max;
	const start = +$('start').value;
	const end = +$('end').value;
	const x = time => padding.left + (time - low) / (high - low || 1) * (width - padding.left - padding.right);
	const y = value => height / 2 - value / maximum * (height / 2 - padding.top);
	context.strokeStyle = '#29363b';
	context.lineWidth = 1;
	for (const value of [0, -maximum, maximum]) {
		const row = y(value);
		context.beginPath(); context.moveTo(padding.left, row); context.lineTo(width - padding.right, row); context.stroke();
	}
	context.fillStyle = '#f2b84b22';
	context.fillRect(x(start), padding.top, x(end) - x(start), height - padding.top - padding.bottom);
	for (const [axis, color] of [['x', '#ef8f75'], ['y', '#8fc7f2'], ['z', '#c59cf2']]) {
		context.strokeStyle = color; context.lineWidth = 1.5; context.beginPath();
		values[axis].forEach((value, i) => {
			if (!Number.isFinite(times[i]) || !Number.isFinite(value)) return;
			(i ? context.lineTo : context.moveTo).call(context, x(times[i]), y(value));
		});
		context.stroke();
	}
	context.fillStyle = '#91a19e'; context.font = '12px system-ui';
	context.fillText('relative position', padding.left, 14);
	context.fillText(low.toFixed(2) + 's', padding.left, height - 8);
	context.fillText(high.toFixed(2) + 's', width - padding.right - 35, height - 8);
}
draw = function() {
	originalDraw();
	$('currentTime').textContent = 'Current time: ' + (+$('start').value).toFixed(3) + ' s';
	drawPosition();
};
"""
HTML = HTML.replace("</script>", POSITION_SCRIPT + "</script>", 1)


def make_handler(recordings: list[Recording], output_root: Path):
	class Handler(BaseHTTPRequestHandler):
		def send_json(self, payload: Any, status: int = 200) -> None:
			data = json.dumps(payload).encode()
			self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)

		def do_GET(self) -> None:  # noqa: N802
			request = urlparse(self.path)
			if request.path == "/":
				data = HTML.encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
			if request.path == "/api/recordings":
				self.send_json([{"label": r.label} for r in recordings]); return
			if request.path == "/api/recording":
				index = int(parse_qs(request.query).get("index", [0])[0]); recording = recordings[index]; sources = read_sources(recording)
				base = numeric_series(sources.get("Accelerometer.csv", "")); gyro = numeric_series(sources.get("Gyroscope.csv", "")); position = integrated_position(sources.get("Accelerometer.csv", "")); self.send_json({"label": recording.label, "time": base.get("time", []), "accelerometer": base.get("magnitude", []), "gyro": gyro.get("magnitude", []), "position": position}); return
			self.send_error(404)

		def do_POST(self) -> None:  # noqa: N802
			if urlparse(self.path).path != "/api/save": self.send_error(404); return
			length = int(self.headers.get("Content-Length", 0)); request = json.loads(self.rfile.read(length)); recording = recordings[int(request["index"])]
			start, end = float(request["start"]), float(request["end"]); sources = read_sources(recording); destination = write_recording(recording, sources, output_root, start, end)
			timed_rows = sum(len(parse_csv(trim_csv(text, start, end))[1]) for name, text in sources.items() if name != "Metadata.csv"); self.send_json({"path": str(destination), "rows": timed_rows})

		def log_message(self, format: str, *args: Any) -> None:
			return
	return Handler


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--data-root", type=Path, default=Path(__file__).parent / "data"); parser.add_argument("--output", type=Path, default=Path(__file__).parent / "curated"); parser.add_argument("--port", type=int, default=8765); parser.add_argument("--no-browser", action="store_true"); args = parser.parse_args()
	recordings = discover_recordings(args.data_root)
	if not recordings: parser.error(f"No recording directories or ZIP files found under {args.data_root}")
	server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(recordings, args.output)); url = f"http://127.0.0.1:{args.port}/"; print(f"Curating {len(recordings)} recordings at {url}"); print(f"Trimmed recordings will be written to {args.output}")
	if not args.no_browser: threading.Timer(0.25, lambda: webbrowser.open(url)).start()
	try: server.serve_forever()
	except KeyboardInterrupt: print("\nStopped.")
	finally: server.server_close()


if __name__ == "__main__": main()
