#!/usr/bin/env python3
"""Look at the shoulder-relative arm recordings in this folder and pick a start and a release timestamp.

3D view: shoulder at the origin, elbow and wrist paths (faint), the part between start and release (bold), and the
arm drawn at start (green) and at release (red). Below: wrist speed with the two timestamps marked.
Set the timestamps with the sliders, or click the speed plot (left = start, right = release).
"Save" writes the pair to selections.csv here (one row per recording, replaced on re-save).

  python3 visualize.py
"""
import csv
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider

HERE = os.path.dirname(os.path.abspath(__file__))
SEL = os.path.join(HERE, "selections.csv")
FILES = sorted(os.path.basename(f) for f in glob.glob(os.path.join(HERE, "trajectory_*.csv")))


def load_selections():
    if not os.path.exists(SEL):
        return {}
    with open(SEL) as f:
        return {r["file"]: (float(r["start"]), float(r["release"])) for r in csv.DictReader(f)}


fig = plt.figure(figsize=(13, 8))
ax3 = fig.add_axes([0.22, 0.38, 0.55, 0.6], projection="3d")
axv = fig.add_axes([0.08, 0.17, 0.88, 0.17])
s_start = Slider(fig.add_axes([0.08, 0.08, 0.8, 0.03]), "start", 0, 1, valinit=0)
s_rel = Slider(fig.add_axes([0.08, 0.04, 0.8, 0.03]), "release", 0, 1, valinit=1)
radio = RadioButtons(fig.add_axes([0.01, 0.75, 0.2, 0.2]), FILES)
save_btn = Button(fig.add_axes([0.85, 0.9, 0.1, 0.05]), "Save")
st = {}  # current recording
for s in (s_start, s_rel):
    s.vline.set_visible(False)  # valinit marker would be wrong after switching files


def load(name):
    d = np.genfromtxt(os.path.join(HERE, name), delimiter=",", names=True)
    t = d["timestamp"]
    el = np.column_stack([d["elbow_rel_x"], d["elbow_rel_y"], d["elbow_rel_z"]])
    wr = np.column_stack([d["wrist_rel_x"], d["wrist_rel_y"], d["wrist_rel_z"]])
    speed = np.linalg.norm(np.gradient(wr, t, axis=0), axis=1)
    st.update(name=name, t=t, el=el, wr=wr, speed=speed)
    start, rel = load_selections().get(name, (t[0], t[np.argmax(speed)]))  # default release: peak wrist speed
    for s, v in ((s_start, start), (s_rel, rel)):
        s.valmin, s.valmax, s.valstep = t[0], t[-1], t  # snap to recorded timestamps
        s.ax.set_xlim(t[0], t[-1])
        s.set_val(v)  # triggers draw()
    draw()


def idx(v):
    return int(np.argmin(abs(st["t"] - v)))


def draw(_=None):
    if not st:
        return
    t, el, wr, speed = st["t"], st["el"], st["wr"], st["speed"]
    i0, i1 = idx(s_start.val), idx(s_rel.val)
    lo, hi = min(i0, i1), max(i0, i1)
    ax3.cla()
    ax3.plot(*wr.T, color="tab:blue", alpha=0.25)
    ax3.plot(*el.T, color="tab:orange", alpha=0.25)
    ax3.plot(*wr[lo:hi + 1].T, color="tab:blue", lw=2.5, label="wrist")
    ax3.plot(*el[lo:hi + 1].T, color="tab:orange", lw=2.5, label="elbow")
    for i, c, lab in ((i0, "green", "start"), (i1, "red", "release")):
        arm = np.vstack([np.zeros(3), el[i], wr[i]])
        ax3.plot(*arm.T, "-o", color=c, lw=3, label=f"{lab} {t[i]:.3f} s")
    ax3.scatter(0, 0, 0, color="k", s=40, label="shoulder")
    pts = np.vstack([el, wr, np.zeros((1, 3))])
    mid, r = (pts.max(0) + pts.min(0)) / 2, (pts.max(0) - pts.min(0)).max() / 2
    ax3.set(xlim=(mid[0] - r, mid[0] + r), ylim=(mid[1] - r, mid[1] + r), zlim=(mid[2] - r, mid[2] + r),
            xlabel="x (m)", ylabel="y (m)", zlabel="z (m)")
    ax3.set_box_aspect((1, 1, 1))
    ax3.legend(loc="upper left", fontsize=8)
    axv.cla()
    axv.plot(t, speed, color="tab:blue")
    axv.axvspan(t[lo], t[hi], color="gold", alpha=0.3)
    axv.axvline(t[i0], color="green")
    axv.axvline(t[i1], color="red")
    axv.set(xlim=(t[0], t[-1]), ylabel="wrist speed (m/s)")
    axv.set_title("left click = start, right click = release", fontsize=8)
    fig.canvas.draw_idle()


def on_click(e):
    if e.inaxes is axv and e.xdata is not None and e.button in (1, 3):
        (s_start if e.button == 1 else s_rel).set_val(st["t"][idx(e.xdata)])


def save(_):
    sel = load_selections()
    sel[st["name"]] = (s_start.val, s_rel.val)
    with open(SEL, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "start", "release"])
        w.writerows([k, f"{a:.6f}", f"{b:.6f}"] for k, (a, b) in sorted(sel.items()))
    print(f"{st['name']}: start {s_start.val:.6f} s, release {s_rel.val:.6f} s -> {SEL}")


s_start.on_changed(draw)
s_rel.on_changed(draw)
radio.on_clicked(load)
save_btn.on_clicked(save)
fig.canvas.mpl_connect("button_press_event", on_click)
load(FILES[0])
plt.show()
