"""Run detector -> tracker -> predictor on a video file without ROS (Phase 2-4 checks, CI, GCP batch jobs).

  python -m frisbee_interception.offline VIDEO [--config config/frisbee.yaml] [--out debug.mp4] [--json out.json]

Timestamps are frame_index / fps from the file. Writes an annotated video and a per-frame JSON log.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from frisbee_interception.core.catch_trigger import CatchConfig, CatchScheduler
from frisbee_interception.core.detection import DetectorConfig, FrisbeeDetector
from frisbee_interception.core.prediction import InterceptPredictor, PredictorConfig
from frisbee_interception.core.tracking import FrisbeeTracker, Measurement, TrackerConfig
from frisbee_interception.viz import draw_overlay


def load_yaml_section(path: str | None, node: str) -> dict:
    if not path:
        return {}
    import yaml

    data = yaml.safe_load(Path(path).read_text()) or {}
    params = (data.get(node) or {}).get("ros__parameters", {}) or {}

    def flat(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    return flat(params)


def run(video: str, cfg_path: str | None, out: str | None, log_path: str | None, rotate: str = "none") -> dict:
    det_p = load_yaml_section(cfg_path, "frisbee_detector")
    trk_p = load_yaml_section(cfg_path, "frisbee_tracker")
    prd_p = load_yaml_section(cfg_path, "frisbee_interceptor")
    det = FrisbeeDetector(DetectorConfig.from_dict({k[len("detection."):]: v for k, v in det_p.items() if k.startswith("detection.")}))
    trk = FrisbeeTracker(TrackerConfig.from_dict({k[len("tracking."):]: v for k, v in trk_p.items() if k.startswith("tracking.")}))
    pcfg = PredictorConfig.from_dict({k[len("prediction."):]: v for k, v in prd_p.items() if k.startswith("prediction.")})
    pred = InterceptPredictor(pcfg)
    # Catch trigger (dry run): simulated only when its calibration is set in the config.
    ccfg = CatchConfig.from_dict({k[len("catch."):]: v for k, v in load_yaml_section(cfg_path, "frisbee_catch_trigger").items()
                                  if k.startswith("catch.")})
    catch = CatchScheduler(ccfg) if not ccfg.calibration_problems() else None
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    log, i = [], 0
    stats = {"frames": 0, "detections": 0, "confirmed_frames": 0, "valid_predictions": 0, "reasons": {}, "catch_fired": []}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if rotate == "cw":
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        t = i / fps
        dets = det.detect(frame)
        state = trk.step(t, [Measurement(t, d.u, d.v, d.diameter_px, d.confidence) for d in dets])
        p = None
        if trk.primary is not None and trk.state in ("confirmed", "coasting"):
            h, w = frame.shape[:2]
            p = pred.at_time(pred.predict_image(trk.history, None, (w / 2.0, h / 2.0)), t)
            stats["confirmed_frames"] += 1
            if p.valid:
                stats["valid_predictions"] += 1
            else:
                stats["reasons"][p.reason.split(" (")[0]] = stats["reasons"].get(p.reason.split(" (")[0], 0) + 1
        if catch is not None:
            ok_p = p is not None and p.valid and p.pixel is not None
            catch.update(t, ok_p, p.intercept_time if ok_p else float("nan"), p.pixel if ok_p else None,
                         p.confidence if ok_p else 0.0)
            fired = catch.due(t)
            if fired is not None:
                stats["catch_fired"].append({"t": round(t, 4), "close_at": round(fired.close_at, 4),
                                             "intercept_time": round(fired.intercept_time, 4)})
        stats["frames"] += 1
        stats["detections"] += len(dets)
        entry = {"frame": i, "t": round(t, 4), "n_det": len(dets), "state": state}
        est = trk.estimate()
        if est:
            entry.update({k: round(est[k], 2) for k in ("u", "v", "d", "du", "dv", "dd")})
        if p is not None:
            entry.update({"valid": p.valid, "reason": p.reason, "tti": None if math.isnan(p.time_to_intercept) else round(p.time_to_intercept, 3),
                          "pixel": None if p.pixel is None else [round(p.pixel[0], 1), round(p.pixel[1], 1)],
                          "confidence": p.confidence, "pixel_sigma": None if p.pixel_sigma is None or math.isnan(p.pixel_sigma) else round(p.pixel_sigma, 1)})
        log.append(entry)
        if out:
            img = draw_overlay(frame, dets, trk, p)
            if writer is None:
                writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0]))
            writer.write(img)
        i += 1
    if writer is not None:
        writer.release()
    if log_path:
        Path(log_path).write_text(json.dumps({"video": video, "stats": stats, "frames": log}, indent=1))
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--rotate", choices=["none", "cw"], default="none")
    a = ap.parse_args()
    print(json.dumps(run(a.video, a.config, a.out, a.json, a.rotate), indent=1))


if __name__ == "__main__":
    main()
