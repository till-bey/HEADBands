"""End-to-end check (detector -> tracker -> image-mode predictor) on rendered throws with exact ground truth."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from frisbee_interception import offline, synthetic


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_synthetic_throw_prediction(tmp_path, seed):
    rng = np.random.default_rng(seed)
    # Static textured background (a fixed camera), 720x1280 portrait like the phone recordings.
    bg = cv2.resize(rng.integers(40, 160, (32, 18, 3), dtype=np.uint8), (720, 1280), interpolation=cv2.INTER_NEAREST)
    video, truth_path = str(tmp_path / "throw.mp4"), str(tmp_path / "truth.json")
    truth = synthetic.render(bg, video, truth_path, seed=seed)
    catch = truth["catch"]["1.0"]

    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "frisbee.yaml").read_text())
    cfg["frisbee_detector"]["ros__parameters"]["detection"]["method"] = "background"
    cfg["frisbee_interceptor"]["ros__parameters"]["prediction"]["catch_diameter_px"] = catch["diameter_px"]
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    log_path = tmp_path / "log.json"
    stats = offline.run(video, str(cfg_path), None, str(log_path))
    assert stats["valid_predictions"] >= 5

    frames = json.loads(log_path.read_text())["frames"]
    valid = [f for f in frames if f.get("valid")]
    # The first valid prediction must come well before contact ...
    assert catch["t_video"] - valid[0]["t"] > 0.3
    for f in valid:
        t_err = abs(f["t"] + f["tti"] - catch["t_video"])
        px_err = np.hypot(f["pixel"][0] - catch["pixel"][0], f["pixel"][1] - catch["pixel"][1])
        # ... errors scale with the time still to go: coarse 0.5 s out, tight in the last 0.3 s
        # (20 px = 4.5 cm at 1 m with this camera).
        assert t_err < max(0.05, 0.4 * f["tti"]), f
        assert px_err < (20 if f["tti"] <= 0.3 else 50), f
