"""Create the CI inputs: a synthetic throw video (+ ground truth) and two configs (image mode, metric mode).

  python3 make_ci_inputs.py OUT_DIR PACKAGE_CONFIG_YAML
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

from frisbee_interception import synthetic

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
base = yaml.safe_load(Path(sys.argv[2]).read_text())

rng = np.random.default_rng(1)
bg = cv2.resize(rng.integers(40, 160, (32, 18, 3), dtype=np.uint8), (720, 1280), interpolation=cv2.INTER_NEAREST)
truth = synthetic.render(bg, str(out / "throw.mp4"), str(out / "throw_truth.json"))
catch = truth["catch"]["1.0"]

# Image mode: only catch_diameter_px is calibrated (the apparent size at the catching plane). Disarmed.
img = yaml.safe_load(yaml.safe_dump(base))
img["frisbee_detector"]["ros__parameters"]["detection"]["method"] = "background"
img["frisbee_interceptor"]["ros__parameters"]["prediction"]["catch_diameter_px"] = catch["diameter_px"]
(out / "cfg_image.yaml").write_text(yaml.safe_dump(img))

# Metric mode: intrinsics + catching plane z = 1 m in front of the camera. The CI publishes an identity
# static TF body_link -> phone_camera_optical_frame, so catch points come out in camera coordinates.
met = yaml.safe_load(yaml.safe_dump(img))
ip = met["frisbee_interceptor"]["ros__parameters"]
ip["prediction"]["catch_diameter_px"] = 0.0
ip["camera"] = {"fx": truth["fx"], "fy": truth["fy"], "cx": truth["cx"], "cy": truth["cy"]}
ip["catch_plane"] = {"point": [0.0, 0.0, 1.0], "normal": [0.0, 0.0, 1.0]}
ip["safety"]["workspace_min"] = [-1.0, -1.0, 0.5]
ip["safety"]["workspace_max"] = [1.0, 1.0, 1.5]
(out / "cfg_metric.yaml").write_text(yaml.safe_dump(met))

print(json.dumps(truth["catch"], indent=1))
