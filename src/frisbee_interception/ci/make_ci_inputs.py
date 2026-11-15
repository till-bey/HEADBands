"""Create the CI inputs: a synthetic throw video (+ ground truth) and two configs (image mode, metric mode).

  python3 make_ci_inputs.py OUT_DIR PACKAGE_CONFIG_YAML CATCH_POSE_JSON
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
pose_file = sys.argv[3]

rng = np.random.default_rng(1)
bg = cv2.resize(rng.integers(40, 160, (32, 18, 3), dtype=np.uint8), (720, 1280), interpolation=cv2.INTER_NEAREST)
truth = synthetic.render(bg, str(out / "throw.mp4"), str(out / "throw_truth.json"))
catch = truth["catch"]["1.0"]

# Image mode: only catch_diameter_px is calibrated (the apparent size at the catching plane). Nothing may move:
# no workspace (interceptor arming refused), no catch calibration (catch arming refused), no corrections.
img = yaml.safe_load(yaml.safe_dump(base))
img["frisbee_detector"]["ros__parameters"]["detection"]["method"] = "background"
img["frisbee_interceptor"]["ros__parameters"]["prediction"]["catch_diameter_px"] = catch["diameter_px"]
img["frisbee_interceptor"]["ros__parameters"]["safety"]["workspace_min"] = [0.0]
img["frisbee_interceptor"]["ros__parameters"]["safety"]["workspace_max"] = [0.0]
img["frisbee_catch_trigger"]["ros__parameters"]["corrections"]["enabled"] = False
(out / "cfg_image.yaml").write_text(yaml.safe_dump(img))

# Metric mode on the robot geometry: the camera optical frame is rotated into body_link (x forward, z up) and placed
# so the throw crosses the hands' plane x = 0.45 near the chest. run_ci.sh publishes this static TF.
R_BC = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])  # optical -> body (rpy -pi/2, 0, -pi/2)
T_BC = np.array([-0.55, 0.219, -0.348])
truth_body = (R_BC @ np.array(catch["point_cam"]) + T_BC).tolist()
met = yaml.safe_load(yaml.safe_dump(img))
ip = met["frisbee_interceptor"]["ros__parameters"]
ip["prediction"]["catch_diameter_px"] = 0.0
ip["camera"] = {"fx": truth["fx"], "fy": truth["fy"], "cx": truth["cx"], "cy": truth["cy"]}
ip["catch_plane"] = {"point": [0.45, 0.0, -0.20], "normal": [1.0, 0.0, 0.0]}
ip["safety"]["workspace_min"] = [0.40, -0.25, -0.35]
ip["safety"]["workspace_max"] = [0.50, 0.25, -0.05]
# Catch: 0.1 s gripper latency, catch window +-80 px around where the throw crosses; hand corrections on.
u, v = (float(x) for x in catch["pixel"])
ct = met["frisbee_catch_trigger"]["ros__parameters"]
ct["catch"].update(gripper_latency_s=0.1, catch_window_px=[u - 80, v - 80, u + 80, v + 80])
ct["corrections"]["enabled"] = True
ct["corrections"]["pose_file"] = pose_file
(out / "cfg_metric.yaml").write_text(yaml.safe_dump(met))
(out / "truth_body.json").write_text(json.dumps({"point_body": truth_body}))

print(json.dumps({"catch": truth["catch"], "point_body": truth_body}, indent=1))
