"""CI: the numpy FK used for the catching pose must match panther_examples' PyKDL Chain (cartesian_follow).

  PYTHONPATH=src/panther_examples python3 fk_crosscheck.py URDF CATCH_POSE_JSON
"""

import json
import sys

import numpy as np
from panther_examples.cartesian_follow import Chain

from frisbee_interception.core.kinematics import ArmChain

urdf, pose = open(sys.argv[1]).read(), json.load(open(sys.argv[2]))
rng = np.random.default_rng(0)
for arm, tip in (("left", "link_lf"), ("right", "link_rf")):
    kdl_chain, mine = Chain(urdf, "body_link", tip), ArmChain(urdf, "body_link", tip)
    assert kdl_chain.names == mine.names, (kdl_chain.names, mine.names)
    worst_p = worst_r = 0.0
    for q in [np.array(pose["nominal"][arm])] + list(rng.uniform(mine.lo, mine.hi, (50, mine.n))):
        f, T = kdl_chain.fk(q), mine.fk(q)
        worst_p = max(worst_p, float(np.linalg.norm(np.array([f.p[0], f.p[1], f.p[2]]) - T[:3, 3])))
        Rk = np.array([[f.M[i, j] for j in range(3)] for i in range(3)])
        worst_r = max(worst_r, float(np.abs(Rk - T[:3, :3]).max()))
    print(f"{arm}: FK numpy vs PyKDL max position difference {worst_p:.2e} m, rotation {worst_r:.2e}")
    assert worst_p < 1e-6 and worst_r < 1e-6
print("FK cross-check OK")
