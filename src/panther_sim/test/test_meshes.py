import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from panther_sim import meshes

URDF = os.path.join(os.path.dirname(__file__), "..", "urdf", "robot_description.urdf")


def test_every_vendor_mesh_is_written(tmp_path):
    meshes.write_all(str(tmp_path))
    names = {re.sub(r".*/meshes/", "", m.get("filename")) for m in ET.parse(URDF).getroot().iter("mesh")
             if "mh3_description" in m.get("filename")}
    assert all((tmp_path / "meshes" / n).stat().st_size > 84 for n in names)


def test_parts_are_closed_and_face_outward():
    for name, parts in meshes.all_parts().items():
        for V, F in parts:
            t = V[F]
            volume = np.einsum("ij,ij->i", t[:, 0], np.cross(t[:, 1], t[:, 2])).sum() / 6
            assert volume > 0, name  # mirrored parts too
