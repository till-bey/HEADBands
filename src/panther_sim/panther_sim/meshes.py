"""Stylized stand-ins for the vendor mh3_description meshes (not public): rounded shells sized from the URDF joints.

write_all(dir) writes <dir>/meshes/<part>/<link>.STL under the vendor's names, so the launch files point the
URDF's mesh paths at <dir> the same way as at a real copy. Looks only: not the vendor shapes, not for collision.
"""
import os
import struct

import numpy as np

N = 48  # segments around every round part


def _frame(axis):
    z = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.cross([1, 0, 0] if abs(z[0]) < 0.9 else [0, 1, 0], z)
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def _closed(P):
    """Triangles over a (rows, N, 3) grid that wraps around; rows start and end on the axis, so it is closed."""
    rows, n = P.shape[:2]
    i, j = np.meshgrid(np.arange(rows - 1), np.arange(n), indexing="ij")
    a, b, c, d = i * n + j, i * n + (j + 1) % n, (i + 1) * n + (j + 1) % n, (i + 1) * n + j
    V = P.reshape(-1, 3)
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)])
    t = V[F]
    if np.einsum("ij,ij->i", t[:, 0] - V.mean(0), np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])).sum() < 0:
        F = F[:, ::-1]  # outward normals, or RViz culls the faces we see
    return V, F


def revolve(profile, center=(0, 0, 0), axis=(0, 0, 1)):
    """Surface of revolution of (r, z) points, bottom (r=0) to top (r=0)."""
    r, z = np.asarray(profile, float).T
    t = np.linspace(0, 2 * np.pi, N, endpoint=False)
    local = np.stack([r[:, None] * np.cos(t), r[:, None] * np.sin(t), np.broadcast_to(z[:, None], (len(r), N))], -1)
    return _closed(local @ _frame(axis).T + center)


def _arc(cr, cz, rad, a0, a1, n=7):
    a = np.linspace(a0, a1, n)
    return list(zip(cr + rad * np.cos(a), cz + rad * np.sin(a)))


def cyl(center, axis, r, h, f=0.006):
    """Cylinder with filleted edges."""
    f = min(f, r / 2, h / 2)
    return revolve([(0, -h / 2)] + _arc(r - f, -h / 2 + f, f, -np.pi / 2, 0) + _arc(r - f, h / 2 - f, f, 0, np.pi / 2)
                   + [(0, h / 2)], center, axis)


def capsule(p0, p1, r0, r1=None):
    """Rounded tube from p0 to p1, radius r0 tapering to r1."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    return revolve(_arc(0, -L / 2, r0, -np.pi / 2, 0, 9) + _arc(0, L / 2, r1 or r0, 0, np.pi / 2, 9),
                   (p0 + p1) / 2, p1 - p0)


def rbox(center, size, e=0.3, R=np.eye(3)):
    """Rounded box (superellipsoid): e=1 is an ellipsoid, toward 0 the corners sharpen."""
    s = lambda w: np.sign(w) * np.abs(w) ** e
    u = np.linspace(-np.pi / 2, np.pi / 2, 25)[:, None]
    v = np.linspace(-np.pi, np.pi, N, endpoint=False)[None, :]
    P = np.stack([s(np.cos(u)) * s(np.cos(v)), s(np.cos(u)) * s(np.sin(v)), np.broadcast_to(s(np.sin(u)), (25, N))], -1)
    return _closed(P * np.asarray(size) / 2 @ R.T + center)


def bar(p0, p1, w, t):
    """Flat rounded bar from p0 to p1 in the y-z plane, w wide along x, t thick."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    z = (p1 - p0) / np.linalg.norm(p1 - p0)
    R = np.column_stack([[1, 0, 0], np.cross(z, [1, 0, 0]), z])
    return rbox((p0 + p1) / 2, (w, t, np.linalg.norm(p1 - p0) + t), 0.4, R)


def mirror_y(parts):
    return [(V * [1, -1, 1], F[:, ::-1]) for V, F in parts]


X, Y, Z = (1, 0, 0), (0, 1, 0), (0, 0, 1)


def arm(s):
    """Left arm for s=1; the right arm is its mirror in y."""
    return {
        0: [cyl((0, 0, 0), Z, 0.042, 0.075, 0.01), capsule((0, 0.02 * s, 0), (0, 0.105 * s, 0), 0.034)],
        1: [cyl((0, 0.035 * s, 0), Y, 0.04, 0.07, 0.01), capsule((0, 0.04 * s, 0), (0, 0.08 * s, 0), 0.03)],
        2: [cyl((0, 0, 0), X, 0.042, 0.085, 0.01), capsule((0, 0, -0.02), (0, 0, -0.115), 0.034)],
        3: [cyl((0, 0, -0.006), Z, 0.043, 0.02), capsule((0, 0, 0), (0, 0, -0.12), 0.038, 0.035)],
        4: [cyl((0, 0, 0), Y, 0.04, 0.085, 0.01), capsule((0, 0, -0.02), (0, 0, -0.12), 0.032)],
        5: [cyl((0, 0, -0.006), Z, 0.037, 0.02), capsule((0, 0, 0), (-0.004, 0, -0.115), 0.032, 0.028)],
        6: [cyl((0, 0, 0), X, 0.031, 0.07, 0.008)],
        7: [cyl((0, 0, 0), Y, 0.03, 0.064, 0.008), capsule((0, 0, -0.015), (0, 0, -0.045), 0.025),
            cyl((0, 0, -0.048), Z, 0.03, 0.006, 0.002)],
    }


def gripper():
    """One gripper's links by vendor suffix; finger x0..x3 goes to +y, x1..x13 is its mirror."""
    finger = [[bar((0, 0, 0), (0, 0.0078, 0.0162), 0.018, 0.008)],
              [bar((0, 0, 0), (0, 0.0574, -0.031), 0.02, 0.011)],
              [bar((0, 0, 0), (0, -0.0191, -0.0515), 0.02, 0.012)],
              [rbox((0, -0.0015, -0.009), (0.02, 0.008, 0.024), 0.35)]]
    links = {"base_link": [cyl((0.0055, -0.003, -0.004), Z, 0.032, 0.008, 0.002),
                           rbox((0.0055, -0.003, -0.045), (0.055, 0.075, 0.075), 0.35)],
             "": [rbox((0, 0, -0.003), (0.05, 0.07, 0.008), 0.3)]}
    for i, p in enumerate(finger):
        links[f"0{i}"], links[f"1{i}"] = p, mirror_y(p)
    return links


def all_parts():
    parts = {
        "mh3_movebase/base_link": [rbox((0.01, 0, 0.24), (0.6, 0.44, 0.12), 0.25),
                                   rbox((0.285, 0, 0.15), (0.06, 0.38, 0.09), 0.3),
                                   rbox((-0.255, 0, 0.15), (0.06, 0.38, 0.09), 0.3),
                                   rbox((-0.12, 0, 0.95), (0.11, 0.16, 1.3), 0.2),
                                   rbox((-0.15, 0.0055, 1.65), (0.08, 0.1, 0.08), 0.3)],  # where the Odin1 sits
        "mh3_movebase/body_link": [rbox((-0.01, 0, -0.05), (0.12, 0.14, 0.2), 0.35),
                                   rbox((-0.15, 0, -0.03), (0.2, 0.1, 0.07), 0.3),
                                   capsule((0.022, 0, 0.04), (0.022, 0, 0.1), 0.02)],
        "mh3_movebase/link_front_radar": [cyl((0, 0, -0.008), Z, 0.028, 0.022)],
        "mh3_movebase/link_back_radar": [cyl((0, 0, -0.008), Z, 0.028, 0.022)],
        "mh3_head/link_head0": [cyl((0, 0, 0), Y, 0.026, 0.06), capsule((0, 0, 0), (0, 0, 0.06), 0.02)],
        "mh3_head/link_head1": [rbox((0.012, 0, 0.075), (0.14, 0.18, 0.135), 0.35), cyl((0, 0, 0.006), Z, 0.035, 0.012)],
        "mh3_head/link_rgbd": [rbox((0, 0, 0), (0.024, 0.12, 0.03), 0.3)],
        "mh3_head/link_both_eyes": [cyl((0.004, y, 0), X, 0.016, 0.01, 0.003) for y in (0.032, -0.032)],
    }
    for w, leg in (("fl", -0.034), ("fr", 0.034), ("bl", -0.034), ("br", 0.034)):
        wy = -0.038 * np.sign(leg)  # the wheel sits on the far side of its fork leg
        parts[f"mh3_movebase/link_{w}_rev"] = [cyl((0, 0, -0.008), Z, 0.04, 0.016), rbox((0, leg, -0.055), (0.055, 0.012, 0.11))]
        parts[f"mh3_movebase/link_{w}_con"] = [cyl((0, wy, 0), Y, 0.075, 0.045, 0.018), cyl((0, wy, 0), Y, 0.045, 0.05),
                                               capsule((0, 0, 0), (0, wy, 0), 0.012)]
    for side, s in (("left", 1), ("right", -1)):
        for i, p in arm(s).items():
            parts[f"mh3_{side}_arm/link_{side[0]}a{i}"] = p
        for suffix, p in gripper().items():
            name = "base_link" if suffix == "base_link" else f"link_{side[0]}f{suffix}"
            parts[f"mh3_{side}_gripper3/{name}"] = p
    return parts


def write_stl(path, parts):
    T = np.concatenate([V[F] for V, F in parts]).astype(np.float32)
    n = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    rec = np.zeros(len(T), np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = n, T
    with open(path, "wb") as f:
        f.write(b"panther_sim stylized mesh".ljust(80, b" ") + struct.pack("<I", len(T)) + rec.tobytes())


def write_all(out):
    for name, parts in all_parts().items():
        path = os.path.join(out, "meshes", name + ".STL")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_stl(path, parts)
    return out
