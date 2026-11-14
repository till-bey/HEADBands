#!/usr/bin/env python3
"""Follow a fixed Cartesian trajectory with one Panther arm, through the robot's own trajectory controller.

IK runs here with PyKDL. The robot or sim only receives one FollowJointTrajectory goal on
<arm>_arm_joint_trajectory_position_controller, with the same controller switch PR #521 does:
dual_arm_forward_position_controller off for the move, back on afterwards.

  python3 cartesian_follow.py --arm left --path example_path.csv            # move
  python3 cartesian_follow.py --arm left --path example_path.csv --dry-run  # plan + check only
  python3 cartesian_follow.py --arm right --joints FILE.csv --dry-run       # joint angles given: no IK, same checks

CSV rows: t,x,y,z,roll,pitch,yaw (s, m, rad), the gripper frame (link_lf / link_rf) in body_link.
body_link is used because the lift moves it on the robot while the URDF has the lift fixed.
"""
import argparse
import random
import sys
import xml.etree.ElementTree as ET

import numpy as np
import PyKDL as kdl
import rclpy
from rclpy.action import ActionClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.srv import ListControllers, SwitchController
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectoryPoint

FORWARD = "dual_arm_forward_position_controller"
MAX_JOINT_VEL = 0.5  # rad/s, our choice; no vendor limit (PR #521 averages 0.35 rad/s on the robot)
BASE = "body_link"


def jnt(values):
    a = kdl.JntArray(len(values))
    for i, v in enumerate(values):
        a[i] = v
    return a


class Chain:
    """KDL chain from BASE to tip, built from the URDF (kdl_parser_py is not packaged for ROS 2)."""

    def __init__(self, urdf, base, tip):
        joints_by_child = {j.find("child").get("link"): j for j in ET.fromstring(urdf).findall("joint")}
        path, link = [], tip
        while link != base:
            path.append(joints_by_child[link])
            link = path[-1].find("parent").get("link")
        self.chain, self.names, lo, hi = kdl.Chain(), [], [], []
        for j in reversed(path):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            origin = kdl.Frame(kdl.Rotation.RPY(*rpy), kdl.Vector(*xyz))
            if j.get("type") == "revolute":
                axis = kdl.Vector(*[float(v) for v in j.find("axis").get("xyz").split()])
                joint = kdl.Joint(j.get("name"), origin.p, origin.M * axis, kdl.Joint.RotAxis)
                self.names.append(j.get("name"))
                lo.append(float(j.find("limit").get("lower")))
                hi.append(float(j.find("limit").get("upper")))
            elif j.get("type") == "fixed":
                joint = kdl.Joint(j.get("name"), kdl.Joint.Fixed)
            else:
                raise ValueError(f"{j.get('name')}: {j.get('type')} joints are not handled")
            # The segment's tip frame is the child link frame: origin, then the joint's rotation.
            self.chain.addSegment(kdl.Segment(j.find("child").get("link"), joint, origin))
        self.lo, self.hi = np.array(lo), np.array(hi)
        # KDL solvers hold C++ references to these: keep each one on self or Python frees it (segfault).
        self._fk = kdl.ChainFkSolverPos_recursive(self.chain)
        self._vel = kdl.ChainIkSolverVel_pinv(self.chain)
        self._ik = kdl.ChainIkSolverPos_NR_JL(self.chain, jnt(lo), jnt(hi), self._fk, self._vel, 500, 1e-6)

    def fk(self, q):
        f = kdl.Frame()
        self._fk.JntToCart(jnt(q), f)
        return f

    def ik(self, goal, seed):
        out = kdl.JntArray(len(self.names))
        if self._ik.CartToJnt(jnt(seed), goal, out) < 0:
            return None
        q = np.array([out[i] for i in range(len(self.names))])
        err = kdl.diff(self.fk(q), goal)  # NR_JL can stop at a limit short of the goal
        return q if err.vel.Norm() < 1e-4 and err.rot.Norm() < 1e-3 else None


def plan(chain, poses, ts, q_now, max_vel, hint=None):
    """Joint trajectory for the poses, or raise with the first limit it would break."""
    # First waypoint: solve from the current joints and random seeds.
    rng = random.Random(0)
    seeds = [q_now] + [np.array([rng.uniform(a, b) for a, b in zip(chain.lo, chain.hi)]) for _ in range(50)]
    firsts = sorted((q for q in (chain.ik(poses[0], s) for s in seeds) if q is not None),
                    key=lambda x: np.abs(x - q_now).max())
    # The file's start_joints (checked with margin by whoever made the path) goes first.
    if hint is not None and (q := chain.ik(poses[0], hint)) is not None:
        firsts.insert(0, q)
    if not firsts:
        raise RuntimeError("waypoint 0 is out of reach within the joint limits")
    # The arm is redundant: the first waypoint has several solutions and the rest of the path may
    # only work from some. Try them nearest-first; report the attempt that got furthest.
    best = (-1, None)
    for q in firsts:
        qs = [q]
        for i, pose in enumerate(poses[1:], 1):  # later waypoints: seed from the previous one
            q = chain.ik(pose, q)
            if q is None:
                if i > best[0]:
                    best = (i, f"waypoint {i} (t={ts[i]:.2f}s) is out of reach within the joint limits")
                break
            qs.append(q)
        else:
            try:
                return timed(chain, qs, ts, q_now, max_vel)
            except RuntimeError as e:
                best = (len(poses), str(e))
    raise RuntimeError(best[1])


def timed(chain, qs, ts, q_now, max_vel):
    """Add the move from q_now and the spline velocities; raise if any joint would exceed max_vel."""
    # Get to the first waypoint first, at 90% of the limit (headroom for tracking error and timestamp jitter) and
    # never faster than MAX_JOINT_VEL even when the path itself may be; a cubic with zero end speeds peaks at 1.5x
    # its mean speed.
    approach = max(1.0, 1.5 * np.abs(qs[0] - q_now).max() / (0.9 * min(max_vel, MAX_JOINT_VEL)))
    qs = np.array([q_now] + qs)
    ts = np.r_[0.0, np.asarray(ts) - ts[0] + approach]
    # Monotone (Fritsch-Butland) velocities: zero where a joint turns, so the controller's cubic
    # never leaves the range of its two end points, and points inside the limits stay inside.
    d = np.diff(qs, axis=0) / np.diff(ts)[:, None]
    same = d[:-1] * d[1:] > 0
    vels = np.zeros_like(qs)
    vels[1:-1] = np.where(same, 2 * d[:-1] * d[1:] / (d[:-1] + d[1:] + ~same), 0.0)
    peak = spline_peak(qs, ts, vels)
    if peak.max() > max_vel * (1 + 1e-6):
        i, k = np.unravel_index(int(peak.argmax()), peak.shape)
        raise RuntimeError(f"{chain.names[k]} would reach {peak[i, k]:.2f} rad/s between t={ts[i]:.3f}s and "
                           f"t={ts[i+1]:.3f}s of the move (limit {max_vel} rad/s); slow the trajectory down")
    # Drop the current-state point: the controller starts from where the arm is.
    return qs[1:], ts[1:], vels[1:]


def spline_peak(qs, ts, vels):
    """Peak |speed| (segments x joints) of the cubic Hermite segments the trajectory controller runs."""
    s = np.linspace(0.0, 1.0, 21)[:, None, None]
    h = np.diff(ts)[None, :, None]
    v = (6 * s**2 - 6 * s) * (qs[:-1] - qs[1:]) / h + (3 * s**2 - 4 * s + 1) * vels[:-1] + (3 * s**2 - 2 * s) * vels[1:]
    return np.abs(v).max(axis=0)


def path_msg(chain, qs, stamp):
    """The planned hand poses (FK of the IK solutions, so what will be executed) for RViz."""
    msg = Path()
    msg.header.frame_id, msg.header.stamp = BASE, stamp
    for q in qs:
        f, p = chain.fk(q), PoseStamped(header=msg.header)
        p.pose.position.x, p.pose.position.y, p.pose.position.z = f.p
        p.pose.orientation.x, p.pose.orientation.y, p.pose.orientation.z, p.pose.orientation.w = f.M.GetQuaternion()
        msg.poses.append(p)
    return msg


def load_csv(path):
    try:
        rows = np.loadtxt(path, delimiter=",", ndmin=2, comments="#")
    except ValueError as e:
        raise SystemExit(f"{path}: not a t,x,y,z,roll,pitch,yaw file (header lines must start with #): {e}")
    if rows.shape[1] != 7 or np.any(np.diff(rows[:, 0]) <= 0):
        raise SystemExit(f"{path}: need 7 columns t,x,y,z,roll,pitch,yaw with t strictly increasing")
    # Optional "# start_joints: q0 ... q7": an arm configuration known to work for the first pose.
    hint = [np.array(line.split(":", 1)[1].split(), float) for line in open(path)
            if line.startswith("# start_joints:")]
    return [kdl.Frame(kdl.Rotation.RPY(*r[4:7]), kdl.Vector(*r[1:4])) for r in rows], rows[:, 0], \
        (hint[0] if hint else None)


def load_joints(path, chain):
    """Rows t,q... in chain.names order (no IK). Raises RuntimeError if a joint is outside its limits."""
    try:
        rows = np.loadtxt(path, delimiter=",", ndmin=2, comments="#")
    except ValueError as e:
        raise SystemExit(f"{path}: not a t,{','.join(chain.names)} file (header lines must start with #): {e}")
    if rows.shape[1] != 1 + len(chain.names) or np.any(np.diff(rows[:, 0]) <= 0):
        raise SystemExit(f"{path}: need columns t,{','.join(chain.names)} with t strictly increasing")
    qs, ts = rows[:, 1:], rows[:, 0]
    out = (qs < chain.lo) | (qs > chain.hi)
    if out.any():
        i, k = np.argwhere(out)[0]
        raise RuntimeError(f"{chain.names[k]} = {qs[i, k]:.3f} at t={ts[i]:.2f}s is outside its limits "
                           f"{chain.lo[k]:.3f}..{chain.hi[k]:.3f}")
    return list(qs), ts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=("left", "right"), required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--path", help="hand poses t,x,y,z,roll,pitch,yaw (IK solved here)")
    src.add_argument("--joints", help="arm joint angles t,q0..q7 in the controller's joint order (no IK)")
    ap.add_argument("--max-joint-vel", type=float, default=MAX_JOINT_VEL)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.path:
        poses, ts, hint = load_csv(args.path)

    rclpy.init()
    node = rclpy.create_node("cartesian_follow")
    got = {}
    node.create_subscription(String, "/robot_description", lambda m: got.setdefault("urdf", m.data),
                             QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))
    samples = []  # (time, {joint: position}) during the move

    def on_joint_states(m):
        got["js"] = dict(zip(m.name, m.position))
        if got.get("moving"):
            samples.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, got["js"]))
    node.create_subscription(JointState, "/joint_states", on_joint_states, 10)
    for _ in range(50):
        if "urdf" in got and "js" in got:
            break
        rclpy.spin_once(node, timeout_sec=0.1)
    else:
        sys.exit("no /robot_description or /joint_states: wrong ROS_DOMAIN_ID or RMW?")

    chain = Chain(got["urdf"], BASE, f"link_{args.arm[0]}f")
    q_now = np.array([got["js"][n] for n in chain.names])
    try:
        if args.joints:
            qs, ts, vels = timed(chain, *load_joints(args.joints, chain), q_now, args.max_joint_vel)
        else:
            qs, ts, vels = plan(chain, poses, ts, q_now, args.max_joint_vel, hint)
    except RuntimeError as e:
        sys.exit(f"refused, nothing sent: {e}")
    print(f"planned {len(qs)} points over {ts[-1]:.1f}s (incl. {ts[0]:.1f}s to reach the start), "
          f"within joint limits and {args.max_joint_vel} rad/s")
    latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    node.create_publisher(Path, "/cartesian_follow/planned_path", latched).publish(
        path_msg(chain, qs, node.get_clock().now().to_msg()))
    if args.dry_run:
        print("plan shown in RViz on /cartesian_follow/planned_path; Ctrl-C to exit")
        try:
            rclpy.spin(node)
        except KeyboardInterrupt:
            pass
        return

    def call(cli, req):
        if not cli.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"{cli.srv_name} not found")
        f = cli.call_async(req)
        rclpy.spin_until_future_complete(node, f, timeout_sec=10.0)
        return f.result()

    jtc = f"{args.arm}_arm_joint_trajectory_position_controller"
    states = {c.name: c.state for c in call(node.create_client(ListControllers, "/controller_manager/list_controllers"),
                                            ListControllers.Request()).controller}
    if jtc not in states:
        sys.exit(f"{jtc} is not loaded")
    # Same plan as PR #521: activate the arm's trajectory controller, release the forward controller.
    activate = [jtc] if states[jtc] != "active" else []
    deactivate = [FORWARD] if states.get(FORWARD) == "active" else []
    switch_cli = node.create_client(SwitchController, "/controller_manager/switch_controller")

    def switch(on, off):
        if not (on or off):
            return
        req = SwitchController.Request(activate_controllers=on, deactivate_controllers=off,
                                       strictness=SwitchController.Request.STRICT, activate_asap=True)
        req.timeout.sec = 5
        res = call(switch_cli, req)
        if not (res and res.ok):
            raise RuntimeError(f"switch_controller failed (activate {on}, deactivate {off})")

    goal = FollowJointTrajectory.Goal()
    goal.trajectory.joint_names = chain.names
    for q, v, t in zip(qs, vels, ts):
        goal.trajectory.points.append(JointTrajectoryPoint(
            positions=q.tolist(), velocities=v.tolist(),
            time_from_start=Duration(sec=int(t), nanosec=int((t % 1) * 1e9))))

    switch(activate, deactivate)
    handle = None
    try:
        action = ActionClient(node, FollowJointTrajectory, f"/{jtc}/follow_joint_trajectory")
        if not action.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"{jtc} action server not found")
        got["moving"] = True
        f = action.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, f)
        handle = f.result()
        if not handle.accepted:
            raise RuntimeError("goal rejected by the controller")
        got["t0"] = node.get_clock().now().nanoseconds * 1e-9  # trajectory time 0, to within a few ms
        f = handle.get_result_async()
        rclpy.spin_until_future_complete(node, f)
        handle = None
        res = f.result().result
        print(f"controller result: error_code={res.error_code} {res.error_string}")
    except KeyboardInterrupt:
        print("interrupted: cancelling the goal")
    finally:
        if handle is not None:
            rclpy.spin_until_future_complete(node, handle.cancel_goal_async(), timeout_sec=3.0)
        got["moving"] = False
        switch(deactivate, activate)  # put the controllers back as they were

    report(chain, samples, qs, ts, args.max_joint_vel, got.get("t0"))


def report(chain, samples, qs, ts, max_vel, t0):
    """What actually happened on /joint_states during the move."""
    if len(samples) < 2:
        print("no /joint_states samples recorded")
        return
    t = np.array([s[0] for s in samples])
    q = np.array([[s[1][n] for n in chain.names] for s in samples])
    ok = True
    for k, n in enumerate(chain.names):
        out = (q[:, k] < chain.lo[k] - 1e-6) | (q[:, k] > chain.hi[k] + 1e-6)
        if out.any():
            ok = False
            print(f"POSITION LIMIT: {n} reached {q[out, k].min():.3f}..{q[out, k].max():.3f}, "
                  f"limit {chain.lo[k]:.3f}..{chain.hi[k]:.3f}")
    w = 5  # speed over 5 samples (~50 ms at 100 Hz), so timestamp jitter does not read as speed
    dt = t[w:] - t[:-w]
    peak = (np.abs(q[w:] - q[:-w])[dt > 1e-3] / dt[dt > 1e-3, None]).max(axis=0)
    for k in np.flatnonzero(peak > max_vel * 1.02):
        ok = False
        print(f"SPEED: {chain.names[k]} peaked at {peak[k]:.3f} rad/s (limit {max_vel})")
    # Hand position vs plan over the Cartesian part only (the move to the start is joint-space).
    pos = lambda x: list(chain.fk(x).p)
    planned = np.array([pos(x) for x in qs])
    tt = t - (t0 if t0 is not None else t[0])
    on_path = (tt >= ts[0]) & (tt <= ts[-1])
    actual = np.array([pos(x) for x in q[on_path]])
    want = np.column_stack([np.interp(tt[on_path], ts, planned[:, i]) for i in range(3)])
    dev = 1000 * np.linalg.norm(actual - want, axis=1).max() if on_path.any() else float("nan")
    print(f"peak joint speed {peak.max():.3f} rad/s ({chain.names[int(peak.argmax())]}), "
          f"max hand deviation from plan {dev:.1f} mm, {len(samples)} samples")
    print("limits respected" if ok else "LIMITS VIOLATED")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
