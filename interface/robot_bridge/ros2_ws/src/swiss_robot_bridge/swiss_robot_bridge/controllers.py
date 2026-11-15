"""Drive the Panther's own controllers directly, the way panther_examples/master_gui.py does.

clean_up removed the master_controller node (/master_controller/joint_trajectory), so the bridge now:
- switches left_/right_arm_joint_trajectory_position_controller in for dual_arm_forward_position_controller
  (and the gripper controllers) through the controller_manager services before an arm trajectory, and
  sends each arm its own JointTrajectory (the controller interpolates);
- streams head positions to head_forward_position_controller, which jumps to every command, by
  interpolating them every 0.05 s (`TimedStream`), and sends gripper commands at their times.

Nothing here runs without rclpy; the stream logic is plain Python so it can be tested anywhere.
"""

from __future__ import annotations

import bisect
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from swiss_robot_bridge.panther_topics import (
    FORWARD_CONTROLLER,
    LEFT_TRAJ_CONTROLLER,
    RIGHT_TRAJ_CONTROLLER,
    SRV_LIST_CONTROLLERS,
    SRV_SWITCH_CONTROLLER,
)

logger = logging.getLogger("swiss_robot_bridge")


class ControllerSwitcher:
    """List and switch ros2_control controllers synchronously (short timeouts, never forever)."""

    def __init__(self, rclpy_mod: Any, node: Any) -> None:
        from controller_manager_msgs.srv import ListControllers, SwitchController

        self._rclpy = rclpy_mod
        self._node = node
        self._list_cls = ListControllers
        self._switch_cls = SwitchController
        self._list = node.create_client(ListControllers, SRV_LIST_CONTROLLERS)
        self._switch = node.create_client(SwitchController, SRV_SWITCH_CONTROLLER)

    def _call(self, client: Any, request: Any, timeout_s: float = 3.0) -> Any:
        if not client.wait_for_service(timeout_sec=1.0):
            return None
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while not future.done() and time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.05)
        return future.result() if future.done() else None

    def states(self) -> dict[str, str]:
        res = self._call(self._list, self._list_cls.Request())
        return {c.name: c.state for c in res.controller} if res is not None else {}

    def ensure_active(self, needed: list[str]) -> bool:
        """Activate `needed` controllers; arm trajectory controllers replace the dual-arm forward one."""
        states = self.states()
        if not states:
            logger.warning("controller_manager not reachable; cannot switch %s in", needed)
            return False
        activate = [c for c in needed if states.get(c) != "active"]
        if not activate:
            return True
        deactivate = []
        if {LEFT_TRAJ_CONTROLLER, RIGHT_TRAJ_CONTROLLER} & set(activate) and states.get(FORWARD_CONTROLLER) == "active":
            deactivate.append(FORWARD_CONTROLLER)
        req = self._switch_cls.Request()
        req.activate_controllers = activate
        req.deactivate_controllers = deactivate
        req.strictness = self._switch_cls.Request.STRICT
        req.activate_asap = True
        res = self._call(self._switch, req)
        ok = bool(res is not None and res.ok)
        log = logger.info if ok else logger.error
        log("Controller switch %s: activate=%s deactivate=%s", "ok" if ok else "FAILED", activate, deactivate)
        return ok


@dataclass
class TimedStream:
    """Points (monotonic time, values) played back by `sample`; linear between points, None = keep."""

    points: list[tuple[float, tuple[float | None, ...]]] = field(default_factory=list)

    def active(self) -> bool:
        return bool(self.points)

    def clear(self) -> None:
        self.points = []

    def sample(self, now: float) -> tuple[float | None, ...] | None:
        """Values at `now`; returns the last point once and then empties the stream."""
        if not self.points:
            return None
        times = [p[0] for p in self.points]
        if now >= times[-1]:
            last = self.points[-1][1]
            self.points = []
            return last
        i = bisect.bisect_right(times, now)
        if i == 0:
            return self.points[0][1]
        (t0, a), (t1, b) = self.points[i - 1], self.points[i]
        s = (now - t0) / (t1 - t0) if t1 > t0 else 1.0
        out: list[float | None] = []
        for va, vb in zip(a, b):
            if va is None or vb is None:
                out.append(vb if vb is not None else va)
            else:
                out.append(va + s * (vb - va))
        return tuple(out)


@dataclass
class StepStream:
    """(monotonic time, key, value) events released once their time has come (gripper commands)."""

    events: list[tuple[float, str, float]] = field(default_factory=list)

    def clear(self) -> None:
        self.events = []

    def due(self, now: float) -> list[tuple[str, float]]:
        ready = [(k, v) for t, k, v in self.events if t <= now]
        self.events = [e for e in self.events if e[0] > now]
        return ready
