"""frisbee_catch_trigger: close the grippers when a throw aimed at the chest arrives (disarmed by default).

The robot is put in its catching pose beforehand with the existing panther_examples tools
(`teach record` -> `teach action` -> `teach play --yes`). This node never moves an arm or the lift.
It only sends one gripper close command per throw, through the existing gripper interface
(unix_msgs/GripperCommand on /left_gripper/gripper_command and /right_gripper/gripper_command,
built like panther_examples/teach.py does).

Subscribes:  /frisbee/intercept (FrisbeeIntercept), /frisbee/disable (std_msgs/Empty)
Publishes:   gripper_topics (unix_msgs/GripperCommand), only when armed
             /frisbee/catch_event (std_msgs/String, JSON): planned / fired / would-fire (disarmed) / missed
Services:    ~/enable (std_srvs/SetBool): refused until gripper_latency_s and catch_window_px are calibrated
"""

from __future__ import annotations

import json
import math

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Empty, String
from std_srvs.srv import SetBool

from frisbee_msgs.msg import FrisbeeIntercept
from frisbee_interception.core.catch_trigger import CatchConfig, CatchScheduler
from frisbee_interception.params import declare_section


class CatchTriggerNode(Node):
    def __init__(self) -> None:
        super().__init__("frisbee_catch_trigger")
        p = self.declare_parameter
        self.cfg = CatchConfig.from_dict(declare_section(self, "catch", CatchConfig()))
        self.topics = list(p("gripper_topics", ["/left_gripper/gripper_command", "/right_gripper/gripper_command"]).value)
        # right gripper: master_gui never closes it beyond 0.67 (panther_examples/robot.py RIGHT_GRIPPER_MAX)
        self.positions = [float(x) for x in p("close_positions", [1.0, 0.67]).value]
        self.velocity = float(p("close_velocity", 1.0).value)
        self.effort = float(p("close_effort", 0.5).value)
        self.max_data_age = float(p("max_data_age_s", 0.25).value)
        self.enabled = False
        if bool(p("enabled", False).value):
            self.get_logger().warn("'enabled: true' in the parameters is ignored: arm with ~/enable after checking the pose")
        if len(self.positions) != len(self.topics):
            raise ValueError("close_positions needs one value per gripper topic")

        self.sched = CatchScheduler(self.cfg)
        self.pubs = []
        try:
            from unix_msgs.msg import GripperCommand  # noqa: PLC0415

            self.GripperCommand = GripperCommand
            self.pubs = [self.create_publisher(GripperCommand, t, 10) for t in self.topics]
        except ImportError:
            self.GripperCommand = None
            self.get_logger().warn("unix_msgs not found: dry run only (catch events are logged, grippers never move)")
        self.event_pub = self.create_publisher(String, "/frisbee/catch_event", 10)
        self.create_subscription(FrisbeeIntercept, "/frisbee/intercept", self._on_intercept, 10)
        self.create_subscription(Empty, "/frisbee/disable", self._on_disable, 10)
        self.create_service(SetBool, "~/enable", self._on_enable)
        self.create_timer(0.005, self._tick)  # 5 ms resolution for the closing time
        problems = self.cfg.calibration_problems()
        self.get_logger().info(
            "frisbee_catch_trigger: DISARMED (dry run)" + (f"; cannot arm yet: {', '.join(problems)}" if problems else "")
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_enable(self, req, res):
        problems = self.cfg.calibration_problems()
        if req.data and problems:
            res.success, res.message = False, "refused: " + ", ".join(problems)
        elif req.data and self.GripperCommand is None:
            res.success, res.message = False, "refused: unix_msgs (gripper interface) not available"
        else:
            self.enabled = bool(req.data)
            self.sched.plan = None
            res.success, res.message = True, f"catch trigger {'ARMED' if self.enabled else 'disarmed'}"
            self.get_logger().warn(res.message)
        return res

    def _on_disable(self, _msg) -> None:
        if self.enabled:
            self.get_logger().warn("catch trigger DISARMED by /frisbee/disable")
        self.enabled = False
        self.sched.plan = None

    def _event(self, kind: str, **kw) -> None:
        self.event_pub.publish(String(data=json.dumps({"event": kind, "armed": self.enabled, **kw})))

    def _on_intercept(self, msg: FrisbeeIntercept) -> None:
        now = self._now()
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        valid = msg.valid and now - stamp <= self.max_data_age and not math.isnan(msg.pixel_u)
        it = Time.from_msg(msg.intercept_time).nanoseconds * 1e-9 if valid else float("nan")
        pixel = (msg.pixel_u, msg.pixel_v) if valid and not math.isnan(msg.pixel_u) else None
        had = self.sched.plan
        plan = self.sched.update(now, bool(valid and pixel is not None), it, pixel, msg.confidence)
        if plan is not None and plan is not had:
            self._event("planned", close_in_s=round(plan.close_at - now, 3), arrival_in_s=round(plan.intercept_time - now, 3),
                        pixel=[round(plan.pixel[0], 1), round(plan.pixel[1], 1)], confidence=plan.confidence)
        elif had is not None and plan is None:
            self._event("cancelled", reason=self.sched.last_reason)

    def _tick(self) -> None:
        now = self._now()
        had = self.sched.plan
        fired = self.sched.due(now)
        if fired is None:
            if had is not None and self.sched.plan is None:
                self._event("missed", reason=self.sched.last_reason)
            return
        info = dict(late_s=round(now - fired.close_at, 3), arrival_in_s=round(fired.intercept_time - now, 3),
                    pixel=[round(fired.pixel[0], 1), round(fired.pixel[1], 1)])
        if not self.enabled:
            self._event("would_fire", **info)
            return
        for pub, pos in zip(self.pubs, self.positions):
            cmd = self.GripperCommand()
            cmd.mode = "parallel"
            cmd.command.name = [""]
            cmd.command.position = [pos]
            cmd.command.velocity = [self.velocity]
            cmd.command.effort = [self.effort]
            pub.publish(cmd)
        self._event("fired", **info)
        self.get_logger().warn(f"CATCH: grippers closing ({json.dumps(info)})")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CatchTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
