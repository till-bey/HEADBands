#!/usr/bin/env python3
"""ROS 2 interface for mobile-base velocity, body lift, feedback, and shutdown.

The CLI and browser dashboard use RobotNavigation. Keep ROS publishers,
subscriptions, timers, and lifecycle management in this module.
"""

import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import JointState

LIFT_MIN_M = 0.0
LIFT_MAX_M = 0.6  # Local interface reference recommends this over the conflicting SDK 0.8 m.
LIFT_FEEDBACK_TIMEOUT_NS = int(1e9)


class BaseController(Node):
    """Use with a single-threaded executor (the default for rclpy.spin).

    Call move_base() from main before spinning, or from this node's callbacks.
    The executor must keep spinning for publishing and timed stopping to work.
    """

    def __init__(self):
        super().__init__("robot_base_control")
        self.publisher = self.create_publisher(Twist, "cmd_vel", 1)
        self._command = Twist()
        self._deadline = None
        self._clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.lift_publisher = self.create_publisher(JointState, "/lifting_controller/commands", 1)
        self._lift_position = None
        self._lift_feedback_at = None
        self._lift_target = None
        self._lift_deadline = None
        self._lift_subscription = self.create_subscription(
            JointState, "/lifting_controller/joint_states", self._on_lift_state, 10)
        self._timer = self.create_timer(0.05, self._tick, clock=self._clock)

    def move_base(self, linear_speed=0.0, angular_speed=0.0, duration=1.0):
        """Set m/s and rad/s for duration seconds, then return immediately.

        Positive speeds mean forward and left. A new call replaces the current
        movement; calls are not queued. Duration uses real elapsed time.
        """
        if not all(math.isfinite(v) for v in (linear_speed, angular_speed, duration)):
            raise ValueError("Speeds and duration must be finite numbers")
        if duration < 0:
            raise ValueError("duration must be zero or greater")
        if duration == 0:
            self.stop()
            return
        if self.publisher.get_subscription_count() == 0:
            raise RuntimeError("No subscriber connected to cmd_vel")

        command = Twist()
        command.linear.x = float(linear_speed)
        command.angular.z = float(angular_speed)
        self._command = command
        self._deadline = self._clock.now().nanoseconds + int(duration * 1e9)

    def _tick(self):
        """Called at 20 Hz while the executor spins."""
        if self._lift_target is not None:
            state = self.lift_status
            if (not state["fresh"] or not state["connected"]
                    or self._clock.now().nanoseconds >= self._lift_deadline):
                self.stop_lift()
            elif abs(state["position"] - self._lift_target) <= 0.002:
                self._lift_target = self._lift_deadline = None
        if self._deadline is None:
            return
        if self._clock.now().nanoseconds >= self._deadline:
            self.stop()
        else:
            self.publisher.publish(self._command)

    def stop(self):
        """Cancel the active movement and request zero velocity immediately."""
        self._deadline = None
        self._command = Twist()
        if rclpy.ok():
            self.publisher.publish(self._command)

    def _on_lift_state(self, message):
        """Accept the documented body_joint feedback, including explicit invalid states."""
        try:
            index = message.name.index("body_joint")
            position = message.position[index]
        except (ValueError, IndexError):
            return
        if not math.isfinite(position):
            self._lift_position = self._lift_feedback_at = None
            self.stop_lift()
            return
        self._lift_position = float(position)
        self._lift_feedback_at = self._clock.now().nanoseconds

    @property
    def lift_status(self):
        fresh = (self._lift_feedback_at is not None and self._lift_position is not None
                 and self._clock.now().nanoseconds - self._lift_feedback_at <= LIFT_FEEDBACK_TIMEOUT_NS)
        return {"connected": self.lift_publisher.get_subscription_count() > 0,
                "position": self._lift_position, "fresh": fresh,
                "moving": self._lift_target is not None}

    def move_lift(self, delta_m):
        """Move the body by a small relative position step; positive raises the real robot."""
        if not math.isfinite(delta_m) or delta_m == 0 or abs(delta_m) > 0.02:
            raise ValueError("Lift step must be finite, nonzero, and at most 0.02 m")
        state = self.lift_status
        if not state["connected"]:
            raise RuntimeError("No subscriber connected to /lifting_controller/commands")
        if not state["fresh"]:
            raise RuntimeError("Fresh body_joint lift feedback is required")
        if state["moving"]:
            raise RuntimeError("Wait for the current body lift step to finish")
        target = state["position"] + delta_m
        if not LIFT_MIN_M - 1e-9 <= target <= LIFT_MAX_M + 1e-9:
            raise ValueError("Lift target must stay between 0.0 and 0.6 m")
        target = min(LIFT_MAX_M, max(LIFT_MIN_M, target))  # Floating-point boundary only.
        message = JointState()
        message.name = [""]
        message.position = [float(target)]
        self.lift_publisher.publish(message)
        self._lift_target = target
        self._lift_deadline = self._clock.now().nanoseconds + int(2e9)
        return target

    def stop_lift(self):
        """Explicitly stop an active lift; command silence is not a documented stop."""
        active = self._lift_target is not None
        self._lift_target = self._lift_deadline = None
        if active and rclpy.ok():
            message = JointState()
            message.name = [""]
            message.velocity = [0.0]
            self.lift_publisher.publish(message)


class RobotNavigation:
    """ROS adapter for an application-owned control loop.

    Use as a context manager and call wait() between velocity updates so ROS
    can publish commands and enforce the BaseController movement deadline.
    """

    def __enter__(self):
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        try:
            self._node = BaseController()
        except BaseException:
            rclpy.shutdown()
            raise
        return self

    @property
    def running(self):
        return rclpy.ok()

    @property
    def connected(self):
        """Whether ROS is running and a drive subscriber is available."""
        return rclpy.ok() and self._node.publisher.get_subscription_count() > 0

    def set_velocity(self, linear_speed=0.0, angular_speed=0.0):
        """Apply m/s and rad/s. Return False if the drive is disconnected."""
        if self._node.publisher.get_subscription_count() == 0:
            self._node.stop()
            return False
        if linear_speed or angular_speed:
            try:
                self._node.move_base(linear_speed, angular_speed, duration=0.15)
            except RuntimeError:
                # Discovery can change between the two subscription checks.
                self._node.stop()
                return False
        else:
            self._node.stop()
        return True

    @property
    def lift_status(self):
        return self._node.lift_status

    def move_lift(self, delta_m):
        return self._node.move_lift(delta_m)

    def stop_lift(self):
        self._node.stop_lift()

    def wait(self, seconds=0.05):
        """Process ROS events for one control period, including timed stops."""
        deadline = time.monotonic() + seconds
        while self.running:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            rclpy.spin_once(self._node, timeout_sec=remaining)

    def log(self, message):
        self._node.get_logger().info(message)

    def __exit__(self, exc_type, exc_value, traceback):
        self._node._timer.cancel()
        try:
            self._node.stop()
            self._node.stop_lift()
            if rclpy.ok():
                time.sleep(0.1)
        finally:
            self._node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


def main(args=None):
    # Preserve Python's Ctrl+C handler so cleanup can publish before shutdown.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = BaseController()
    connection_deadline = node._clock.now().nanoseconds + int(5e9)

    def start_example():
        # Check discovery without blocking other callbacks.
        if node.publisher.get_subscription_count() > 0:
            start_timer.cancel()
            # Running this script commands forward motion in a clear test area.
            node.move_base(linear_speed=0.1, duration=2.0)
            node.get_logger().info("Movement scheduled; this callback returns now")
        elif node._clock.now().nanoseconds >= connection_deadline:
            start_timer.cancel()
            node.get_logger().error("No subscriber on cmd_vel; no movement sent")

    start_timer = node.create_timer(0.1, start_example, clock=node._clock)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        start_timer.cancel()
        node._timer.cancel()
        node.stop()
        node.stop_lift()
        if rclpy.ok():
            time.sleep(0.1)  # Brief opportunity for the stop message to transmit.
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
