"""Button-controlled navigation using main.control_loop(), shared with the CLI.

The camera/UI thread supplies observations. A separate worker owns the ROS node
so camera waits cannot block velocity expiry, Stop, or browser lease checks.
"""

from threading import Event, Lock, Thread
import time
import math

if __package__:
    from .controller import PController
    from .main import control_loop
else:
    from controller import PController
    from main import control_loop


def _create_robot():
    # Preview and configuration remain usable without ROS installed or sourced.
    if __package__:
        from .navigation import RobotNavigation
    else:
        from navigation import RobotNavigation
    return RobotNavigation()


class WebRobotControl:
    def __init__(self, robot_factory=None, clock=time.monotonic):
        self._factory = robot_factory or _create_robot
        self._clock = clock
        self._lock = Lock()
        self._shutdown = Event()
        self._thread = None
        self._connection = "disconnected"
        self._drive_connected = False
        self._enabled = False
        self._lease_deadline = 0.0
        self._message = "ROS disconnected; preview only"
        self._linear_speed = self._angular_speed = 0.0
        self._position = None
        self._observed_at = 0.0
        self._camera_status = "No camera observation"
        self._controller = PController()
        self._target_timeout = 0.5
        self._target_generation = 0
        self._run_generation = 0
        self._lift = {"connected": False, "fresh": False, "position": None, "moving": False}
        self._lift_pending = None
        self._lift_stop_pending = False
        self._head = {"connected": False, "fresh": False, "pitch": None, "yaw": None, "moving": False}
        self._head_pending = None
        self._head_stop_pending = False

    def _adjusting(self):
        return (self._lift["moving"] or self._lift_pending is not None
                or self._head["moving"] or self._head_pending is not None)

    def update_target(self, position, observed_at, controller, *, target_timeout, camera_status="Tracking",
                      expected_generation=None):
        with self._lock:
            if expected_generation is not None and expected_generation != self._target_generation:
                return  # A camera switch invalidated this in-flight observation.
            if self._adjusting():
                return  # Ground projection changes while the camera is being raised.
            self._position, self._observed_at = position, observed_at
            self._controller = controller
            self._target_timeout = target_timeout
            self._camera_status = camera_status

    def clear_target(self, reason="Camera disconnected; navigation stopped"):
        with self._lock:
            self._target_generation += 1
            self._run_generation += 1
            self._position = None
            self._observed_at = 0.0
            self._enabled = False
            self._lift_pending = None
            self._lift_stop_pending = True
            self._head_pending = None
            self._head_stop_pending = True
            self._message = reason

    def target_generation(self):
        with self._lock:
            return self._target_generation

    def connect(self):
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return "ROS connection is already active"
            self._shutdown = Event()
            self._run_generation += 1
            self._enabled = False
            self._connection = "connecting"
            self._message = "Connecting ROS"
            self._thread = Thread(target=self._run, daemon=True, name="browser-navigation")
            self._thread.start()
        return "Connecting ROS"

    def start(self, *, expected_generation=None):
        with self._lock:
            if expected_generation is not None and expected_generation != self._run_generation:
                raise ValueError("Run request expired after a Stop or settings change. Click Run again.")
            if self._connection != "connected" or self._shutdown.is_set():
                raise ValueError("Connect ROS before running the controller")
            if not self._drive_connected:
                raise ValueError("No drive subscriber connected to cmd_vel")
            if self._adjusting():
                raise ValueError("Wait for body/head movement to stop before running navigation")
            if (self._position is None
                    or self._clock() - self._observed_at > self._target_timeout
                    or self._controller.compute(self._position).state == "lost"):
                raise ValueError("Connect the camera and detect a fresh frisbee before starting")
            self._enabled = True
            self._lease_deadline = self._clock() + 3.0
            self._message = "Navigation started"
        return "Navigation started using the applied camera and controller settings"

    def stop(self, reason="Stopped"):
        with self._lock:
            self._run_generation += 1
            self._enabled = False
            self._lift_pending = None
            self._lift_stop_pending = True
            self._head_pending = None
            self._head_stop_pending = True
            self._message = reason
        return "Stop requested"

    def disconnect(self):
        self.stop("Disconnecting ROS")
        self._shutdown.set()
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._connection = "disconnected"
                self._drive_connected = False
                self._message = "ROS disconnected; stopped"
        return "ROS disconnect requested"

    def heartbeat(self):
        with self._lock:
            self._expire_lease()
            if self._enabled or self._adjusting():
                self._lease_deadline = self._clock() + 3.0
        return "Browser connected"

    def _expire_lease(self):
        # Call only while holding the lock. A late heartbeat cannot restart motion.
        if (self._enabled or self._adjusting()) and self._clock() >= self._lease_deadline:
            self._run_generation += 1
            self._enabled = False
            self._lift_pending = None
            self._lift_stop_pending = True
            self._head_pending = None
            self._head_stop_pending = True
            self._message = "Browser disconnected; navigation, lift and head stopped"

    def jog_lift(self, direction, *, expected_generation=None):
        with self._lock:
            self._expire_lease()
            if direction not in ("up", "down"):
                raise ValueError("Lift direction must be up or down")
            if expected_generation is not None and expected_generation != self._run_generation:
                raise ValueError("Lift request expired after a Stop or settings change. Click again.")
            if self._connection != "connected" or self._shutdown.is_set():
                raise ValueError("Connect ROS before adjusting body height")
            if not self._lift["connected"] or not self._lift["fresh"]:
                raise ValueError("A connected lift controller and fresh height feedback are required")
            if self._adjusting():
                raise ValueError("Wait for the current body/head movement to finish")
            delta = 0.01 if direction == "up" else -0.01
            if not -1e-9 <= self._lift["position"] + delta <= 0.6 + 1e-9:
                raise ValueError("Lift target must stay between 0.0 and 0.6 m")
            self._enabled = False
            self._position = None
            self._observed_at = 0.0
            self._target_generation += 1
            self._run_generation += 1
            self._lift_pending = delta
            self._lease_deadline = self._clock() + 3.0
            self._message = f"Body {direction} requested; navigation stopped"
        return "Body step requested (1 cm)"

    def jog_head(self, direction, *, expected_generation=None):
        with self._lock:
            self._expire_lease()
            if direction not in ("up", "down"):
                raise ValueError("Head direction must be up or down")
            if expected_generation is not None and expected_generation != self._run_generation:
                raise ValueError("Head request expired after Stop or a settings change. Click again.")
            if self._connection != "connected" or self._shutdown.is_set():
                raise ValueError("Connect ROS before adjusting the head")
            if not self._head["connected"] or not self._head["fresh"]:
                raise ValueError("A connected head controller and fresh pitch/yaw feedback are required")
            if self._adjusting():
                raise ValueError("Wait for the current body/head movement to finish")
            delta = math.radians(5) * (-1 if direction == "up" else 1)
            if not -0.26 <= self._head["pitch"] + delta <= 1.41:
                raise ValueError("Head pitch must stay between -0.26 and 1.41 rad")
            self._enabled = False
            self._position = None
            self._observed_at = 0.0
            self._target_generation += 1
            self._run_generation += 1
            self._head_pending = delta
            self._lease_deadline = self._clock() + 3.0
            self._message = f"Head {direction} requested; navigation stopped"
        return "Head step requested (5 degrees); update camera tilt if the camera moves with the head"

    def _service_head(self, robot):
        with self._lock:
            self._expire_lease()
            if self._head_stop_pending or self._shutdown.is_set():
                robot.stop_head()
                self._head_stop_pending = False
            if self._head_pending is not None and not self._shutdown.is_set():
                delta, self._head_pending = self._head_pending, None
                try:
                    robot.set_velocity(0.0, 0.0)
                    robot.move_head(delta)
                except (ValueError, RuntimeError) as exc:
                    self._message = f"Head movement rejected: {exc}"
            previous_moving = self._head["moving"]
            self._head = robot.head_status.copy()
            if previous_moving != self._head["moving"]:
                self._target_generation += 1
                self._position = None
                self._observed_at = 0.0
                self._enabled = False

    def _service_lift(self, robot):
        # Only the ROS worker touches the adapter. Serialize against Stop so a
        # queued command can never be sent after Stop has cancelled it.
        with self._lock:
            self._expire_lease()
            if self._lift_stop_pending or self._shutdown.is_set():
                robot.stop_lift()
                self._lift_stop_pending = False
            if self._lift_pending is not None and not self._shutdown.is_set():
                delta, self._lift_pending = self._lift_pending, None
                try:
                    robot.set_velocity(0.0, 0.0)
                    robot.move_lift(delta)
                except (ValueError, RuntimeError) as exc:
                    self._message = f"Body lift rejected: {exc}"
            previous_moving = self._lift["moving"]
            self._lift = robot.lift_status.copy()
            if previous_moving != self._lift["moving"]:
                # Discard observations acquired across a change in body height.
                self._target_generation += 1
                self._position = None
                self._observed_at = 0.0
                self._enabled = False

    def status(self):
        with self._lock:
            self._expire_lease()
            return {
                "connection": self._connection,
                "generation": self._run_generation,
                "running": self._enabled,
                "drive_connected": self._drive_connected,
                "status": self._message,
                "linear_speed": self._linear_speed,
                "angular_speed": self._angular_speed,
                "lift_connected": self._lift["connected"],
                "lift_fresh": self._lift["fresh"],
                "lift_position": self._lift["position"],
                "lift_moving": self._lift["moving"] or self._lift_pending is not None,
                "head_connected": self._head["connected"],
                "head_fresh": self._head["fresh"],
                "head_pitch": self._head["pitch"],
                "head_yaw": self._head["yaw"],
                "head_moving": self._head["moving"] or self._head_pending is not None,
            }

    def snapshot(self):
        """Reader interface consumed by the shared main.control_loop()."""
        with self._lock:
            self._expire_lease()
            if not self._enabled:
                return None, self._clock(), self._message
            return self._position, self._observed_at, self._camera_status

    def _send_velocity(self, robot, linear_speed, angular_speed):
        with self._lock:
            self._expire_lease()
            self._drive_connected = bool(robot.connected)
            if not self._drive_connected:
                if self._enabled:
                    self._run_generation += 1
                self._enabled = False
                self._message = "Waiting for cmd_vel subscriber; navigation stopped"
            # Recheck at send time: Stop may arrive after the control step read a target.
            if (not self._enabled or self._shutdown.is_set() or self._position is None
                    or self._clock() - self._observed_at > self._target_timeout):
                linear_speed = angular_speed = 0.0
            sent = robot.set_velocity(linear_speed, angular_speed)
            if not sent:
                if self._enabled:
                    self._run_generation += 1
                self._enabled = self._drive_connected = False
                self._message = "Drive disconnected; navigation stopped"
            self._linear_speed = linear_speed if sent else 0.0
            self._angular_speed = angular_speed if sent else 0.0
            return sent

    def _prepare_step(self, robot):
        self._service_head(robot)
        self._service_lift(robot)
        with self._lock:
            return self._controller, self._target_timeout

    def _report_status(self, message):
        with self._lock:
            if self._enabled:
                self._message = message

    def _run(self):
        try:
            with self._factory() as robot:
                with self._lock:
                    self._connection = "connected"
                    self._drive_connected = bool(robot.connected)
                    self._message = "ROS connected; stopped"
                guarded = _GuardedRobot(self, robot)
                control_loop(self, self._controller, guarded,
                             prepare_step=lambda: self._prepare_step(robot),
                             on_status=self._report_status)
        except Exception as exc:
            with self._lock:
                self._connection = "error"
                self._message = f"ROS unavailable or failed: {exc}. Check the sourced ROS environment."
        finally:
            with self._lock:
                self._run_generation += 1
                self._enabled = self._drive_connected = False
                self._linear_speed = self._angular_speed = 0.0
                self._lift = {"connected": False, "fresh": False, "position": None, "moving": False}
                self._lift_pending = None
                self._head = {"connected": False, "fresh": False, "pitch": None, "yaw": None, "moving": False}
                self._head_pending = None
                if self._connection != "error":
                    self._connection = "disconnected"
                    self._message = "ROS disconnected; stopped"

    def close(self):
        self.disconnect()
        if self._thread is not None:
            self._thread.join(timeout=3.0)


class _GuardedRobot:
    def __init__(self, control, robot):
        self._control, self._robot = control, robot

    @property
    def running(self):
        return not self._control._shutdown.is_set() and self._robot.running

    def set_velocity(self, linear_speed, angular_speed):
        return self._control._send_velocity(self._robot, linear_speed, angular_speed)

    def wait(self, seconds):
        self._robot.wait(seconds)

    def log(self, message):
        self._robot.log(message)
