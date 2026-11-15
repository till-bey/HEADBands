"""CI stand-in for the robot's ROS interfaces the catch node uses (no hardware, no simulator).

  python3 mock_robot.py URDF CATCH_POSE_JSON

Publishes /robot_description (latched) and /joint_states (arms at the catching pose, then wherever the last
trajectory goal went), serves /controller_manager/list_controllers and switch_controller (forward controller active
at start, like the robot), runs FollowJointTrajectory servers for both arm trajectory controllers, and publishes
everything it received on /mock_robot/status (std_msgs/String JSON, 2 Hz).
"""

import json
import sys
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from controller_manager_msgs.msg import ControllerState
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from sensor_msgs.msg import JointState
from std_msgs.msg import String

FORWARD = "dual_arm_forward_position_controller"
TRAJ = {"left": "left_arm_joint_trajectory_position_controller",
        "right": "right_arm_joint_trajectory_position_controller"}


class MockRobot(Node):
    def __init__(self, urdf: str, pose: dict):
        super().__init__("mock_robot")
        cb = ReentrantCallbackGroup()
        self.names = pose["joint_names"]
        self.q = {n: v for arm in self.names for n, v in zip(self.names[arm], pose["nominal"][arm])}
        self.states = {FORWARD: "active", TRAJ["left"]: "inactive", TRAJ["right"]: "inactive"}
        self.switches, self.goals = [], []
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_publisher(String, "/robot_description", latched).publish(String(data=urdf))
        self.js_pub = self.create_publisher(JointState, "/joint_states", 10)
        self.status_pub = self.create_publisher(String, "/mock_robot/status", 10)
        self.create_service(ListControllers, "/controller_manager/list_controllers", self._list, callback_group=cb)
        self.create_service(SwitchController, "/controller_manager/switch_controller", self._switch, callback_group=cb)
        for arm, ctrl in TRAJ.items():
            ActionServer(self, FollowJointTrajectory, f"/{ctrl}/follow_joint_trajectory",
                         execute_callback=lambda gh, a=arm: self._execute(a, gh),
                         goal_callback=lambda g, a=arm: self._goal(a, g),
                         cancel_callback=lambda _g: CancelResponse.ACCEPT, callback_group=cb)
        self.create_timer(0.02, self._publish_js, callback_group=cb)
        self.create_timer(0.5, self._status, callback_group=cb)

    def _list(self, _req, res):
        res.controller = [ControllerState(name=n, state=s) for n, s in self.states.items()]
        return res

    def _switch(self, req, res):
        self.switches.append({"t": time.monotonic(), "activate": list(req.activate_controllers),
                              "deactivate": list(req.deactivate_controllers)})
        for c in req.deactivate_controllers:
            self.states[c] = "inactive"
        for c in req.activate_controllers:
            self.states[c] = "active"
        res.ok = True
        return res

    def _goal(self, arm, goal):
        # like ros2_control: a trajectory controller that is not active rejects goals
        accepted = self.states[TRAJ[arm]] == "active"
        pt = goal.trajectory.points[-1]
        self.goals.append({"t": time.monotonic(), "arm": arm, "accepted": accepted,
                           "joint_names": list(goal.trajectory.joint_names), "positions": list(pt.positions),
                           "duration": pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9})
        return GoalResponse.ACCEPT if accepted else GoalResponse.REJECT

    def _execute(self, arm, gh):
        pt = gh.request.trajectory.points[-1]
        time.sleep(min(pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9, 1.0))
        self.q.update(zip(gh.request.trajectory.joint_names, pt.positions))
        gh.succeed()
        return FollowJointTrajectory.Result()

    def _publish_js(self):
        msg = JointState(name=list(self.q), position=list(self.q.values()))
        msg.header.stamp = self.get_clock().now().to_msg()
        self.js_pub.publish(msg)

    def _status(self):
        self.status_pub.publish(String(data=json.dumps({"states": self.states, "switches": self.switches,
                                                        "goals": self.goals})))


def main():
    rclpy.init()
    node = MockRobot(open(sys.argv[1]).read(), json.load(open(sys.argv[2])))
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
