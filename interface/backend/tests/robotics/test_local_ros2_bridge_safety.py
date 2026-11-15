"""Unit tests for the UniX AI Panther local ROS 2 safety supervisor, topic adapter, and on-demand camera manager."""

from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

BRIDGE_SRC = (
    Path(__file__).resolve().parents[3]
    / "robot_bridge"
    / "ros2_ws"
    / "src"
    / "swiss_robot_bridge"
)
if str(BRIDGE_SRC) not in sys.path:
    sys.path.insert(0, str(BRIDGE_SRC))

from swiss_robot_bridge.bridge_node import ROS2HardwareAdapter
from swiss_robot_bridge.camera_manager import (
    PantherCameraManager,
    extract_center_depth_meters,
)
from swiss_robot_bridge.local_safety import (
    LocalSafetyRejectedError,
    LocalSafetySupervisor,
)
from swiss_robot_bridge.panther_topics import (
    PANTHER_RMW_IMPLEMENTATION,
    PANTHER_ROS_DOMAIN_ID,
    SRV_ULTRASONIC_SWITCH,
    TOPIC_BATTERY_STATE,
    TOPIC_CAMERA_COLOR_RAW,
    TOPIC_CAMERA_DEPTH_RAW,
    TOPIC_CMD_VEL,
    TOPIC_DIAGNOSTICS,
    TOPIC_FRONT_SCAN,
    TOPIC_HEAD_COMMANDS,
    TOPIC_LEFT_GRIPPER_CMD,
    TOPIC_LIFT_COMMANDS,
    TOPIC_ODOM,
    TOPIC_PREEMPT_TELEOP,
    TOPIC_RIGHT_GRIPPER_CMD,
)


def test_panther_ros2_topics_match_real_unix_ai_panther_reference() -> None:
    assert PANTHER_ROS_DOMAIN_ID == 15
    assert PANTHER_RMW_IMPLEMENTATION == "rmw_cyclonedds_cpp"
    assert TOPIC_CMD_VEL == "/cmd_vel"
    assert TOPIC_PREEMPT_TELEOP == "/preempt_teleop"
    assert TOPIC_HEAD_COMMANDS == "/head_forward_position_controller/commands"
    assert TOPIC_LIFT_COMMANDS == "/lifting_controller/commands"
    assert TOPIC_LEFT_GRIPPER_CMD == "/left_gripper/gripper_command"
    assert TOPIC_RIGHT_GRIPPER_CMD == "/right_gripper/gripper_command"
    assert SRV_ULTRASONIC_SWITCH == "/movebase3_drive_controller/ultrasonic_switch"
    assert TOPIC_BATTERY_STATE == "/battery_state_broadcaster/battery_state"
    assert TOPIC_ODOM == "/odom"
    assert TOPIC_DIAGNOSTICS == "/diagnostics"
    assert TOPIC_FRONT_SCAN == "/scan"
    assert TOPIC_CAMERA_COLOR_RAW == "/camera/color/image_raw"
    assert TOPIC_CAMERA_DEPTH_RAW == "/camera/depth/image_raw"


def test_local_estop_works_completely_offline_and_halts_motors() -> None:
    sup = LocalSafetySupervisor()
    sup.record_cloud_heartbeat(10.0)
    now = datetime.now(tz=UTC)

    setpoint = sup.validate_and_compute_setpoint(
        {
            "command_id": "cmd-1",
            "action": "move_to",
            "target": {"x": 1.0, "y": 0.0, "z": 0.0},
            "speed": 0.2,
            "requires_confirmation": True,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=20)).isoformat(),
        },
        now_utc=now,
    )
    assert setpoint.linear_x == pytest.approx(0.2)

    # Trigger local E-Stop offline -> immediate zero velocity and blocks subsequent commands
    halted = sup.trigger_local_estop()
    assert halted.linear_x == 0.0
    assert halted.angular_z == 0.0
    assert sup.local_estop_latched is True

    with pytest.raises(LocalSafetyRejectedError, match="Local E-Stop"):
        sup.validate_and_compute_setpoint(
            {
                "command_id": "cmd-2",
                "action": "move_to",
                "target": {"x": 0.5, "y": 0.0, "z": 0.0},
                "speed": 0.1,
                "confirmed_by_human": True,
                "expires_at": (now + timedelta(seconds=20)).isoformat(),
            },
            now_utc=now,
        )


def test_local_watchdog_zeroes_velocity_on_cloud_disconnect() -> None:
    sup = LocalSafetySupervisor(watchdog_timeout_seconds=2.0)
    sup.record_cloud_heartbeat(100.0)
    now = datetime.now(tz=UTC)
    sup.validate_and_compute_setpoint(
        {
            "command_id": "cmd-watch",
            "action": "move_to",
            "target": {"x": 1.0, "y": 0.0, "z": 0.0},
            "speed": 0.25,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=20)).isoformat(),
        },
        now_utc=now,
    )
    assert sup.evaluate_watchdog(101.0).linear_x == pytest.approx(0.25)
    # 3.5 seconds without cloud heartbeat -> automatic motor stop!
    assert sup.evaluate_watchdog(103.5).linear_x == 0.0


def test_panther_plan_timed_cmd_vel_lift_gripper_head_and_lidar_stop() -> None:
    sup = LocalSafetySupervisor(watchdog_timeout_seconds=10.0)
    sup.record_cloud_heartbeat(50.0)
    now = datetime.now(tz=UTC)

    # 1. Move + vertical lift clamped strictly to Panther URDF [0.0, 0.60]m and timed duration
    move_plan = sup.validate_and_compute_panther_plan(
        {
            "command_id": "cmd-panther-move",
            "action": "move_to",
            "target": {"x": 1.0, "y": 0.0, "z": 0.95},
            "speed": 0.25,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=30)).isoformat(),
        },
        now_utc=now,
        monotonic_now=50.0,
    )
    assert move_plan.twist.linear_x == pytest.approx(0.25)
    assert move_plan.twist_duration_seconds == pytest.approx(4.0)
    assert move_plan.lift_position_m == pytest.approx(0.80)
    assert move_plan.enable_ultrasonics is True
    assert sup.evaluate_watchdog(52.0).linear_x == pytest.approx(0.25)
    # Once 4.0s duration completes -> /cmd_vel is automatically zeroed (never latched indefinitely)!
    sup.record_cloud_heartbeat(54.1)
    assert sup.evaluate_watchdog(54.1).linear_x == 0.0

    # 2. Pick & place map to Panther parallel grippers (1.0 closed, 0.0 open) + camera trigger
    pick_plan = sup.validate_and_compute_panther_plan(
        {
            "command_id": "cmd-pick",
            "action": "pick_object",
            "object_label": "frisbee",
            "target": {"x": 0.5, "y": -0.2, "z": 0.1},
            "speed": 0.15,
            "confirmed_by_human": True,
            "rationale": "Grasp frisbee with right gripper",
            "expires_at": (now + timedelta(seconds=30)).isoformat(),
        },
        now_utc=now,
    )
    assert pick_plan.gripper_side == "right"
    assert pick_plan.gripper_position == pytest.approx(1.0)
    assert pick_plan.gripper_mode == "parallel"
    assert pick_plan.trigger_camera is True

    place_plan = sup.validate_and_compute_panther_plan(
        {
            "command_id": "cmd-place",
            "action": "place_object",
            "object_label": "frisbee",
            "speed": 0.15,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=30)).isoformat(),
        },
        now_utc=now,
    )
    assert place_plan.gripper_side == "left"
    assert place_plan.gripper_position == pytest.approx(0.0)
    assert place_plan.trigger_camera is True

    # 3. Inspect / look_at aims Panther head within PR #521 [-0.34, +0.34] pitch & [-1.2, +1.2] yaw
    inspect_plan = sup.validate_and_compute_panther_plan(
        {
            "command_id": "cmd-look",
            "action": "inspect_object",
            "object_label": "frisbee",
            "target": {"x": 0.8, "y": 0.3, "z": 0.0},
            "speed": 0.1,
            "requires_confirmation": False,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=30)).isoformat(),
        },
        now_utc=now,
    )
    assert inspect_plan.head_positions is not None
    assert -0.34 <= inspect_plan.head_positions[0] <= 0.34
    assert -1.20 <= inspect_plan.head_positions[1] <= 1.20
    assert inspect_plan.twist.linear_x == 0.0
    assert inspect_plan.trigger_camera is True

    # 4. Front LiDAR (/scan) obstacle < 0.35m blocks forward motion
    sup.front_obstacle_distance_m = 0.22
    with pytest.raises(LocalSafetyRejectedError, match="Front LiDAR obstacle"):
        sup.validate_and_compute_panther_plan(
            {
                "command_id": "cmd-blocked",
                "action": "move_to",
                "target": {"x": 1.0, "y": 0.0, "z": 0.0},
                "speed": 0.2,
                "confirmed_by_human": True,
                "expires_at": (now + timedelta(seconds=30)).isoformat(),
            },
            now_utc=now,
        )


def test_panther_ros2_adapter_diagnostics_estop_battery_and_scan_callbacks() -> None:
    sup = LocalSafetySupervisor()
    cam = PantherCameraManager(allow_sim_fallback=True)
    hw = ROS2HardwareAdapter(supervisor=sup, camera_manager=cam)

    # BatteryState reporting 0.83 fraction -> normalized to 83.0%
    hw._on_battery(SimpleNamespace(percentage=0.83))
    assert sup.battery_percent == pytest.approx(83.0)

    # Front LaserScan reporting obstacle at 0.28m directly ahead (0 rad)
    hw._on_front_scan(
        SimpleNamespace(
            ranges=[2.0, 0.28, 1.8],
            range_min=0.05,
            range_max=10.0,
            angle_min=-0.2,
            angle_increment=0.2,
        )
    )
    assert sup.front_obstacle_distance_m == pytest.approx(0.28)

    # Real rclpy messages carry level as bytes (b"\x00" = OK): must not raise, must not latch
    hw._on_diagnostics(
        SimpleNamespace(status=[SimpleNamespace(name="diagnostic_broadcaster: CAN bus", message="", level=b"\x00", values=[])])
    )
    hw._on_diagnostics(SimpleNamespace(status=[SimpleNamespace(name="E-Stop", message="", level=b"\x00", values=[])]))
    assert sup.local_estop_latched is False

    # Panther /diagnostics reporting E-Stop engaged -> latches local E-Stop offline
    diag_msg = SimpleNamespace(
        status=[
            SimpleNamespace(
                name="panther_safety: E-Stop",
                message="Emergency stop active",
                level=2,
                values=[SimpleNamespace(key="estop", value="true")],
            )
        ]
    )
    hw._on_diagnostics(diag_msg)
    assert sup.local_estop_latched is True


def test_panther_camera_manager_decodes_ros_image_depth_and_triggers_on_demand() -> None:
    cam = PantherCameraManager(
        camera_launch_cmd="",
        idle_timeout_seconds=10.0,
        allow_sim_fallback=True,
    )

    # Ingest 16UC1 depth image (1350 mm = 1.35 m)
    depth_mm = np.full((24, 32), 1350, dtype=np.uint16)
    depth_msg = SimpleNamespace(height=24, width=32, encoding="16UC1", data=depth_mm.tobytes())
    assert extract_center_depth_meters(depth_msg) == pytest.approx(1.35)
    cam.ingest_ros_depth_image(depth_msg)

    # Ingest raw sensor_msgs/Image (bgr8) from /camera/color/image_raw
    bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    bgr[:, :] = (30, 110, 240)
    img_msg = SimpleNamespace(height=48, width=64, encoding="bgr8", step=64 * 3, data=bgr.tobytes())
    captured = cam.ingest_ros_image(img_msg, TOPIC_CAMERA_COLOR_RAW)
    assert captured is not None
    assert captured.source == "/camera/color/image_raw"
    assert captured.center_depth_m == pytest.approx(1.35)
    assert len(captured.image_bytes) > 64

    # On-demand capture returns fresh frame and activates lazy subscription callback
    activated = False

    def _activate() -> None:
        nonlocal activated
        activated = True

    on_demand = asyncio.run(cam.capture_frame_on_demand(activate_ros_camera_cb=_activate))
    assert activated is True
    assert on_demand is not None

    # Idle shutdown deactivates high-bandwidth camera subscription after idle timeout
    deactivated = False

    def _deactivate() -> None:
        nonlocal deactivated
        deactivated = True

    assert cam._last_requested_monotonic is not None
    shut = cam.check_idle_shutdown(
        cam._last_requested_monotonic + 15.0,
        deactivate_ros_camera_cb=_deactivate,
    )
    assert shut is True
    assert deactivated is True


def test_local_gate_rejects_unconfirmed_expired_and_raw_topic_commands() -> None:
    sup = LocalSafetySupervisor()
    now = datetime.now(tz=UTC)

    # Unconfirmed motion rejected locally even if it bypassed cloud
    with pytest.raises(LocalSafetyRejectedError, match="confirmed_by_human"):
        sup.validate_and_compute_setpoint(
            {
                "command_id": "c-unconf",
                "action": "move_to",
                "target": {"x": 0.5, "y": 0.0, "z": 0.0},
                "speed": 0.1,
                "confirmed_by_human": False,
                "expires_at": (now + timedelta(seconds=10)).isoformat(),
            },
            now_utc=now,
        )

    # Expired command rejected locally
    with pytest.raises(LocalSafetyRejectedError, match="expired"):
        sup.validate_and_compute_setpoint(
            {
                "command_id": "c-exp",
                "action": "move_to",
                "target": {"x": 0.5, "y": 0.0, "z": 0.0},
                "speed": 0.1,
                "confirmed_by_human": True,
                "expires_at": (now - timedelta(seconds=1)).isoformat(),
            },
            now_utc=now,
        )

    # Arbitrary ROS topic field rejected locally
    with pytest.raises(LocalSafetyRejectedError, match="Direct ROS topic"):
        sup.validate_and_compute_setpoint(
            {
                "command_id": "c-topic",
                "action": "stop",
                "ros_topic": "/joint_trajectory_controller/commands",
                "expires_at": (now + timedelta(seconds=10)).isoformat(),
            },
            now_utc=now,
        )


def test_sequential_multistep_command_queue_and_closed_loop_odom_precision() -> None:
    sup = LocalSafetySupervisor(watchdog_timeout_seconds=30.0)
    sup.record_cloud_heartbeat(10.0)
    now = datetime.now(tz=UTC)

    # 1. Confirm Step 1 (5.0m forward @ 0.25 m/s = 20.0s) and Step 2 (rotate -90 deg @ 0.4 rad/s) back-to-back
    p1, immediate1 = sup.submit_cloud_command(
        {
            "command_id": "step-5m",
            "action": "move_to",
            "target": {"x": 5.0, "y": 0.0, "z": 0.0},
            "speed": 0.25,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=60)).isoformat(),
        },
        now_utc=now,
        monotonic_now=10.0,
    )
    assert immediate1 is True
    assert p1.target_distance_m == pytest.approx(5.0)
    assert p1.twist_duration_seconds == pytest.approx(20.0)

    p2, immediate2 = sup.submit_cloud_command(
        {
            "command_id": "step-rot90",
            "action": "rotate",
            "yaw_rad": -1.5708,
            "speed": 0.2,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=60)).isoformat(),
        },
        now_utc=now,
        monotonic_now=10.1,
    )
    assert immediate2 is False
    assert sup.active_command_id == "step-5m"
    assert len(sup.pending_plans) == 1

    # Mid-way at t=20.0s (10s elapsed of 20s): Step 1 is STILL driving at 0.25 m/s (not overwritten by Step 2!)
    sup.record_cloud_heartbeat(20.0)
    mid_twist = sup.evaluate_watchdog(20.0)
    assert mid_twist.linear_x == pytest.approx(0.25)
    assert mid_twist.angular_z == pytest.approx(0.0)

    # At t=30.1s (Step 1 finished full 5.00m): Step 2 promotes automatically!
    sup.record_cloud_heartbeat(30.1)
    rot_twist = sup.evaluate_watchdog(30.1)
    promoted = sup.pop_promoted_plan()
    assert promoted is not None
    assert promoted.command_id == "step-rot90"
    assert sup.active_command_id == "step-rot90"
    assert rot_twist.linear_x == pytest.approx(0.0)
    assert rot_twist.angular_z == pytest.approx(-0.4)

    # Finish Step 2 -> motors halt cleanly to zero
    sup.record_cloud_heartbeat(35.0)
    done_twist = sup.evaluate_watchdog(35.0)
    assert done_twist.angular_z == pytest.approx(0.0)
    assert sup.active_command_id is None

    # 2. Closed-loop /odom distance tracking decelerates in final 20cm and halts within 1cm of 5.00m
    sup.update_odometry(0.0, 0.0, 0.0, monotonic_now=100.0)
    sup.record_cloud_heartbeat(100.0)
    sup.submit_cloud_command(
        {
            "command_id": "odom-5m",
            "action": "move_to",
            "target": {"x": 5.0, "y": 0.0, "z": 0.0},
            "speed": 0.25,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=60)).isoformat(),
        },
        now_utc=now,
        monotonic_now=100.0,
    )
    # At 4.90m (10cm remaining < 20cm decel zone) -> speed scaled smoothly to 50% (0.125 m/s)
    sup.update_odometry(4.90, 0.0, 0.0, monotonic_now=119.5)
    sup.record_cloud_heartbeat(119.5)
    decel_twist = sup.evaluate_watchdog(119.5)
    assert decel_twist.linear_x == pytest.approx(0.125, abs=1e-3)

    # At 4.995m (within 1cm of 5.00m) -> stops immediately!
    sup.update_odometry(4.995, 0.0, 0.0, monotonic_now=120.2)
    sup.record_cloud_heartbeat(120.2)
    stop_twist = sup.evaluate_watchdog(120.2)
    assert stop_twist.linear_x == pytest.approx(0.0)
    assert sup.active_command_id is None


def test_headbands_clean_up_throw_trajectory_gripper_sentinel_and_detector() -> None:
    from swiss_robot_bridge.camera_manager import detect_ground_frisbee_candidates
    from swiss_robot_bridge.panther_topics import (
        PANTHER_SIM_ROS_DOMAIN_ID,
        TOPIC_MASTER_TRAJ,
        can_service,
        single_position,
    )

    assert PANTHER_SIM_ROS_DOMAIN_ID == 77
    assert TOPIC_MASTER_TRAJ == "/master_controller/joint_trajectory"
    assert can_service("joint_la0", "enable") == "/can0/joint_la0/enable"
    assert can_service("joint_ra7", "disable") == "/can1/joint_ra7/disable"

    # 1. Panther gripper state sentinel filter (-32.767 from int16 -32767 / 1000 is ignored)
    assert single_position(SimpleNamespace(position=[-32.767])) is None
    assert single_position(SimpleNamespace(position=[0.85])) == pytest.approx(0.85)

    sup = LocalSafetySupervisor()
    cam = PantherCameraManager(allow_sim_fallback=True)
    hw = ROS2HardwareAdapter(supervisor=sup, camera_manager=cam)
    hw._on_left_gripper_state(SimpleNamespace(position=[0.65]))
    assert hw.left_gripper_pos == pytest.approx(0.65)
    hw._on_left_gripper_state(SimpleNamespace(position=[-32.767]))
    assert hw.left_gripper_pos == pytest.approx(0.65)  # Not overwritten by -32.767 sentinel!

    # 2. Retargeted right-arm frisbee throw trajectory plan on /master_controller/joint_trajectory
    now = datetime.now(tz=UTC)
    throw_plan = sup.validate_and_compute_panther_plan(
        {
            "command_id": "cmd-throw-frisbee",
            "action": "throw_object",
            "object_label": "frisbee",
            "speed": 0.25,
            "confirmed_by_human": True,
            "expires_at": (now + timedelta(seconds=30)).isoformat(),
        },
        now_utc=now,
    )
    assert throw_plan.master_trajectory_joints is not None
    assert throw_plan.master_trajectory_joints[0] == "joint_ra0"
    assert throw_plan.master_trajectory_joints[-1] == "joint_rf"
    assert len(throw_plan.master_trajectory_points) >= 5
    import time as _time

    t0 = _time.monotonic()
    hw.apply_panther_plan(throw_plan)
    assert len(hw.last_master_trajectory) == len(throw_plan.master_trajectory_points)
    assert "joint_ra7" in hw.joint_positions
    # The gripper follows the trajectory: closed (fully, to 1.0) after the first 1 s pause,
    # held during the swing, opened only at the release point, never at the start.
    close_t = next(t for t, _q, g in throw_plan.master_trajectory_points if g is not None)
    hw.tick_streams(t0 + close_t + 0.06)
    assert hw.right_gripper_pos == pytest.approx(1.0)
    release_t = next(t for t, _q, g in throw_plan.master_trajectory_points if g == 0.0)
    hw.tick_streams(t0 + release_t - 0.2)
    assert hw.right_gripper_pos == pytest.approx(1.0)
    hw.tick_streams(t0 + release_t + 0.1)
    assert hw.right_gripper_pos == pytest.approx(0.0)

    # 3. Classical CPU ground-frisbee ellipse detector (perception/detector/__init__.py)
    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    rgb[:, :] = (40, 45, 50)
    yy, xx = np.ogrid[:240, :320]
    white_disc = ((xx - 160) / 32.0) ** 2 + ((yy - 160) / 16.0) ** 2 <= 1.0
    rgb[white_disc] = (230, 232, 228)
    cands = detect_ground_frisbee_candidates(rgb)
    assert len(cands) >= 1
    assert abs(cands[0].cx - 160) <= 5
    assert abs(cands[0].cy - 160) <= 5



def test_throw_points_play_the_v5_csv_at_speed_after_a_1s_pause() -> None:
    from swiss_robot_bridge.local_safety import LocalSafetyRejectedError, throw_points
    from swiss_robot_bridge.local_safety import THROWS

    csv = THROWS["standard"]

    assert len(csv) == 48 and csv[-1][0] == pytest.approx(4.766071)
    pts = throw_points(2.0)
    assert pts[0] == (0.0, csv[0][1], None)  # hold the first row 1 s, then its gripper value, then 1 s more
    assert pts[1] == (1.0, csv[0][1], csv[0][2])
    assert pts[2][0] == pytest.approx(2.0)
    assert pts[-1][0] == pytest.approx(2.0 + csv[-1][0] / 2, abs=1e-3)  # twice as fast
    release = next(t for t, _q, g in pts if g == 0.0)
    assert release == pytest.approx(2.0 + next(t for t, _q, g in csv if g == 0.0) / 2, abs=1e-3)
    for bad in (0.0, -1.0, float("nan")):
        with pytest.raises(LocalSafetyRejectedError):
            throw_points(bad)


def test_throw_is_retimed_from_the_current_pose_with_its_gripper_name() -> None:
    from swiss_robot_bridge.local_safety import retime_from_current_pose, throw_points
    from swiss_robot_bridge.panther_topics import RIGHT_ARM_JOINTS

    pts = throw_points(1.0)
    names = RIGHT_ARM_JOINTS + ("joint_rf",)  # 9 names, 8 positions per point
    current = {n: q for n, q in zip(RIGHT_ARM_JOINTS, pts[0][1])}
    assert retime_from_current_pose(names, pts, current) == pts  # already at the start: no delay
    current["joint_ra3"] -= 0.9  # 0.9 rad away at 0.45 rad/s: 2 s to get there first
    assert retime_from_current_pose(names, pts, current)[0][0] == pytest.approx(2.0, abs=1e-3)


def test_throw_versions_load_from_csv_and_the_early_one_releases_0_2s_sooner() -> None:
    from swiss_robot_bridge.local_safety import THROWS, LocalSafetyRejectedError, throw_points

    std, early = THROWS["standard"], THROWS["early"]
    assert [r[0] for r in std] == [r[0] for r in early]  # same timing (standard has its own wrist since v6)
    opens = lambda rows: next(t for t, _q, g in rows if g == 0.0)  # noqa: E731
    assert opens(std) - opens(early) == pytest.approx(0.2, abs=0.03)
    assert throw_points(1.0) == throw_points(1.0, "standard")
    with pytest.raises(LocalSafetyRejectedError, match="Unknown throw version"):
        throw_points(1.0, "sideways")


@pytest.mark.parametrize(
    "body, error",
    [
        ("0,0,0,0,0,0,0,0,0\n", "need 10 numbers"),
        ("0.5,0,0,0,0,0,0,0,0,1\n1,0,0,0,0,0,0,0,0,1\n", "start at 0"),
        ("0,0,0,0,0,0,0,0,0,1\n1,0,0,0,0,0,0,0,0,2\n", "outside \\[0, 1\\]"),
        ("0,0,0,0,0,-9,0,0,0,1\n1,0,0,0,0,0,0,0,0,1\n", "joint_ra4"),
    ],
)
def test_a_bad_throw_csv_is_refused(tmp_path, body: str, error: str) -> None:
    from swiss_robot_bridge.local_safety import load_throw

    f = tmp_path / "bad.csv"
    f.write_text("# t,...\n" + body)
    with pytest.raises(ValueError, match=error):
        load_throw(f)


def test_the_bridge_takes_the_throw_version_from_the_command() -> None:
    sup = LocalSafetySupervisor()
    now = datetime.now(tz=UTC)
    cmd = {"command_id": "t1", "action": "throw_object", "object_label": "frisbee", "confirmed_by_human": True,
           "expires_at": (now + timedelta(seconds=30)).isoformat()}
    opens = lambda plan: next(t for t, _q, g in plan.master_trajectory_points if g == 0.0)  # noqa: E731
    std = sup.validate_and_compute_panther_plan(cmd, now_utc=now)
    early = sup.validate_and_compute_panther_plan({**cmd, "command_id": "t2", "throw_variant": "early"}, now_utc=now)
    assert opens(std) - opens(early) == pytest.approx(0.2, abs=0.03)
