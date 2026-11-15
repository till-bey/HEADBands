"""Strict allowlisted schemas for robot action proposals, validation, camera snapshots, and Panther telemetry."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_PANTHER_ACTIVE_TOPICS: tuple[str, ...] = (
    "/cmd_vel",
    "/preempt_teleop",
    "/head_forward_position_controller/commands",
    "/lifting_controller/commands",
    "/left_gripper/gripper_command",
    "/right_gripper/gripper_command",
    "/camera/color/image_raw",
)


class AllowedActionType(StrEnum):
    """Explicit allowlist of permitted high-level robot actions (Rule 3: no arbitrary commands)."""

    MOVE_TO = "move_to"
    ROTATE = "rotate"
    LOOK_AT = "look_at"
    INSPECT_OBJECT = "inspect_object"
    PICK_OBJECT = "pick_object"
    PLACE_OBJECT = "place_object"
    RETURN_HOME = "return_home"
    STOP = "stop"


DANGEROUS_ACTIONS_REQUIRING_CONFIRMATION: frozenset[AllowedActionType] = frozenset(
    {
        AllowedActionType.MOVE_TO,
        AllowedActionType.ROTATE,
        AllowedActionType.PICK_OBJECT,
        AllowedActionType.PLACE_OBJECT,
        AllowedActionType.RETURN_HOME,
    }
)


class RobotOperationalState(StrEnum):
    """Reported local robot state (Rule 10: reject commands when in any unsafe state)."""

    READY = "ready"
    BUSY = "busy"
    ESTOP = "estop"
    FAULT = "fault"
    LOW_BATTERY = "low_battery"
    UNSAFE = "unsafe"
    DISCONNECTED = "disconnected"


class Target3D(BaseModel):
    """Workspace-relative target coordinates in meters."""

    model_config = ConfigDict(extra="forbid")

    x: float = Field(description="Forward/backward position in meters relative to safe origin.")
    y: float = Field(description="Left/right position in meters relative to safe origin.")
    z: float = Field(default=0.0, description="Vertical height in meters (>= 0; Panther lift [0, 0.6]m).")


class DetectedObject2D(BaseModel):
    """Spatial object detection returned by Gemini Robotics ER 2."""

    model_config = ConfigDict(extra="forbid")

    label: str
    point_yx_norm: list[int] | None = Field(
        default=None,
        min_length=2,
        max_length=2,
        description="Normalized [y, x] coordinates in [0, 1000] image space.",
    )
    box_2d_norm: list[int] | None = Field(
        default=None,
        min_length=4,
        max_length=4,
        description="Normalized [ymin, xmin, ymax, xmax] bounding box in [0, 1000].",
    )
    confidence: float = Field(default=0.85, ge=0.0, le=1.0)


class CameraFrameSnapshot(BaseModel):
    """On-demand camera snapshot captured from `/camera/color/image_raw` on the Panther ROS 2 bridge."""

    model_config = ConfigDict(extra="ignore")

    request_id: str
    jpeg_base64: str = Field(min_length=4)
    mime_type: str = "image/jpeg"
    width: int = Field(default=640, ge=1, le=8192)
    height: int = Field(default=480, ge=1, le=8192)
    source: str = "/camera/color/image_raw"
    center_depth_m: float | None = None
    driver_triggered: bool = False
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))


class RobotActionProposal(BaseModel):
    """Application-level proposal produced by ER 2 or assistant tools (not yet approved/executed)."""

    model_config = ConfigDict(extra="forbid")

    command_id: str = Field(default_factory=lambda: uuid4().hex)
    action: AllowedActionType
    target: Target3D | None = None
    yaw_rad: float | None = Field(default=None, description="Target rotation angle in radians.")
    object_label: str | None = Field(default=None, max_length=120)
    speed: float = Field(default=0.15, description="Requested linear speed in m/s.")
    requires_confirmation: bool = True
    rationale: str = Field(default="", max_length=600)


class ValidatedRobotCommand(BaseModel):
    """Safety-validated command envelope with mandatory expiration timestamp."""

    model_config = ConfigDict(extra="forbid")

    command_id: str
    action: AllowedActionType
    target: Target3D | None = None
    yaw_rad: float | None = None
    object_label: str | None = None
    speed: float
    requires_confirmation: bool
    confirmed_by_human: bool = False
    confirmed_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    expires_at: datetime
    rationale: str = ""


class RoboticsReasoningResult(BaseModel):
    """Structured output of Gemini Robotics ER 2 visual/spatial reasoning."""

    scene_summary: str
    task_progress_percent: int = Field(default=0, ge=0, le=100)
    safety_observations: list[str] = Field(default_factory=list)
    detected_objects: list[DetectedObject2D] = Field(default_factory=list)
    proposals: list[RobotActionProposal] = Field(default_factory=list)
    model: str = "gemini-robotics-er-2-preview"
    provider: str = "gemini_robotics_er"


class RobotStatus(BaseModel):
    """Current local UniX AI Panther robot state, camera readiness, and watchdog telemetry snapshot."""

    robot_id: str = "leukerbad-bot-01"
    state: RobotOperationalState = RobotOperationalState.READY
    estop_active: bool = False
    battery_percent: float = Field(default=95.0, ge=0.0, le=100.0)
    position: Target3D = Field(default_factory=lambda: Target3D(x=0.0, y=0.0, z=0.0))
    bridge_connected: bool = False
    last_heartbeat_utc: datetime | None = None
    active_command_id: str | None = None
    ros_domain_id: int = 15
    camera_ready: bool = False
    camera_driver_active: bool = False
    camera_source: str | None = None
    front_obstacle_distance_m: float | None = None
    lift_height_m: float | None = None
    left_gripper_pos: float | None = None
    right_gripper_pos: float | None = None
    head_joints_rad: list[float] | None = None
    active_topics: list[str] = Field(default_factory=lambda: list(DEFAULT_PANTHER_ACTIVE_TOPICS))
    capabilities: list[str] = Field(
        default_factory=lambda: [a.value for a in AllowedActionType]
    )


class RobotTelemetryEvent(BaseModel):
    """Telemetry frame sent from the local UniX AI Panther ROS 2 bridge."""

    model_config = ConfigDict(extra="forbid")

    robot_id: str = "leukerbad-bot-01"
    state: RobotOperationalState
    estop_active: bool = False
    battery_percent: float = Field(default=100.0, ge=0.0, le=100.0)
    position: Target3D = Field(default_factory=lambda: Target3D(x=0.0, y=0.0, z=0.0))
    active_command_id: str | None = None
    last_completed_command_id: str | None = None
    last_error: str | None = None
    ros_domain_id: int = 15
    camera_ready: bool = False
    camera_driver_active: bool = False
    camera_source: str | None = None
    front_obstacle_distance_m: float | None = None
    lift_height_m: float | None = None
    left_gripper_pos: float | None = None
    right_gripper_pos: float | None = None
    head_joints_rad: list[float] | None = None
    active_topics: list[str] = Field(default_factory=list)
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
