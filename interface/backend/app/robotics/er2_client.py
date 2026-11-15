"""Gemini Robotics ER 2 client (`gemini-robotics-er-2-preview`) for UniX AI Panther spatial & multi-step planning."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from uuid import uuid4

from google import genai
from google.genai import types

from app.core.config import Settings
from app.core.logging import get_logger
from app.robotics.action_schema import (
    AllowedActionType,
    DetectedObject2D,
    RobotActionProposal,
    RoboticsReasoningResult,
    RobotStatus,
    Target3D,
)
from app.robotics.intent_parser import (
    ExplicitClauseSpec,
    parse_explicit_clause_spec,
    parse_multilingual_robot_intent,
)

logger = get_logger(__name__)

_FILLER_OBJECT_LABELS = frozenset(
    {
        "",
        "none",
        "null",
        "nil",
        "n/a",
        "na",
        "undefined",
        "unknown",
        "safe_path",
        "clear_path",
        "open_path",
        "free_path",
        "path",
        "floor",
        "ground",
        "waypoint",
        "forward",
        "forwards",
        "backward",
        "backwards",
        "left",
        "right",
        "straight",
        "ahead",
        "home",
        "base",
        "origin",
        "start",
        "target",
        "destination",
        "goal",
        "position",
        "point",
        "space",
        "free_space",
        "empty_space",
        "corridor",
        "room",
    }
)

ER2_SYSTEM_INSTRUCTION = """You are the Embodied Spatial & Task Planning Engine powered by Gemini Robotics ER 2 for a UniX AI Panther mobile manipulator.
You NEVER control motors directly and NEVER emit raw ROS topic paths, shell commands, semicolons, or executable code.
Robot hardware capabilities (mapped deterministically downstream by the local ROS 2 safety bridge):
- Holonomic 4WS mobile base: `move_to` (+x forward, -x backward, +y left strafe, -y right strafe in meters) and `rotate` (`yaw_rad` in [-3.14159, +3.14159] rad).
- Vertical torso lift: `target.z` in [0.0, 0.60] meters.
- 2-DOF pan/tilt head with Orbbec Gemini 2 RGB-D camera: `look_at` or `inspect_object` aims the head camera and captures a frame without wheel motion (`requires_confirmation=false`).
- Dual parallel grippers (left and right): `pick_object` closes the gripper to grasp (`requires_confirmation=true`), `place_object` opens the gripper to release (`requires_confirmation=true`). Mention "left" or "right" in `rationale` when a specific side is requested.

Given an operator task instruction, current robot status, and optional live Panther camera image:
1. Summarize the task/scene concisely in `scene_summary` (describe visible objects, floor clearance, and distances when a camera image is provided) and note safety observations.
2. When a camera image is provided, detect relevant objects (frisbees, bottles, obstacles, targets) with normalized [y, x] points and [ymin, xmin, ymax, xmax] boxes in [0, 1000].
3. Decompose single-step AND multi-step instructions into an ordered sequence of high-level proposals using ONLY:
   move_to, rotate, look_at, inspect_object, pick_object, place_object, return_home, stop.
4. Conventions:
   - move_to: +x = forward (m), -x = backward (m), +y = left (m), -y = right (m), z = torso lift height in [0.0, 0.6] m (default 0.0).
   - rotate: yaw_rad in [-3.14159, +3.14159] (+1.5708 = 90 deg left/CCW, -1.5708 = 90 deg right/CW, 3.14159 = 180 deg turn around).
   - return_home: target=(0.0, 0.0, 0.0).
   - speed: conservative in [0.1, 0.35] m/s (0.0 for stop).
   - Do not use semicolons or slash-prefixed ROS topic names inside `rationale` or `object_label`.
5. CRITICAL NUMERIC PRECISION RULE:
   - Whenever the operator specifies an explicit distance or angle (e.g. "move forward 5 meters" -> x=5.0, "move backward 2.5 meters" -> x=-2.5, "move left 0.5 meters" -> y=0.5, "turn right 90 degrees" -> yaw_rad=-1.5708, "turn around" -> yaw_rad=3.14159), you MUST output that EXACT numeric magnitude in a single step per clause.
   - NEVER shrink, cap, or subdivide an explicit distance like 5.0m or 3.0m down to 1.0m or 1.2m. Downstream safety validators enforce workspace bounds and LiDAR clearance deterministically.
6. STRICT FIELD HYGIENE RULE (NO PLACEHOLDER VALUES):
   - On `move_to` and `return_home`: ALWAYS set `yaw_rad` to `null`. Set `object_label` to `null` UNLESS moving toward a specific visible physical object in an attached camera image (NEVER output filler strings such as "safe_path", "clear_path", "path", "floor", "ground", "waypoint", or "none").
   - On `rotate`: ALWAYS set `target` to `null` and `object_label` to `null`.
   - On `stop`: ALWAYS set `target` to `null`, `yaw_rad` to `null`, `object_label` to `null`, and `speed` to `0.0`.
   - Examples:
     * "move forward 5 meters" -> 1 proposal: `action="move_to", target={"x": 5.0, "y": 0.0, "z": 0.0}, yaw_rad=null, object_label=null, speed=0.25, requires_confirmation=true`
     * "move forward 3 meters and turn right 90 degrees" -> 2 proposals:
       1) `action="move_to", target={"x": 3.0, "y": 0.0, "z": 0.0}, yaw_rad=null, object_label=null, speed=0.25, requires_confirmation=true`
       2) `action="rotate", target=null, yaw_rad=-1.5708, object_label=null, speed=0.2, requires_confirmation=true`
"""


class RoboticsReasoningProvider(ABC):
    """Interface for high-level visual/spatial & multi-step robotics reasoning."""

    provider_name: str = "base"

    @abstractmethod
    async def analyze_and_propose(
        self,
        instruction: str,
        robot_status: RobotStatus,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/jpeg",
        raw_user_instruction: str | None = None,
    ) -> RoboticsReasoningResult:
        raise NotImplementedError


def _spec_to_proposal(spec: ExplicitClauseSpec, *, fallback_speed: float | None = None) -> RobotActionProposal:
    act = AllowedActionType(spec.action)
    tgt = (
        Target3D(x=float(spec.x or 0.0), y=float(spec.y or 0.0), z=float(spec.z or 0.0))
        if act in {AllowedActionType.MOVE_TO, AllowedActionType.RETURN_HOME}
        else None
    )
    spd = 0.0 if act == AllowedActionType.STOP else float(fallback_speed or spec.speed)
    return RobotActionProposal(
        action=act,
        target=tgt,
        yaw_rad=spec.yaw_rad if act == AllowedActionType.ROTATE else None,
        object_label=None,
        speed=spd,
        requires_confirmation=(act != AllowedActionType.STOP),
        rationale=spec.rationale,
    )


class GeminiRoboticsER2Client(RoboticsReasoningProvider):
    """Vertex AI / Google Gen AI client for `gemini-robotics-er-2-preview` (global endpoint)."""

    provider_name = "gemini_robotics_er"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: genai.Client | None = None
        if settings.app_env != "test" and settings.gcp_project_id:
            try:
                self._get_client()
            except Exception:
                self._client = None

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not self._settings.gcp_project_id:
                raise RuntimeError("GCP_PROJECT_ID is required for GeminiRoboticsER2Client.")
            # Note: gemini-robotics-er-2-preview uses the 'global' endpoint (documented residency exception)
            self._client = genai.Client(
                vertexai=True,
                project=self._settings.gcp_project_id,
                location=self._settings.gcp_robotics_location,
            )
        return self._client

    @staticmethod
    def _build_clause_specs(
        instruction: str,
        raw_user_instruction: str | None = None,
    ) -> list[ExplicitClauseSpec | None]:
        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(instruction) if c.strip()]
        specs = [parse_explicit_clause_spec(c) for c in clauses]
        if raw_user_instruction:
            raw_clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(raw_user_instruction) if c.strip()]
            if len(raw_clauses) == len(specs):
                for idx, spec in enumerate(specs):
                    raw_spec = parse_explicit_clause_spec(raw_clauses[idx])
                    if raw_spec is not None and raw_spec.has_explicit_number:
                        if spec is None or (not spec.has_explicit_number and spec.action == raw_spec.action):
                            specs[idx] = raw_spec
        return specs

    @staticmethod
    def _format_metric_constraints_prompt(clause_specs: list[ExplicitClauseSpec | None]) -> str:
        lines: list[str] = []
        for idx, spec in enumerate(clause_specs):
            if spec is None or not spec.has_explicit_number:
                continue
            if spec.action in {"move_to", "return_home"}:
                lines.append(
                    f'- Clause {idx + 1} ("{spec.clause}"): action={spec.action}, '
                    f"exact target=(x={spec.x or 0.0:g}m, y={spec.y or 0.0:g}m, z={spec.z or 0.0:g}m), "
                    f"yaw_rad=null, object_label=null"
                )
            elif spec.action == "rotate" and spec.yaw_rad is not None:
                deg = round(math.degrees(spec.yaw_rad), 1)
                lines.append(
                    f'- Clause {idx + 1} ("{spec.clause}"): action=rotate, '
                    f"exact yaw_rad={spec.yaw_rad:.5f} ({deg:g} deg), target=null, object_label=null"
                )
            elif spec.action == "stop":
                lines.append(f'- Clause {idx + 1} ("{spec.clause}"): action=stop, speed=0.0')
        if not lines:
            return ""
        return (
            "\nExact Operator Metric Constraints (MUST preserve exact values without shrinking):\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _enforce_exact_clause_metrics(
        proposals: list[RobotActionProposal],
        clause_specs: list[ExplicitClauseSpec | None],
        *,
        has_image: bool = False,
    ) -> list[RobotActionProposal]:
        if not clause_specs:
            return proposals

        # Case 1: Pure blind kinematic instruction (no camera frame, every clause is an unambiguous kinematic command)
        if not has_image and all(spec is not None for spec in clause_specs):
            concrete_specs = [spec for spec in clause_specs if spec is not None]
            if len(proposals) == len(concrete_specs) and all(
                p.action.value == s.action for p, s in zip(proposals, concrete_specs, strict=True)
            ):
                for prop, spec in zip(proposals, concrete_specs, strict=True):
                    if spec.action == "move_to":
                        if spec.has_explicit_number or prop.target is None:
                            prop.target = Target3D(
                                x=float(spec.x or 0.0),
                                y=float(spec.y or 0.0),
                                z=float(spec.z or 0.0),
                            )
                    elif spec.action == "return_home":
                        prop.target = Target3D(x=0.0, y=0.0, z=0.0)
                    elif spec.action == "rotate" and spec.yaw_rad is not None:
                        if spec.has_explicit_number or prop.yaw_rad is None:
                            prop.yaw_rad = spec.yaw_rad
                return proposals

            # ER 2 merged, dropped, or artificially subdivided steps on a pure blind instruction -> restore exact steps
            rebuilt: list[RobotActionProposal] = []
            for idx, spec in enumerate(concrete_specs):
                donor = proposals[idx] if idx < len(proposals) and proposals[idx].action.value == spec.action else None
                rebuilt_prop = _spec_to_proposal(
                    spec,
                    fallback_speed=donor.speed if donor is not None else None,
                )
                if donor is not None and donor.rationale:
                    rebuilt_prop.rationale = donor.rationale
                rebuilt.append(rebuilt_prop)
            return rebuilt

        # Case 2: Hybrid / visual instruction -> snap only clauses that stated an explicit metric distance/angle
        explicit_specs = [s for s in clause_specs if s is not None and s.has_explicit_number]
        if not explicit_specs:
            return proposals

        prop_cursor = 0
        for spec in explicit_specs:
            for j in range(prop_cursor, len(proposals)):
                prop = proposals[j]
                if prop.action.value != spec.action:
                    continue
                if spec.action == "move_to":
                    prop.target = Target3D(
                        x=float(spec.x or 0.0),
                        y=float(spec.y or 0.0),
                        z=float(spec.z or 0.0),
                    )
                    prop.object_label = None
                elif spec.action == "return_home":
                    prop.target = Target3D(x=0.0, y=0.0, z=0.0)
                    prop.object_label = None
                elif spec.action == "rotate" and spec.yaw_rad is not None:
                    prop.yaw_rad = spec.yaw_rad
                prop_cursor = j + 1
                break
        return proposals

    def _sanitize_er2_proposals(
        self,
        proposals: list[RobotActionProposal],
        *,
        has_image: bool = False,
    ) -> None:
        for proposal in proposals:
            proposal.command_id = uuid4().hex
            if proposal.rationale:
                proposal.rationale = proposal.rationale.replace(";", ",").strip()
            if proposal.object_label is not None:
                cleaned_lbl = proposal.object_label.replace(";", ",").strip()
                norm_lbl = re.sub(r"[\s\-]+", "_", cleaned_lbl.lower())
                proposal.object_label = None if norm_lbl in _FILLER_OBJECT_LABELS else cleaned_lbl

            if proposal.action == AllowedActionType.STOP:
                proposal.speed = 0.0
                proposal.target = None
                proposal.yaw_rad = None
                proposal.object_label = None
                continue

            proposal.speed = min(
                max(float(proposal.speed or 0.25), 0.05),
                self._settings.robot_max_linear_speed_mps,
            )

            if proposal.action == AllowedActionType.MOVE_TO:
                proposal.yaw_rad = None
                if not has_image:
                    proposal.object_label = None
            elif proposal.action == AllowedActionType.RETURN_HOME:
                proposal.target = Target3D(x=0.0, y=0.0, z=0.0)
                proposal.yaw_rad = None
                proposal.object_label = None
            elif proposal.action == AllowedActionType.ROTATE:
                proposal.target = None
                proposal.object_label = None
            elif proposal.action in {AllowedActionType.INSPECT_OBJECT, AllowedActionType.PICK_OBJECT}:
                if not proposal.object_label and proposal.target is None:
                    proposal.object_label = "workspace_target"

            if proposal.yaw_rad is not None and math.isfinite(proposal.yaw_rad):
                if 3.2 < abs(proposal.yaw_rad) <= 360.0:
                    rad = math.radians(proposal.yaw_rad)
                    rad = (rad + math.pi) % (2.0 * math.pi) - math.pi
                    proposal.yaw_rad = round(rad, 5)
                else:
                    proposal.yaw_rad = max(-math.pi, min(math.pi, proposal.yaw_rad))

    async def analyze_and_propose(
        self,
        instruction: str,
        robot_status: RobotStatus,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/jpeg",
        raw_user_instruction: str | None = None,
    ) -> RoboticsReasoningResult:
        client = self._get_client()
        contents: list[types.Part | str] = []
        has_image = bool(image_bytes)

        if image_bytes:
            contents.append(types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type))

        clause_specs = self._build_clause_specs(instruction, raw_user_instruction)
        metric_constraints_block = self._format_metric_constraints_prompt(clause_specs)

        obs_note = (
            f", front_obstacle_m={robot_status.front_obstacle_distance_m:.2f}"
            if robot_status.front_obstacle_distance_m is not None
            else ""
        )
        prompt = (
            f"UniX AI Panther Status: state={robot_status.state}, battery={robot_status.battery_percent:.0f}%, "
            f"position=({robot_status.position.x:.2f}, {robot_status.position.y:.2f}, {robot_status.position.z:.2f})"
            f"{obs_note}, camera_frame_attached={has_image}, "
            f"max_linear_speed_mps={self._settings.robot_max_linear_speed_mps}, "
            f"max_workspace_radius_m={self._settings.robot_max_workspace_radius_m}.\n"
            f"Operator Task Instruction: {instruction}"
            f"{metric_constraints_block}"
        )
        contents.append(prompt)

        config = types.GenerateContentConfig(
            system_instruction=ER2_SYSTEM_INSTRUCTION,
            response_mime_type="application/json",
            response_schema=RoboticsReasoningResult,
            temperature=0.1,
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW),
        )

        response = await client.aio.models.generate_content(
            model=self._settings.gemini_robotics_model,
            contents=contents,
            config=config,
        )

        if response.parsed and isinstance(response.parsed, RoboticsReasoningResult):
            parsed = response.parsed
        else:
            parsed = RoboticsReasoningResult.model_validate_json(response.text or "{}")

        parsed.proposals = self._enforce_exact_clause_metrics(
            parsed.proposals,
            clause_specs,
            has_image=has_image,
        )
        self._sanitize_er2_proposals(parsed.proposals, has_image=has_image)
        parsed.model = self._settings.gemini_robotics_model
        parsed.provider = self.provider_name
        return parsed


_CLAUSE_SPLIT_RE = re.compile(
    r"\s+(?:and\s+then|then|and|puis|et\s+ensuite|et|und\s+dann|dann|und|dänn|und\s+dänn"
    r"|y\s+luego|y\s+despu[ée]s|luego|y(?!\s+(?:cinco|uno|una|dos|tres|cuatro|seis|siete|ocho|nueve)\b)"
    r"|a\s+pak|a\s+potom|pak|potom|a(?=\s+(?:oto[čc]|zato[čc]|jdi\b|jd[ěe]te\b|je[ďd]|popoje[ďd]|posu[ňn]|zastav|st[ůu]j|vra[ťt]|couvej|couvni)))\s+"
    r"|[，,;；]|然后|然後|接着|接著|再",
    re.IGNORECASE,
)


class MockRoboticsER2Client(RoboticsReasoningProvider):
    """Deterministic multi-step spatial reasoning mock for offline development and automated safety testing."""

    provider_name = "mock"

    async def analyze_and_propose(
        self,
        instruction: str,
        robot_status: RobotStatus,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/jpeg",
        raw_user_instruction: str | None = None,
    ) -> RoboticsReasoningResult:
        lower_instr = instruction.lower()
        mentions_frisbee = any(
            tok in lower_instr for tok in ("frisbee", "disc", "scheibe", "disque", "disco", "talíř", "飞盘", "飛盤")
        )
        primary_label = "frisbee" if mentions_frisbee else "water_bottle"

        detected = (
            [
                DetectedObject2D(
                    label=primary_label,
                    point_yx_norm=[680, 560] if mentions_frisbee else [540, 480],
                    box_2d_norm=[610, 470, 740, 650] if mentions_frisbee else [420, 430, 660, 530],
                    confidence=0.95,
                )
            ]
            if image_bytes
            else []
        )

        clauses = [c.strip() for c in _CLAUSE_SPLIT_RE.split(instruction) if c.strip()]
        proposals: list[RobotActionProposal] = []

        for clause in clauses:
            parsed_intent = parse_multilingual_robot_intent(clause)
            if parsed_intent is not None:
                t_name, args = parsed_intent
                if t_name == "emergency_stop_robot":
                    proposals.append(
                        RobotActionProposal(
                            action=AllowedActionType.STOP,
                            speed=0.0,
                            requires_confirmation=False,
                            rationale=str(args.get("reason", clause)),
                        )
                    )
                elif t_name == "propose_robot_action":
                    act = AllowedActionType(str(args["action"]))
                    tgt = (
                        Target3D(
                            x=float(args.get("x", 0.0) or 0.0),
                            y=float(args.get("y", 0.0) or 0.0),
                            z=float(args.get("z", 0.0) or 0.0),
                        )
                        if any(k in args for k in ("x", "y", "z"))
                        else None
                    )
                    proposals.append(
                        RobotActionProposal(
                            action=act,
                            target=tgt,
                            yaw_rad=float(args["yaw_rad"]) if args.get("yaw_rad") is not None else None,
                            speed=float(args.get("speed", 0.2)),
                            requires_confirmation=(act != AllowedActionType.STOP),
                            rationale=str(args.get("rationale", clause)),
                        )
                    )
                continue

            lower_c = clause.lower()
            if any(
                w in lower_c
                for w in (
                    "pick",
                    "grab",
                    "grasp",
                    "greif",
                    "heb",
                    "nimm",
                    "attrape",
                    "saisis",
                    "agarra",
                    "recoge",
                    "uchop",
                    "seber",
                    "抓",
                    "捡",
                    "撿",
                )
            ):
                proposals.append(
                    RobotActionProposal(
                        action=AllowedActionType.PICK_OBJECT,
                        object_label=primary_label,
                        target=Target3D(x=0.55, y=0.0, z=0.12),
                        speed=0.15,
                        requires_confirmation=True,
                        rationale=f"Close Panther gripper on {primary_label}: {clause[:80]}",
                    )
                )
            elif any(
                w in lower_c
                for w in (
                    "place",
                    "put down",
                    "release",
                    "drop",
                    "ableg",
                    "stell",
                    "pose",
                    "suelta",
                    "coloca",
                    "polož",
                    "poloz",
                    "放下",
                    "松开",
                    "鬆開",
                )
            ):
                proposals.append(
                    RobotActionProposal(
                        action=AllowedActionType.PLACE_OBJECT,
                        object_label=primary_label,
                        target=Target3D(x=0.50, y=0.0, z=0.10),
                        speed=0.15,
                        requires_confirmation=True,
                        rationale=f"Open Panther gripper to release {primary_label}: {clause[:80]}",
                    )
                )
            elif any(
                w in lower_c
                for w in (
                    "inspect",
                    "look",
                    "see",
                    "find",
                    "locate",
                    "camera",
                    "schau",
                    "sieh",
                    "lueg",
                    "gseh",
                    "regarde",
                    "vois",
                    "cherche",
                    "check",
                    "prüf",
                    "inspeccion",
                    "mira",
                    "ves",
                    "busca",
                    "revisa",
                    "zkontroluj",
                    "podívej",
                    "podivej",
                    "vidíš",
                    "najdi",
                    "prohlédni",
                    "prohledni",
                    "检查",
                    "檢查",
                    "查看",
                    "看",
                    "找",
                )
            ):
                proposals.append(
                    RobotActionProposal(
                        action=AllowedActionType.INSPECT_OBJECT,
                        object_label="workspace_center" if not detected else detected[0].label,
                        target=Target3D(x=0.8, y=0.0, z=0.05),
                        speed=0.1,
                        requires_confirmation=False,
                        rationale=f"Aim Panther head camera and inspect {primary_label}: {clause[:80]}",
                    )
                )

        if not proposals:
            proposals = [
                RobotActionProposal(
                    action=AllowedActionType.MOVE_TO,
                    target=Target3D(x=0.5, y=0.2, z=0.0),
                    speed=0.15,
                    requires_confirmation=True,
                    rationale="Conservative navigation step toward requested target.",
                )
            ]

        return RoboticsReasoningResult(
            scene_summary=(
                f"Live Panther camera frame analyzed: clear floor with {detected[0].label} detected ahead."
                if image_bytes and detected
                else f"Planned {len(proposals)} embodied step(s) for instruction."
            ),
            task_progress_percent=25 if image_bytes else 20,
            safety_observations=[
                "Path checked within safe workspace envelope, human confirmation required for motion."
            ],
            detected_objects=detected,
            proposals=proposals,
            model="mock-er2",
            provider=self.provider_name,
        )
