"""LLMProvider interface, Vertex AI Gemini implementation, MockLLMProvider, and Orchestrator."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Any

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

from app.assistant.memory import SessionMemoryStore, TurnMessage
from app.assistant.prompts import build_system_prompt
from app.assistant.tools import AssistantToolExecutor, get_tool_declarations
from app.core.config import Settings
from app.core.logging import get_logger
from app.robotics.intent_parser import (
    default_quick_ack,
    detect_unhandled_robot_command,
    format_er2_plan_summary,
    format_robot_tool_reply,
    should_trigger_robot_camera,
)
from app.speech.language.detection import detect_language_and_dialect_from_text

logger = get_logger(__name__)

QuickAckCallback = Callable[[str, str, str | None, str], Awaitable[None]]


class ToolCallSpec(BaseModel):
    """Normalized function call emitted by an LLMProvider."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AssistantResponse(BaseModel):
    """Structured assistant generation output."""

    text: str
    language: str = "de-DE"
    dialect: str | None = None
    quick_ack_text: str | None = None
    er2_model: str | None = None
    camera_triggered: bool = False
    camera_captured: bool = False
    camera_frame: dict[str, Any] | None = None
    detected_objects: list[dict[str, Any]] = Field(default_factory=list)
    scene_summary: str | None = None
    tool_calls: list[ToolCallSpec] = Field(default_factory=list)
    robot_commands: list[dict[str, Any]] = Field(default_factory=list)
    robot_rejections: list[str] = Field(default_factory=list)
    provider: str = "unknown"
    model: str = "unknown"


class LLMProvider(ABC):
    """Replaceable LLM conversation & tool-calling interface (README §6)."""

    provider_name: str = "base"

    @abstractmethod
    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AssistantResponse:
        raise NotImplementedError


class GeminiLLMProvider(LLMProvider):
    """Google Cloud Vertex AI Gemini provider via the official `google-genai` SDK."""

    provider_name = "google"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: genai.Client | None = None
        if settings.app_env != "test" and settings.gcp_project_id:
            self._client = genai.Client(
                vertexai=True,
                project=settings.gcp_project_id,
                location=settings.gcp_llm_location,
            )

    def _get_client(self) -> genai.Client:
        if self._client is None:
            if not self._settings.gcp_project_id:
                raise RuntimeError("GCP_PROJECT_ID must be set to use GeminiLLMProvider.")
            self._client = genai.Client(
                vertexai=True,
                project=self._settings.gcp_project_id,
                location=self._settings.gcp_llm_location,
            )
        return self._client

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AssistantResponse:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("LLM generation cancelled before request")

        client = self._get_client()
        contents: list[types.Content] = []
        for msg in messages:
            role = "model" if msg.get("role") == "assistant" else "user"
            text_val = str(msg.get("content", ""))
            contents.append(
                types.Content(
                    role=role,
                    parts=[types.Part.from_text(text=text_val)],
                )
            )

        genai_tools: list[types.Tool] | None = None
        if tools:
            func_decls = [
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters_json_schema=t.get("parameters"),
                )
                for t in tools
            ]
            genai_tools = [types.Tool(function_declarations=func_decls)]

        thinking_cfg: types.ThinkingConfig | None = None
        if "gemini-3" in self._settings.gemini_model:
            thinking_cfg = types.ThinkingConfig(thinking_level=types.ThinkingLevel.MINIMAL)

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=genai_tools,
            temperature=0.3,
            max_output_tokens=220,
            thinking_config=thinking_cfg,
        )

        response = await client.aio.models.generate_content(
            model=self._settings.gemini_model,
            contents=contents,
            config=config,
        )

        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("LLM generation interrupted")

        tool_calls: list[ToolCallSpec] = []
        if response.function_calls:
            for fc in response.function_calls:
                if fc.name:
                    tool_calls.append(
                        ToolCallSpec(
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                        )
                    )

        text_chunks: list[str] = []
        if response.candidates:
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text and not getattr(part, "thought", False):
                            text_chunks.append(part.text)
        reply_text = "".join(text_chunks).strip()

        return AssistantResponse(
            text=reply_text,
            tool_calls=tool_calls,
            provider=self.provider_name,
            model=self._settings.gemini_model,
        )


class MockLLMProvider(LLMProvider):
    """Language- and dialect-faithful mock LLM supporting Swiss German, German, French, and robot tools."""

    provider_name = "mock"

    DIALECT_GREETINGS: dict[str, str] = {
        "vs": "Grüezi üs Leukerbad! Wier sind parat — wie chan ich dier hüt hälfe?",
        "be": "Grüessech mitenand! Ig bi parat — wie chan ig öich hüt häufe?",
        "zh": "Grüezi mitenand! Mir sind parat — wie chan ich dir hüt hälfe?",
        "bs": "Sali zämme! Mir sin parat — wie kann ich dir hüt hälfe?",
        "sg": "Grüezi wohl! Mir sind parat — wie chan ich dir hüt hälfe?",
    }

    async def generate_response(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        system_instruction: str | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> AssistantResponse:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("Mock LLM interrupted")

        last_msg = messages[-1] if messages else {"content": "", "language": "de-DE", "dialect": "vs"}
        user_text = str(last_msg.get("content", ""))
        lower = user_text.lower()
        lang = str(last_msg.get("language", "de-DE"))
        dialect = str(last_msg.get("dialect") or "vs")

        # If a tool result was just injected, summarize it in the active language
        if last_msg.get("role") == "tool":
            if lang == "gsw":
                txt = "D Roboter-Afrag isch prüeft worde und wartet jetzt uf dini Bestätigung."
            elif lang.startswith("fr"):
                txt = "La proposition d'action robotique a été validée et attend votre confirmation."
            elif lang.startswith("en"):
                txt = "The robot action proposal has been safety-checked and is awaiting your confirmation."
            elif lang.startswith("es"):
                txt = "La propuesta de acción del robot ha sido verificada y espera su confirmación."
            elif lang.startswith(("cs", "cz")):
                txt = "Návrh akce robota prošel bezpečnostní kontrolou a čeká na vaše potvrzení."
            elif lang == "zh-TW":
                txt = "機器人動作建議已通過安全檢查，正在等待您的確認。"
            elif lang.startswith(("zh", "cmn")):
                txt = "机器人动作建议已通过安全检查，正在等待您的确认。"
            else:
                txt = "Der Robotervorschlag wurde sicherheitsgeprüft und wartet auf Ihre Bestätigung."
            return AssistantResponse(text=txt, language=lang, dialect=dialect, provider="mock", model="mock-gemini")

        # Detect robot tool intent when tools are offered
        if tools:
            if any(
                k in lower
                for k in (
                    "not-aus",
                    "emergency stop",
                    "estop",
                    "arrête tout",
                    "arrêt d'urgence",
                    "sofort stopp",
                    "parada de emergencia",
                    "nouzové zastavení",
                    "nouzove zastaveni",
                    "急停",
                    "紧急停止",
                    "緊急停止",
                )
            ):
                return AssistantResponse(
                    text="",
                    language=lang,
                    dialect=dialect,
                    tool_calls=[ToolCallSpec(name="emergency_stop_robot", arguments={"reason": user_text})],
                    provider="mock",
                    model="mock-gemini",
                )
            if any(
                k in lower
                for k in (
                    "status",
                    "état",
                    "batterie",
                    "battery",
                    "zustand",
                    "estado",
                    "batería",
                    "stav robota",
                    "baterie",
                    "状态",
                    "狀態",
                    "电池",
                    "電池",
                )
            ):
                return AssistantResponse(
                    text="",
                    language=lang,
                    dialect=dialect,
                    tool_calls=[ToolCallSpec(name="get_robot_status", arguments={})],
                    provider="mock",
                    model="mock-gemini",
                )
            if any(
                k in lower
                for k in (
                    "move",
                    "go forward",
                    "go back",
                    "go left",
                    "go right",
                    "step ",
                    "drive ",
                    "turn",
                    "rotate",
                    "return home",
                    "back to base",
                    "inspect",
                    "pick ",
                    "place ",
                    "fahr",
                    "geh ",
                    "gehe ",
                    "lauf",
                    "gang",
                    "füre",
                    "fürä",
                    "vorwärts",
                    "rückwärts",
                    "zrugg",
                    "dreh",
                    "träi",
                    "dräi",
                    "avance",
                    "recule",
                    "tourne",
                    "demi-tour",
                    "retourne",
                    "移动",
                    "移動",
                    "前进",
                    "前進",
                    "向前",
                    "往前",
                    "向后",
                    "往后",
                    "后退",
                    "後退",
                    "左转",
                    "左轉",
                    "右转",
                    "右轉",
                    "掉头",
                    "掉頭",
                    "转身",
                    "轉身",
                    "回到起点",
                    "回到起點",
                )
            ) or detect_unhandled_robot_command(user_text, lang) == "plan_robotics_task_with_er2":
                return AssistantResponse(
                    text="",
                    language=lang,
                    dialect=dialect,
                    tool_calls=[
                        ToolCallSpec(
                            name="plan_robotics_task_with_er2",
                            arguments={
                                "english_instruction": user_text,
                                "quick_spoken_ack": default_quick_ack(lang, dialect),
                                "needs_camera": should_trigger_robot_camera(user_text),
                            },
                        )
                    ],
                    provider="mock",
                    model="mock-gemini",
                )

        if lang == "gsw":
            greeting = self.DIALECT_GREETINGS.get(dialect, self.DIALECT_GREETINGS["vs"])
            reply = f"{greeting} Du häsch gseit: {user_text}"
        elif lang.startswith("fr"):
            reply = f"Bonjour ! Je vous écoute en français. Vous avez dit : {user_text}"
        elif lang.startswith("en"):
            reply = f"Hello! I am listening in English. You said: {user_text}"
        elif lang.startswith("es"):
            reply = f"¡Hola! Le escucho en español. Usted dijo: {user_text}"
        elif lang.startswith(("cs", "cz")):
            reply = f"Dobrý den! Poslouchám vás česky. Řekli jste: {user_text}"
        elif lang == "zh-TW":
            reply = f"您好！我正在用繁體中文為您服務。您剛才說：{user_text}"
        elif lang.startswith(("zh", "cmn")):
            reply = f"您好！我正在用中文为您服务。您刚才说：{user_text}"
        elif lang == "de-CH":
            reply = f"Grüezi! Ich antworte auf Schweizer Hochdeutsch. Sie sagten: {user_text}"
        else:
            reply = f"Guten Tag! Ich helfe Ihnen gerne weiter. Sie sagten: {user_text}"

        return AssistantResponse(
            text=reply,
            language=lang,
            dialect=dialect if lang == "gsw" else None,
            provider="mock",
            model="mock-gemini",
        )


class ConversationOrchestrator:
    """Coordinates session context, language/dialect state, Gemini Flash, and Gemini Robotics ER 2."""

    def __init__(
        self,
        settings: Settings,
        llm_provider: LLMProvider,
        memory: SessionMemoryStore,
        tool_executor: AssistantToolExecutor,
    ) -> None:
        self._settings = settings
        self._llm = llm_provider
        self._memory = memory
        self._tools = tool_executor

    async def handle_user_turn(
        self,
        session_id: str,
        user_text: str,
        language: str,
        dialect: str | None = None,
        normalized_text: str | None = None,
        cancel_event: asyncio.Event | None = None,
        on_quick_ack: QuickAckCallback | None = None,
    ) -> AssistantResponse:
        """Process one user turn with Gemini Flash (quick ack + translation) -> Gemini Robotics ER 2 (heavy planning)."""
        session = await self._memory.get_or_create(session_id)
        effective_dialect = dialect or session.active_dialect or self._settings.piper_default_dialect
        session = await self._memory.update_language_state(
            session_id=session_id,
            active_language=language,
            active_dialect=effective_dialect if language == "gsw" else session.active_dialect,
        )

        await self._memory.append_message(
            session_id,
            TurnMessage(
                role="user",
                content=user_text,
                normalized_content=normalized_text,
                language=language,
                dialect=effective_dialect if language == "gsw" else None,
            ),
        )

        emitted_robot_commands: list[dict[str, Any]] = []
        emitted_robot_rejections: list[str] = []

        history_payload = [
            {
                "role": m.role,
                "content": m.content,
                "language": m.language,
                "dialect": m.dialect,
            }
            for m in session.messages
        ]

        sys_prompt = build_system_prompt(
            active_language=language,
            active_dialect=effective_dialect,
            robotics_enabled=self._settings.robotics_enabled,
        )
        tool_defs = get_tool_declarations() if self._settings.robotics_enabled else None

        llm_resp = await self._llm.generate_response(
            messages=history_payload,
            tools=tool_defs,
            system_instruction=sys_prompt,
            cancel_event=cancel_event,
        )

        # Deterministic safety net: if Gemini Flash answered in text without calling a tool on an
        # unmistakable physical robot command (e.g. in German), guarantee delegation to ER 2 / E-Stop!
        if self._settings.robotics_enabled and not llm_resp.tool_calls:
            fallback_tool = detect_unhandled_robot_command(user_text, language) or (
                detect_unhandled_robot_command(normalized_text, "de-DE") if normalized_text else None
            )
            if fallback_tool == "emergency_stop_robot":
                llm_resp.tool_calls = [
                    ToolCallSpec(name="emergency_stop_robot", arguments={"reason": user_text})
                ]
            elif fallback_tool == "plan_robotics_task_with_er2":
                logger.info(
                    "Synthesizing plan_robotics_task_with_er2 fallback after LLM omitted tool call",
                    extra={"session_id": session_id, "language": language, "user_text": user_text[:120]},
                )
                llm_resp.tool_calls = [
                    ToolCallSpec(
                        name="plan_robotics_task_with_er2",
                        arguments={
                            "english_instruction": normalized_text or user_text,
                            "quick_spoken_ack": default_quick_ack(language, effective_dialect),
                            "needs_camera": should_trigger_robot_camera(user_text)
                            or should_trigger_robot_camera(normalized_text or ""),
                        },
                    )
                ]

        captured_quick_ack: str | None = None
        er2_model_used: str | None = None
        er2_camera_triggered: bool = False
        er2_camera_captured: bool = False
        er2_camera_frame: dict[str, Any] | None = None
        er2_detected_objects: list[dict[str, Any]] = []
        er2_scene_summary: str | None = None

        if llm_resp.tool_calls:
            used_er2_planner = False
            quick_ack_emitted = False
            er2_validated_all: list[dict[str, Any]] = []
            er2_rejected_all: list[str] = []
            other_direct_replies: list[str] = []
            other_tool_calls_count = 0
            tool_summaries: list[str] = []

            for call in llm_resp.tool_calls:
                if call.name == "plan_robotics_task_with_er2":
                    used_er2_planner = True
                    raw_ack = str(call.arguments.get("quick_spoken_ack") or llm_resp.text or "").strip()
                    ack_text = raw_ack or default_quick_ack(language, effective_dialect)
                    eng_instr = str(call.arguments.get("english_instruction") or user_text).strip()
                    if captured_quick_ack is None:
                        captured_quick_ack = ack_text

                    # Stream quick spoken ack immediately over WebSocket BEFORE awaiting ER 2 heavy lifting!
                    if on_quick_ack is not None and not quick_ack_emitted:
                        quick_ack_emitted = True
                        await on_quick_ack(
                            ack_text,
                            language,
                            effective_dialect if language == "gsw" else None,
                            eng_instr,
                        )

                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError("Turn interrupted before Gemini Robotics ER 2 planning")

                    result_dict, validated_cmds = await self._tools.execute(
                        call.name,
                        {**call.arguments, "_raw_user_text": normalized_text or user_text},
                    )

                    if cancel_event is not None and cancel_event.is_set():
                        raise asyncio.CancelledError("Turn interrupted after Gemini Robotics ER 2 planning")

                    er2_validated_all.extend(validated_cmds)
                    emitted_robot_commands.extend(validated_cmds)
                    rej_list = [str(r) for r in result_dict.get("rejected_reasons", [])]
                    er2_rejected_all.extend(rej_list)
                    emitted_robot_rejections.extend(rej_list)
                    if result_dict.get("er2_model"):
                        er2_model_used = str(result_dict["er2_model"])
                    if result_dict.get("camera_triggered"):
                        er2_camera_triggered = True
                    if result_dict.get("camera_captured"):
                        er2_camera_captured = True
                    if isinstance(result_dict.get("camera_frame"), dict):
                        er2_camera_frame = result_dict["camera_frame"]
                    if isinstance(result_dict.get("detected_objects"), list):
                        er2_detected_objects.extend(result_dict["detected_objects"])
                    if result_dict.get("scene_summary"):
                        er2_scene_summary = str(result_dict["scene_summary"])
                    compact_res = {k: v for k, v in result_dict.items() if k != "camera_frame"}
                    tool_summaries.append(f"Tool {call.name} result: {compact_res}")
                    continue

                other_tool_calls_count += 1
                result_dict, validated_cmds = await self._tools.execute(call.name, call.arguments)
                if call.name == "switch_voice_dialect" and "switched_dialect" in result_dict:
                    effective_dialect = str(result_dict["switched_dialect"])
                    await self._memory.update_language_state(
                        session_id=session_id,
                        active_language="gsw",
                        active_dialect=effective_dialect,
                    )
                emitted_robot_commands.extend(validated_cmds)
                if result_dict.get("rejected_reason"):
                    emitted_robot_rejections.append(str(result_dict["rejected_reason"]))
                if result_dict.get("rejected_reasons"):
                    emitted_robot_rejections.extend(str(r) for r in result_dict["rejected_reasons"])

                formatted = format_robot_tool_reply(
                    tool_name=call.name,
                    arguments=call.arguments,
                    result=result_dict,
                    language=language,
                    dialect=effective_dialect,
                )
                if formatted:
                    other_direct_replies.append(formatted)
                tool_summaries.append(f"Tool {call.name} result: {result_dict}")

            if used_er2_planner:
                plan_summary = format_er2_plan_summary(
                    er2_validated_all,
                    er2_rejected_all,
                    language,
                    scene_summary=er2_scene_summary,
                    camera_captured=er2_camera_captured,
                )
                extra = f" {' '.join(other_direct_replies)}" if other_direct_replies else ""
                if quick_ack_emitted:
                    final_text = f"{plan_summary}{extra}".strip()
                    memory_text = f"{captured_quick_ack or ''} {final_text}".strip()
                else:
                    final_text = f"{captured_quick_ack or ''} {plan_summary}{extra}".strip()
                    memory_text = final_text
            elif other_direct_replies and len(other_direct_replies) == other_tool_calls_count:
                final_text = " ".join(other_direct_replies)
                memory_text = final_text
            elif llm_resp.text.strip():
                final_text = llm_resp.text.strip()
                memory_text = final_text
            else:
                history_payload.append(
                    {
                        "role": "tool",
                        "content": "\n".join(tool_summaries),
                        "language": language,
                        "dialect": effective_dialect,
                    }
                )
                followup = await self._llm.generate_response(
                    messages=history_payload,
                    tools=None,
                    system_instruction=sys_prompt,
                    cancel_event=cancel_event,
                )
                final_text = followup.text or llm_resp.text
                memory_text = final_text
        else:
            final_text = llm_resp.text
            memory_text = final_text

        # Align outgoing TTS language/dialect with the actual reply language (e.g., if user asked for a translation)
        reply_lang = language
        reply_dialect = effective_dialect
        if not llm_resp.tool_calls and final_text.strip():
            det_reply_lang, det_reply_dial = detect_language_and_dialect_from_text(
                final_text,
                hint_language=language,
                default_dialect=effective_dialect,
            )
            if det_reply_lang and det_reply_lang != "auto":
                if language in {"de-DE", "de-CH"} and det_reply_lang == "gsw":
                    # Never silently flip a Standard German reply into SwissDial Piper TTS
                    pass
                else:
                    reply_lang = det_reply_lang
                    if det_reply_dial:
                        reply_dialect = det_reply_dial

        await self._memory.append_message(
            session_id,
            TurnMessage(
                role="assistant",
                content=memory_text,
                language=reply_lang,
                dialect=reply_dialect if reply_lang == "gsw" else None,
            ),
        )

        model_label = f"{llm_resp.model} + {er2_model_used}" if er2_model_used else llm_resp.model
        return AssistantResponse(
            text=final_text,
            language=reply_lang,
            dialect=reply_dialect if reply_lang == "gsw" else None,
            quick_ack_text=captured_quick_ack,
            er2_model=er2_model_used,
            camera_triggered=er2_camera_triggered,
            camera_captured=er2_camera_captured,
            camera_frame=er2_camera_frame,
            detected_objects=er2_detected_objects,
            scene_summary=er2_scene_summary,
            tool_calls=llm_resp.tool_calls,
            robot_commands=emitted_robot_commands,
            robot_rejections=emitted_robot_rejections,
            provider=llm_resp.provider,
            model=model_label,
        )
