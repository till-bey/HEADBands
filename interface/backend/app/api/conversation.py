"""Session conversation REST endpoints supporting multilingual text turns and interruption."""

from __future__ import annotations

import base64
from typing import Annotated, Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.container import ServiceContainer, get_container
from app.core.security import require_api_auth
from app.speech.language.detection import (
    detect_language_and_dialect_from_text,
    normalize_language_code,
)
from app.speech.language.normalization import build_swiss_german_gloss

router = APIRouter(
    prefix="/api/v1/conversations",
    tags=["conversation"],
    dependencies=[Depends(require_api_auth)],
)


class TextTurnRequest(BaseModel):
    """Text message request with optional language/dialect selection and TTS toggle."""

    text: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="auto")
    dialect: str | None = Field(default=None)
    synthesize_audio: bool = Field(default=False)
    voice: str | None = Field(default=None)


@router.post("/{session_id}/messages")
async def post_conversation_message(
    session_id: str,
    body: TextTurnRequest,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    """Handle a multilingual conversation turn, auto-switching language/dialect as needed."""
    cancel_event = container.memory.start_new_turn_cancel_token(session_id)
    session = await container.memory.get_or_create(session_id)

    hint_lang = (
        body.language
        if body.language not in {"auto", ""}
        else (session.active_language if session.messages else "auto")
    )
    detected_lang, detected_dialect = detect_language_and_dialect_from_text(
        body.text,
        hint_language=hint_lang,
        default_dialect=body.dialect or session.active_dialect,
    )
    target_lang = (
        normalize_language_code(body.language)
        if body.language not in {"auto", ""}
        else detected_lang
    )
    target_dialect = body.dialect or detected_dialect or session.active_dialect
    gloss = build_swiss_german_gloss(body.text) if target_lang == "gsw" else None

    assistant_resp = await container.orchestrator.handle_user_turn(
        session_id=session_id,
        user_text=body.text,
        language=target_lang,
        dialect=target_dialect if target_lang == "gsw" else None,
        normalized_text=gloss,
        cancel_event=cancel_event,
    )

    audio_b64: str | None = None
    tts_provider_used: str | None = None
    if body.synthesize_audio and assistant_resp.text:
        tts_route = container.speech_router.select_tts_route(
            language=assistant_resp.language,
            dialect=assistant_resp.dialect,
            requested_voice=body.voice,
        )
        wav_bytes = await tts_route.provider.synthesize(
            text=assistant_resp.text,
            language=tts_route.language,
            voice=tts_route.voice,
            dialect=tts_route.dialect,
            speaking_rate=container.settings.tts_speaking_rate,
            cancel_event=cancel_event,
        )
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        tts_provider_used = tts_route.provider.provider_name

    return {
        "session_id": session_id,
        "detected_language": target_lang,
        "detected_dialect": assistant_resp.dialect,
        "normalized_user_text": gloss,
        "assistant": assistant_resp.model_dump(mode="json"),
        "audio_wav_base64": audio_b64,
        "tts_provider": tts_provider_used,
    }


@router.get("/{session_id}")
async def get_conversation(
    session_id: str,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    session = await container.memory.get_or_create(session_id)
    return session.model_dump(mode="json")


@router.post("/{session_id}/interrupt")
async def interrupt_conversation(
    session_id: str,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    cancelled = container.memory.interrupt_turn(session_id)
    return {"session_id": session_id, "interrupted": cancelled}


@router.delete("/{session_id}")
async def delete_conversation(
    session_id: str,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> dict[str, Any]:
    await container.memory.delete_session(session_id)
    deleted_blobs = await container.audio_storage.delete_session_recordings(session_id)
    return {"session_id": session_id, "deleted": True, "deleted_audio_objects": deleted_blobs}
