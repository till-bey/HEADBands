"""Real-time Voice REST & full-duplex WebSocket endpoints (16kHz PCM VAD, streaming TTS, barge-in)."""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Annotated, Any
from uuid import uuid4

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Response,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from app.container import ServiceContainer, get_container
from app.core.logging import get_logger
from app.core.security import require_api_auth, verify_constant_time_token
from app.speech.asr.base import Transcription
from app.speech.audio_utils import StreamingAudioBuffer, pcm16_to_wav
from app.speech.language.detection import (
    detect_language_and_dialect_from_text,
    normalize_language_code,
)
from app.speech.language.normalization import build_swiss_german_gloss

logger = get_logger(__name__)

router = APIRouter(tags=["voice"])


class SynthesizeSpeechRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    language: str = Field(default="de-DE")
    dialect: str | None = Field(default=None)
    voice: str | None = Field(default=None)
    speaking_rate: float = Field(default=1.0, ge=0.5, le=2.0)


@router.post("/api/v1/voice/transcribe", dependencies=[Depends(require_api_auth)])
async def transcribe_audio_endpoint(
    container: Annotated[ServiceContainer, Depends(get_container)],
    audio: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form()] = "auto",
    session_id: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Transcribe uploaded WAV or 16kHz 16-bit mono PCM audio."""
    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio is empty.")

    hint: str | None = None
    if session_id:
        sess = await container.memory.get_or_create(session_id)
        hint = sess.active_language

    transcription = await container.speech_router.transcribe_routed(
        audio=raw_bytes,
        requested_language=language,
        session_language_hint=hint,
        is_partial=False,
    )

    if session_id:
        await container.audio_storage.save_audio_if_enabled(
            audio_bytes=pcm16_to_wav(raw_bytes),
            session_id=session_id,
            language=transcription.language,
        )

    return transcription.model_dump(mode="json")


@router.post("/api/v1/voice/synthesize", dependencies=[Depends(require_api_auth)])
async def synthesize_speech_endpoint(
    body: SynthesizeSpeechRequest,
    container: Annotated[ServiceContainer, Depends(get_container)],
) -> Response:
    """Synthesize playable WAV audio using language- and dialect-aware TTS routing."""
    route = container.speech_router.select_tts_route(
        language=body.language,
        dialect=body.dialect,
        requested_voice=body.voice,
    )
    wav_bytes = await route.provider.synthesize(
        text=body.text,
        language=route.language,
        voice=route.voice,
        dialect=route.dialect,
        speaking_rate=body.speaking_rate,
    )
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-TTS-Provider": route.provider.provider_name,
            "X-TTS-Language": route.language,
            "X-TTS-Dialect": route.dialect or "",
        },
    )


@router.post("/api/v1/voice/turn", dependencies=[Depends(require_api_auth)])
async def full_voice_turn_endpoint(
    container: Annotated[ServiceContainer, Depends(get_container)],
    audio: Annotated[UploadFile, File(...)],
    language: Annotated[str, Form()] = "auto",
    dialect: Annotated[str | None, Form()] = None,
    voice: Annotated[str | None, Form()] = None,
    session_id: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Complete voice turn: Audio -> ASR -> Gemini Orchestrator -> Language-aware TTS."""
    sid = session_id or f"sess-{uuid4().hex[:12]}"
    raw_bytes = await audio.read()
    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded audio is empty.")

    cancel_event = container.memory.start_new_turn_cancel_token(sid)
    sess = await container.memory.get_or_create(sid)

    transcription = await container.speech_router.transcribe_routed(
        audio=raw_bytes,
        requested_language=language,
        session_language_hint=sess.active_language if sess.messages else None,
        is_partial=False,
    )

    await container.audio_storage.save_audio_if_enabled(
        audio_bytes=pcm16_to_wav(raw_bytes),
        session_id=sid,
        language=transcription.language,
    )

    if not transcription.text.strip():
        return {
            "session_id": sid,
            "transcription": transcription.model_dump(mode="json"),
            "assistant": None,
            "audio_wav_base64": None,
        }

    chosen_dialect = dialect or transcription.detected_dialect or sess.active_dialect
    assistant_resp = await container.orchestrator.handle_user_turn(
        session_id=sid,
        user_text=transcription.text,
        language=transcription.language,
        dialect=chosen_dialect if transcription.language == "gsw" else None,
        normalized_text=transcription.normalized_text,
        cancel_event=cancel_event,
    )

    tts_route = container.speech_router.select_tts_route(
        language=assistant_resp.language,
        dialect=assistant_resp.dialect,
        requested_voice=voice,
    )
    wav_bytes = await tts_route.provider.synthesize(
        text=assistant_resp.text,
        language=tts_route.language,
        voice=tts_route.voice,
        dialect=tts_route.dialect,
        speaking_rate=container.settings.tts_speaking_rate,
        cancel_event=cancel_event,
    )

    return {
        "session_id": sid,
        "transcription": transcription.model_dump(mode="json"),
        "assistant": assistant_resp.model_dump(mode="json"),
        "audio_wav_base64": base64.b64encode(wav_bytes).decode("ascii") if wav_bytes else None,
        "tts_provider": tts_route.provider.provider_name,
        "tts_voice": tts_route.voice,
    }


@router.websocket("/ws/voice")
async def voice_streaming_websocket(websocket: WebSocket) -> None:
    """Full-duplex low-latency voice WebSocket supporting 16kHz PCM VAD, partial ASR, streaming TTS & barge-in."""
    container: ServiceContainer = websocket.app.state.container
    if container.settings.api_auth_token:
        token = websocket.headers.get("authorization") or websocket.query_params.get("token")
        if not verify_constant_time_token(token, container.settings.api_auth_token):
            await websocket.close(code=4401, reason="Unauthorized")
            return

    await websocket.accept()
    session_id = websocket.query_params.get("session_id") or f"ws-{uuid4().hex[:12]}"
    selected_language = websocket.query_params.get("language") or "auto"
    selected_dialect = websocket.query_params.get("dialect") or container.settings.piper_default_dialect
    selected_voice: str | None = websocket.query_params.get("voice")
    speaking_rate: float = container.settings.tts_speaking_rate

    audio_buffer = StreamingAudioBuffer()
    active_turn_task: asyncio.Task[None] | None = None
    partial_asr_task: asyncio.Task[None] | None = None
    turn_transcript_committed: bool = False
    last_partial_result: Transcription | None = None

    await websocket.send_json(
        {
            "type": "session_ready",
            "session_id": session_id,
            "language": selected_language,
            "dialect": selected_dialect,
            "audio_format": {"sample_rate_hz": 16000, "channels": 1, "bits": 16},
        }
    )

    async def _cancel_partial_task() -> None:
        nonlocal partial_asr_task
        if partial_asr_task is not None and not partial_asr_task.done():
            partial_asr_task.cancel()
            try:
                await partial_asr_task
            except (asyncio.CancelledError, Exception):
                pass
        partial_asr_task = None

    async def _run_partial_asr(pcm_snapshot: bytes, req_lang: str) -> None:
        nonlocal last_partial_result
        # Avoid queuing partial snapshots against remote CPU faster-whisper so final GSW EoU stays unblocked
        if (
            req_lang == "gsw"
            and container.settings.stt_swiss_german_provider == "faster_whisper"
            and container.settings.asr_endpoint
        ):
            return
        try:
            partial = await container.speech_router.transcribe_routed(
                audio=pcm_snapshot,
                requested_language=req_lang,
                is_partial=True,
            )
            if partial.text.strip():
                last_partial_result = partial
                await websocket.send_json(
                    {"type": "partial_transcript", "text": partial.text, "language": partial.language}
                )
        except (asyncio.CancelledError, Exception):
            pass

    async def _cancel_active_turn(reason: str) -> None:
        nonlocal active_turn_task, turn_transcript_committed
        await _cancel_partial_task()
        container.memory.interrupt_turn(session_id)
        if active_turn_task is not None and not active_turn_task.done():
            active_turn_task.cancel()
            try:
                await active_turn_task
            except (asyncio.CancelledError, Exception):
                pass
            await websocket.send_json({"type": "interrupted", "reason": reason})
        active_turn_task = None
        turn_transcript_committed = False

    async def _process_utterance_audio(
        pcm_bytes: bytes,
        fallback_partial: Transcription | None = None,
    ) -> None:
        nonlocal turn_transcript_committed
        turn_transcript_committed = False
        cancel_event = container.memory.start_new_turn_cancel_token(session_id)
        sess = await container.memory.get_or_create(session_id)
        try:
            transcription = await container.speech_router.transcribe_routed(
                audio=pcm_bytes,
                requested_language=selected_language,
                session_language_hint=sess.active_language if sess.messages else None,
                is_partial=False,
            )
            if not transcription.text.strip() and fallback_partial is not None and fallback_partial.text.strip():
                recovered_gloss = (
                    build_swiss_german_gloss(fallback_partial.text)
                    if fallback_partial.language == "gsw"
                    else fallback_partial.normalized_text
                )
                transcription = fallback_partial.model_copy(
                    update={
                        "is_partial": False,
                        "normalized_text": recovered_gloss,
                    }
                )

            await websocket.send_json(
                {
                    "type": "final_transcript",
                    "transcription": transcription.model_dump(mode="json"),
                }
            )
            turn_transcript_committed = True
            if not transcription.text.strip():
                return

            await container.audio_storage.save_audio_if_enabled(
                audio_bytes=pcm16_to_wav(pcm_bytes),
                session_id=session_id,
                language=transcription.language,
            )

            await _generate_and_stream_reply(
                user_text=transcription.text,
                turn_lang=transcription.language,
                turn_dialect=transcription.detected_dialect or selected_dialect,
                normalized_text=transcription.normalized_text,
                cancel_event=cancel_event,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Error processing voice utterance", extra={"session_id": session_id, "error": str(exc)})
            await websocket.send_json({"type": "error", "message": str(exc)})

    async def _generate_and_stream_reply(
        user_text: str,
        turn_lang: str,
        turn_dialect: str | None,
        normalized_text: str | None,
        cancel_event: asyncio.Event,
    ) -> None:
        chunk_idx = 0
        ack_tts_task: asyncio.Task[None] | None = None

        async def _stream_tts_segment(
            segment_text: str,
            seg_lang: str,
            seg_dialect: str | None,
        ) -> None:
            nonlocal chunk_idx
            if not segment_text.strip():
                return
            tts_route = container.speech_router.select_tts_route(
                language=seg_lang,
                dialect=seg_dialect,
                requested_voice=selected_voice,
            )
            async for sentence_text, wav_chunk in tts_route.provider.synthesize_stream(
                text=segment_text,
                language=tts_route.language,
                voice=tts_route.voice,
                dialect=tts_route.dialect,
                speaking_rate=speaking_rate,
                cancel_event=cancel_event,
            ):
                await websocket.send_json(
                    {
                        "type": "tts_audio_chunk",
                        "chunk_index": chunk_idx,
                        "sentence": sentence_text,
                        "language": tts_route.language,
                        "dialect": tts_route.dialect,
                        "provider": tts_route.provider.provider_name,
                        "audio_wav_base64": base64.b64encode(wav_chunk).decode("ascii"),
                    }
                )
                chunk_idx += 1

        async def _on_quick_ack(
            ack_text: str,
            ack_lang: str,
            ack_dial: str | None,
            translated_task: str,
        ) -> None:
            nonlocal ack_tts_task
            await websocket.send_json(
                {
                    "type": "quick_ack",
                    "text": ack_text,
                    "language": ack_lang,
                    "dialect": ack_dial,
                    "translated_task": translated_task,
                    "model": container.settings.gemini_model,
                }
            )
            ack_tts_task = asyncio.create_task(_stream_tts_segment(ack_text, ack_lang, ack_dial))

        try:
            assistant_resp = await container.orchestrator.handle_user_turn(
                session_id=session_id,
                user_text=user_text,
                language=turn_lang,
                dialect=turn_dialect if turn_lang == "gsw" else None,
                normalized_text=normalized_text,
                cancel_event=cancel_event,
                on_quick_ack=_on_quick_ack,
            )

            if ack_tts_task is not None:
                await ack_tts_task
                ack_tts_task = None

            await websocket.send_json(
                {
                    "type": "assistant_response",
                    "assistant": assistant_resp.model_dump(mode="json"),
                }
            )

            if assistant_resp.text.strip():
                await _stream_tts_segment(
                    assistant_resp.text,
                    assistant_resp.language,
                    assistant_resp.dialect,
                )

            if chunk_idx > 0 or assistant_resp.text.strip():
                await websocket.send_json({"type": "tts_stream_complete", "chunks": chunk_idx})
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.error("Error generating assistant reply", extra={"session_id": session_id, "error": str(exc)})
            await websocket.send_json({"type": "error", "message": str(exc)})
        finally:
            if ack_tts_task is not None and not ack_tts_task.done():
                ack_tts_task.cancel()
                try:
                    await ack_tts_task
                except (asyncio.CancelledError, Exception):
                    pass

    try:
        while True:
            ws_message = await websocket.receive()
            if ws_message.get("type") == "websocket.disconnect":
                break

            if "bytes" in ws_message and ws_message["bytes"] is not None:
                pcm_frame = ws_message["bytes"]
                vad_event = audio_buffer.push_pcm(pcm_frame)

                if vad_event.speech_started:
                    await websocket.send_json({"type": "vad_state", "state": "speech_started", "rms": round(vad_event.rms, 4)})

                if (
                    vad_event.barge_in_confirmed
                    and turn_transcript_committed
                    and active_turn_task is not None
                    and not active_turn_task.done()
                ):
                    # Sustained speech (>=280ms) after previous transcript was committed interrupts TTS/reply
                    await _cancel_active_turn("barge_in_speech_detected")

                if vad_event.should_emit_partial and vad_event.utterance_pcm:
                    # Run partial ASR in the background so websocket.receive() never stalls incoming PCM/EoU!
                    if partial_asr_task is None or partial_asr_task.done():
                        partial_asr_task = asyncio.create_task(
                            _run_partial_asr(vad_event.utterance_pcm, selected_language)
                        )

                if vad_event.end_of_utterance and vad_event.utterance_pcm:
                    await _cancel_partial_task()
                    captured_partial = last_partial_result
                    last_partial_result = None
                    await websocket.send_json({"type": "vad_state", "state": "end_of_utterance"})
                    if active_turn_task is not None and not active_turn_task.done():
                        if not turn_transcript_committed:
                            # Let in-flight ASR finish posting final_transcript before starting next turn
                            try:
                                await asyncio.wait_for(asyncio.shield(active_turn_task), timeout=2.5)
                            except (TimeoutError, asyncio.CancelledError, Exception):
                                await _cancel_active_turn("new_utterance")
                        else:
                            await _cancel_active_turn("new_utterance")
                    active_turn_task = asyncio.create_task(
                        _process_utterance_audio(vad_event.utterance_pcm, captured_partial)
                    )

            elif "text" in ws_message and ws_message["text"] is not None:
                ctrl = json.loads(ws_message["text"])
                ctrl_type = ctrl.get("type")

                if ctrl_type == "config":
                    if "language" in ctrl:
                        selected_language = str(ctrl["language"])
                    if ctrl.get("dialect"):
                        selected_dialect = str(ctrl["dialect"])
                    if "voice" in ctrl:
                        selected_voice = ctrl["voice"] or None
                    if "speaking_rate" in ctrl:
                        speaking_rate = max(0.5, min(2.0, float(ctrl["speaking_rate"])))
                    await websocket.send_json(
                        {
                            "type": "config_updated",
                            "language": selected_language,
                            "dialect": selected_dialect,
                            "voice": selected_voice,
                            "speaking_rate": speaking_rate,
                        }
                    )

                elif ctrl_type == "interrupt":
                    audio_buffer.reset()
                    last_partial_result = None
                    await _cancel_active_turn("client_interrupt")

                elif ctrl_type == "end_of_utterance":
                    buffered_speech_ms = audio_buffer.speech_duration_ms
                    flushed = audio_buffer.flush()
                    await _cancel_partial_task()
                    captured_partial = last_partial_result
                    last_partial_result = None
                    turn_busy = active_turn_task is not None and not active_turn_task.done()
                    # When user clicks "Stop Microphone" after VAD already dispatched active_turn_task,
                    # ignore tiny trailing click/breath frames (<350ms) so the real turn finishes cleanly!
                    if flushed and (not turn_busy or buffered_speech_ms >= 350.0):
                        if turn_busy:
                            await _cancel_active_turn("manual_end_of_utterance")
                        active_turn_task = asyncio.create_task(
                            _process_utterance_audio(flushed, captured_partial)
                        )

                elif ctrl_type == "text_input":
                    text_val = str(ctrl.get("text", "")).strip()
                    if text_val:
                        await _cancel_active_turn("new_text_turn")
                        turn_transcript_committed = True
                        cancel_ev = container.memory.start_new_turn_cancel_token(session_id)
                        sess = await container.memory.get_or_create(session_id)
                        hint_lang = (
                            selected_language
                            if selected_language != "auto"
                            else (sess.active_language if sess.messages else "auto")
                        )
                        lang_det, dial_det = detect_language_and_dialect_from_text(
                            text_val,
                            hint_language=hint_lang,
                            default_dialect=selected_dialect or sess.active_dialect,
                        )
                        eff_lang = (
                            normalize_language_code(selected_language)
                            if selected_language != "auto"
                            else lang_det
                        )
                        eff_dial = dial_det or selected_dialect
                        gloss = build_swiss_german_gloss(text_val) if eff_lang == "gsw" else None
                        active_turn_task = asyncio.create_task(
                            _generate_and_stream_reply(
                                user_text=text_val,
                                turn_lang=eff_lang,
                                turn_dialect=eff_dial,
                                normalized_text=gloss,
                                cancel_event=cancel_ev,
                            )
                        )

                elif ctrl_type == "emergency_stop":
                    await _cancel_active_turn("emergency_stop")
                    st = await container.robot_gateway.trigger_emergency_stop("WebSocket emergency stop")
                    await websocket.send_json({"type": "estop_latched", "status": st.model_dump(mode="json")})

                elif ctrl_type == "ping":
                    await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        pass
    finally:
        await _cancel_active_turn("websocket_closed")
