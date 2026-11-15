"""Tests for Phase 4: WebSocket voice streaming, language switching, and barge-in interruption."""

from __future__ import annotations

import asyncio
import base64
import math
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from app.assistant.orchestrator import AssistantResponse, LLMProvider
from app.core.config import Settings
from app.main import create_app
from app.speech.asr.base import Transcription
from app.speech.audio_utils import is_wav_bytes, synthesize_sine_wav
from app.speech.language.detection import detect_language_and_dialect_from_text


def _build_test_app() -> TestClient:
    settings = Settings(
        app_env="test",
        stt_default_provider="mock",
        stt_swiss_german_provider="mock",
        stt_standard_de_provider="mock",
        stt_fr_provider="mock",
        stt_en_provider="mock",
        stt_es_provider="mock",
        stt_cs_provider="mock",
        stt_zh_provider="mock",
        tts_default_provider="mock",
        tts_swiss_german_provider="mock",
        llm_provider="mock",
        robotics_provider="mock",
        serve_frontend=False,
    )
    return TestClient(create_app(settings_override=settings))


def test_multilingual_conversation_switching_and_audio() -> None:
    with _build_test_app() as client:
        sid = "session-switch-01"

        # Turn 1: Swiss German (Wallisertiitsch)
        r1 = client.post(
            f"/api/v1/conversations/{sid}/messages",
            json={
                "text": "Grüezi, wier sind z Leukerbad, wie isch s Wätter hüt?",
                "language": "auto",
                "synthesize_audio": True,
            },
        )
        assert r1.status_code == 200
        d1 = r1.json()
        assert d1["detected_language"] == "gsw"
        assert d1["detected_dialect"] == "vs"
        assert d1["normalized_user_text"] is not None
        wav1 = base64.b64decode(d1["audio_wav_base64"])
        assert is_wav_bytes(wav1)

        # Turn 2: Switch mid-conversation to French
        r2 = client.post(
            f"/api/v1/conversations/{sid}/messages",
            json={
                "text": "Bonjour, pouvez-vous me répondre en français maintenant ?",
                "language": "auto",
                "synthesize_audio": True,
            },
        )
        assert r2.status_code == 200
        d2 = r2.json()
        assert d2["detected_language"] == "fr-FR"
        assert "Bonjour" in d2["assistant"]["text"]

        # Turn 3: Switch mid-conversation to Standard German
        r3 = client.post(
            f"/api/v1/conversations/{sid}/messages",
            json={
                "text": "Guten Tag, können wir jetzt auf Deutsch weitermachen?",
                "language": "auto",
                "synthesize_audio": True,
            },
        )
        assert r3.status_code == 200
        d3 = r3.json()
        assert d3["detected_language"] == "de-DE"
        assert "Guten Tag" in d3["assistant"]["text"]

        # Turn 4: Switch mid-conversation to English (still with language='auto')
        r4 = client.post(
            f"/api/v1/conversations/{sid}/messages",
            json={
                "text": "Hello, can we continue in English now please?",
                "language": "auto",
                "synthesize_audio": True,
            },
        )
        assert r4.status_code == 200
        d4 = r4.json()
        assert d4["detected_language"] == "en-US"
        assert "Hello" in d4["assistant"]["text"]
        assert is_wav_bytes(base64.b64decode(d4["audio_wav_base64"]))

        # Turn 5: Switch mid-conversation to Mandarin Chinese (still with language='auto')
        r5 = client.post(
            f"/api/v1/conversations/{sid}/messages",
            json={
                "text": "你好，现在我们可以用中文继续交流吗？",
                "language": "auto",
                "synthesize_audio": True,
            },
        )
        assert r5.status_code == 200
        d5 = r5.json()
        assert d5["detected_language"] == "zh-CN"
        assert "您好" in d5["assistant"]["text"]
        assert is_wav_bytes(base64.b64decode(d5["audio_wav_base64"]))

        # Turn 6: Switch mid-conversation to Spanish (still with language='auto')
        r6 = client.post(
            f"/api/v1/conversations/{sid}/messages",
            json={
                "text": "Hola, ¿podemos continuar en español ahora por favor?",
                "language": "auto",
                "synthesize_audio": True,
            },
        )
        assert r6.status_code == 200
        d6 = r6.json()
        assert d6["detected_language"] == "es-ES"
        assert "Hola" in d6["assistant"]["text"]
        assert is_wav_bytes(base64.b64decode(d6["audio_wav_base64"]))

        # Turn 7: Switch mid-conversation to Czech (still with language='auto')
        r7 = client.post(
            f"/api/v1/conversations/{sid}/messages",
            json={
                "text": "Dobrý den, můžeme teď pokračovat česky prosím?",
                "language": "auto",
                "synthesize_audio": True,
            },
        )
        assert r7.status_code == 200
        d7 = r7.json()
        assert d7["detected_language"] == "cs-CZ"
        assert "Dobrý den" in d7["assistant"]["text"]
        assert is_wav_bytes(base64.b64decode(d7["audio_wav_base64"]))

        # Verify session history retained all 14 messages (7 user + 7 assistant)
        r_hist = client.get(f"/api/v1/conversations/{sid}")
        assert r_hist.status_code == 200
        assert len(r_hist.json()["messages"]) == 14


def test_rest_full_voice_turn_endpoint() -> None:
    with _build_test_app() as client:
        sample_wav = synthesize_sine_wav(duration_ms=250, freq_hz=440.0)
        resp = client.post(
            "/api/v1/voice/turn",
            data={"language": "gsw", "dialect": "vs"},
            files={"audio": ("input.wav", sample_wav, "audio/wav")},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["transcription"]["language"] == "gsw"
        assert payload["assistant"]["language"] == "gsw"
        assert payload["assistant"]["dialect"] == "vs"
        out_wav = base64.b64decode(payload["audio_wav_base64"])
        assert is_wav_bytes(out_wav)

        resp_zh = client.post(
            "/api/v1/voice/turn",
            data={"language": "zh-CN"},
            files={"audio": ("input_zh.wav", sample_wav, "audio/wav")},
        )
        assert resp_zh.status_code == 200
        payload_zh = resp_zh.json()
        assert payload_zh["transcription"]["language"] == "zh-CN"
        assert payload_zh["assistant"]["language"] == "zh-CN"
        assert is_wav_bytes(base64.b64decode(payload_zh["audio_wav_base64"]))

        for explicit_lang in ("es-ES", "es-US", "cs-CZ"):
            resp_loc = client.post(
                "/api/v1/voice/turn",
                data={"language": explicit_lang},
                files={"audio": (f"input_{explicit_lang}.wav", sample_wav, "audio/wav")},
            )
            assert resp_loc.status_code == 200
            payload_loc = resp_loc.json()
            assert payload_loc["transcription"]["language"] == explicit_lang
            assert payload_loc["assistant"]["language"] == explicit_lang
            assert is_wav_bytes(base64.b64decode(payload_loc["audio_wav_base64"]))


def test_websocket_streaming_voice_vad_and_interrupt() -> None:
    with _build_test_app() as client:
        with client.websocket_connect("/ws/voice?language=gsw&dialect=vs") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "session_ready"
            assert ready["dialect"] == "vs"

            # Stream 300ms of speech PCM followed by manual end_of_utterance
            t = np.arange(4800, dtype=np.float32) / 16000.0
            pcm_speech = (0.35 * np.sin(2.0 * math.pi * 320.0 * t) * 32767).astype(np.int16).tobytes()
            ws.send_bytes(pcm_speech)

            vad_msg = ws.receive_json()
            assert vad_msg["type"] == "vad_state"
            assert vad_msg["state"] == "speech_started"

            ws.send_json({"type": "end_of_utterance"})

            seen_types: list[str] = []
            for _ in range(10):
                evt = ws.receive_json()
                seen_types.append(evt["type"])
                if evt["type"] == "tts_stream_complete":
                    break

            assert "final_transcript" in seen_types
            assert "assistant_response" in seen_types
            assert "tts_audio_chunk" in seen_types
            assert "tts_stream_complete" in seen_types

            # Verify live config update & interrupt control message
            ws.send_json({"type": "config", "language": "fr-FR", "speaking_rate": 1.15})
            cfg_ack = ws.receive_json()
            assert cfg_ack["type"] == "config_updated"
            assert cfg_ack["language"] == "fr-FR"

            ws.send_json({"type": "interrupt"})


def test_go_forward_5m_queues_robot_confirmation_across_all_7_languages() -> None:
    with _build_test_app() as client:
        commands = [
            ("go forward 5m", "en-US"),
            ("avance de 5 mètres", "fr-FR"),
            ("geh 5 Meter vorwärts", "de-DE"),
            ("gang 5 Meter füre", "gsw"),
            ("avanza 5 metros", "es-ES"),
            ("jdi 5 metrů dopředu", "cs-CZ"),
            ("向前走五米", "zh-CN"),
        ]
        for idx, (utterance, expected_lang) in enumerate(commands):
            resp = client.post(
                f"/api/v1/conversations/robot-5m-{idx}/messages",
                json={"text": utterance, "language": "auto", "synthesize_audio": True},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["detected_language"] == expected_lang, (utterance, data)
            cmds = data["assistant"]["robot_commands"]
            assert len(cmds) == 1, (utterance, data)
            assert cmds[0]["action"] == "move_to"
            assert cmds[0]["target"]["x"] == 5.0
            assert cmds[0]["requires_confirmation"] is True
            assert is_wav_bytes(base64.b64decode(data["audio_wav_base64"]))

        status_resp = client.get("/api/v1/robotics/status").json()
        assert len(status_resp["pending_confirmations"]) == 7

        # Regression: E-Stop followed by Reset E-Stop must restore READY state and allow new proposals
        client.post("/api/v1/robotics/estop", json={"reason": "operator test"})
        reset_res = client.post("/api/v1/robotics/estop/reset").json()
        assert reset_res["status"]["state"] == "ready"
        after_reset = client.post(
            "/api/v1/conversations/robot-after-reset/messages",
            json={"text": "go forward 5m", "language": "auto"},
        ).json()
        assert len(after_reset["assistant"]["robot_commands"]) == 1


def test_two_tier_flash_plus_er2_multistep_move_5m_and_turn_around() -> None:
    with _build_test_app() as client:
        # 1. WebSocket streaming: verify quick_ack arrives BEFORE assistant_response and ER 2 decomposes both steps
        with client.websocket_connect("/ws/voice?language=auto") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "session_ready"

            ws.send_json({"type": "text_input", "text": "Move forward 5 m and turn around."})

            events: list[dict] = []
            for _ in range(12):
                evt = ws.receive_json()
                events.append(evt)
                if evt["type"] == "tts_stream_complete":
                    break

            event_types = [e["type"] for e in events]
            assert "quick_ack" in event_types
            assert "assistant_response" in event_types
            assert event_types.index("quick_ack") < event_types.index("assistant_response")

            ack_evt = next(e for e in events if e["type"] == "quick_ack")
            assert ack_evt["text"]

            asst_evt = next(e for e in events if e["type"] == "assistant_response")["assistant"]
            assert asst_evt["er2_model"] == "mock-er2"
            assert "mock-er2" in asst_evt["model"]
            cmds = asst_evt["robot_commands"]
            assert len(cmds) == 2, cmds
            assert cmds[0]["action"] == "move_to"
            assert cmds[0]["target"]["x"] == 5.0
            assert cmds[1]["action"] == "rotate"
            assert abs(cmds[1]["yaw_rad"]) > 3.1

        # 2. Multilingual compound multi-step instructions across DE, FR, GSW, ES, CS, and ZH
        multilingual_multistep = [
            ("Geh 5 Meter vorwärts und dreh dich um.", "de-DE"),
            ("Avance de 5 mètres et fais demi-tour.", "fr-FR"),
            ("Gang füüf Meter füre und dräi dich ume.", "gsw"),
            ("Avanza cinco metros hacia adelante y date la vuelta.", "es-ES"),
            ("Jdi pět metrů dopředu a otoč se.", "cs-CZ"),
            ("向前走五米然后掉头。", "zh-CN"),
        ]
        for idx, (utterance, expected_lang) in enumerate(multilingual_multistep):
            resp = client.post(
                f"/api/v1/conversations/multistep-{idx}/messages",
                json={"text": utterance, "language": "auto"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["detected_language"] == expected_lang, (utterance, data)
            rcmds = data["assistant"]["robot_commands"]
            assert len(rcmds) == 2, (utterance, rcmds)
            assert rcmds[0]["action"] == "move_to"
            assert rcmds[0]["target"]["x"] == 5.0
            assert rcmds[1]["action"] == "rotate"
            assert abs(rcmds[1]["yaw_rad"]) > 3.1

        # 3. Verify Spanish & Czech directional rotations don't collide with 180-deg turn-around or clause splitting
        rot_cases = [
            ("Gira cuarenta y cinco grados a la derecha.", "es-ES", -45),
            ("Gira a la izquierda.", "es-ES", 90),
            ("Otoč se o devadesát stupňů doprava.", "cs-CZ", -90),
            ("Otoč se doleva.", "cs-CZ", 90),
        ]
        for idx, (utterance, expected_lang, expected_deg) in enumerate(rot_cases):
            resp = client.post(
                f"/api/v1/conversations/rot-es-cs-{idx}/messages",
                json={"text": utterance, "language": "auto"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["detected_language"] == expected_lang, (utterance, data)
            rcmds = data["assistant"]["robot_commands"]
            assert len(rcmds) == 1, (utterance, rcmds)
            assert rcmds[0]["action"] == "rotate"
            assert round(math.degrees(rcmds[0]["yaw_rad"])) == expected_deg, (utterance, rcmds)


def test_german_detection_and_verb_final_robot_commands() -> None:
    # Ensure Standard German words ("Zürich", "Juli", "Müsli", "Brokkoli", "in Gang") stay de-DE
    for de_sentence in (
        "Guten Tag aus Zürich im Juli, ich esse Müsli und Brokkoli.",
        "Bitte setz das System jetzt in Gang.",
        "Geh 5 Meter vorwärts und dreh dich um.",
        "Bitte 3 Meter vorwärts fahren und dann umdrehen.",
    ):
        lang, dial = detect_language_and_dialect_from_text(de_sentence, hint_language="auto")
        assert lang == "de-DE", (de_sentence, lang, dial)
        assert dial is None

    # Swiss greeting + Standard High German grammar -> de-CH (Swiss Standard German), not gsw
    ch_lang, ch_dial = detect_language_and_dialect_from_text(
        "Grüezi, gehen Sie bitte 4 Meter vorwärts und drehen Sie sich um.",
        hint_language="auto",
    )
    assert ch_lang == "de-CH"
    assert ch_dial is None

    with _build_test_app() as client:
        r_vf = client.post(
            "/api/v1/conversations/de-verb-final/messages",
            json={"text": "Bitte 3 Meter vorwärts fahren und dann umdrehen.", "language": "auto"},
        )
        assert r_vf.status_code == 200
        d_vf = r_vf.json()
        assert d_vf["detected_language"] == "de-DE"
        cmds = d_vf["assistant"]["robot_commands"]
        assert len(cmds) == 2
        assert cmds[0]["action"] == "move_to"
        assert cmds[0]["target"]["x"] == 3.0
        assert cmds[1]["action"] == "rotate"
        assert abs(cmds[1]["yaw_rad"]) > 3.1


def test_german_voice_ws_resists_transient_click_and_stop_button_after_vad_eou() -> None:
    app = create_app(
        settings_override=Settings(
            app_env="test",
            stt_default_provider="mock",
            stt_swiss_german_provider="mock",
            stt_standard_de_provider="mock",
            stt_fr_provider="mock",
            stt_en_provider="mock",
            stt_zh_provider="mock",
            tts_default_provider="mock",
            tts_swiss_german_provider="mock",
            llm_provider="mock",
            robotics_provider="mock",
            serve_frontend=False,
        )
    )
    with TestClient(app) as client:
        # Simulate slower final German ASR (~180ms) while partial ASR already detected German
        async def _fake_german_transcribe_routed(
            audio: bytes,
            requested_language: str = "auto",
            session_language_hint: str | None = None,
            is_partial: bool = False,
        ) -> Transcription:
            if not is_partial:
                await asyncio.sleep(0.18)
            return Transcription(
                text="Geh 5 Meter vorwärts und dreh dich um.",
                language="de-DE",
                confidence=0.96,
                provider="google",
                model="chirp_3",
                latency_ms=180.0 if not is_partial else 40.0,
                is_partial=is_partial,
            )

        app.state.container.speech_router.transcribe_routed = _fake_german_transcribe_routed  # type: ignore[method-assign]

        with client.websocket_connect("/ws/voice?language=auto") as ws:
            ready = ws.receive_json()
            assert ready["type"] == "session_ready"

            # 1. Stream 1000ms of speech in 125ms chunks -> triggers speech_started + partial_transcript
            t_chunk = np.arange(2000, dtype=np.float32) / 16000.0
            speech_chunk = (0.30 * np.sin(2.0 * math.pi * 300.0 * t_chunk) * 32767).astype(np.int16).tobytes()
            silence_chunk = np.zeros(2000, dtype=np.int16).tobytes()

            for _ in range(8):
                ws.send_bytes(speech_chunk)
            # 2. Stream 750ms of silence -> triggers automatic VAD end_of_utterance
            for _ in range(6):
                ws.send_bytes(silence_chunk)

            # 3. Immediately simulate a single 125ms mouse-click spike + clicking "Stop Microphone"
            #    while final German ASR is still in-flight!
            ws.send_bytes(speech_chunk)
            ws.send_json({"type": "end_of_utterance"})

            events: list[dict[str, Any]] = []
            for _ in range(20):
                evt = ws.receive_json()
                events.append(evt)
                if evt["type"] == "tts_stream_complete":
                    break

            event_types = [e["type"] for e in events]
            assert "interrupted" not in event_types, events
            assert "final_transcript" in event_types, events
            assert "quick_ack" in event_types, events
            assert "assistant_response" in event_types, events

            final_t = next(e for e in events if e["type"] == "final_transcript")["transcription"]
            assert final_t["language"] == "de-DE"
            assert "5 Meter vorwärts" in final_t["text"]

            asst = next(e for e in events if e["type"] == "assistant_response")["assistant"]
            assert len(asst["robot_commands"]) == 2
            assert asst["robot_commands"][0]["action"] == "move_to"
            assert asst["robot_commands"][0]["target"]["x"] == 5.0
            assert asst["robot_commands"][1]["action"] == "rotate"
            assert abs(asst["robot_commands"][1]["yaw_rad"]) > 3.1


def test_orchestrator_fallback_when_llm_omits_tool_call_on_german_robot_command() -> None:
    class _ForgetfulGermanLLM(LLMProvider):
        provider_name = "forgetful-stub"

        async def generate_response(
            self,
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]] | None = None,
            system_instruction: str | None = None,
            cancel_event: asyncio.Event | None = None,
        ) -> AssistantResponse:
            return AssistantResponse(
                text="Alles klar, ich fahre jetzt 5 Meter vorwärts!",
                language="de-DE",
                tool_calls=[],
                provider=self.provider_name,
                model="stub-flash",
            )

    app = create_app(
        settings_override=Settings(
            app_env="test",
            stt_default_provider="mock",
            stt_swiss_german_provider="mock",
            stt_standard_de_provider="mock",
            stt_fr_provider="mock",
            stt_en_provider="mock",
            stt_zh_provider="mock",
            tts_default_provider="mock",
            tts_swiss_german_provider="mock",
            llm_provider="mock",
            robotics_provider="mock",
            serve_frontend=False,
        )
    )
    with TestClient(app) as client:
        app.state.container.orchestrator._llm = _ForgetfulGermanLLM()
        resp = client.post(
            "/api/v1/conversations/de-fallback-01/messages",
            json={"text": "Geh 5 Meter vorwärts und dreh dich um.", "language": "de-DE"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        rcmds = payload["assistant"]["robot_commands"]
        assert len(rcmds) == 2, payload
        assert rcmds[0]["action"] == "move_to"
        assert rcmds[0]["target"]["x"] == 5.0
        assert rcmds[1]["action"] == "rotate"
        assert abs(rcmds[1]["yaw_rad"]) > 3.1
