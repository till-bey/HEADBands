"""Tests for Phase 2 & Phase 3: Language detection, dialect preservation, routing, and audio VAD."""

from __future__ import annotations

import math

import numpy as np

from app.core.config import Settings
from app.speech.asr.base import MockASRProvider
from app.speech.audio_utils import (
    StreamingAudioBuffer,
    extract_pcm16_from_wav,
    is_wav_bytes,
    pcm16_to_wav,
    split_into_sentences,
)
from app.speech.language.detection import (
    clean_cjk_spacing,
    detect_language_and_dialect_from_text,
    normalize_language_code,
)
from app.speech.language.normalization import build_swiss_german_gloss
from app.speech.language.routing import SpeechRouter
from app.speech.tts.base import MockTTSProvider
from app.speech.tts.piper import resolve_swissdial_voice_key


def test_language_and_swiss_dialect_detection() -> None:
    # Wallisertiitsch / Leukerbad
    lang, dialect = detect_language_and_dialect_from_text(
        "Grüezi, wier sind z Leukerbad und s Wätter isch güet!"
    )
    assert lang == "gsw"
    assert dialect == "vs"

    # Bärndütsch
    lang_be, dialect_be = detect_language_and_dialect_from_text(
        "Grüessech mitenand, ig ha nid gwüsst wie das geit."
    )
    assert lang_be == "gsw"
    assert dialect_be == "be"

    # Züridütsch
    lang_zh, dialect_zh = detect_language_and_dialect_from_text(
        "Mir händ z Züri nöd vil Ziit hüt."
    )
    assert lang_zh == "gsw"
    assert dialect_zh == "zh"

    # Standard German (with ß)
    lang_de, dialect_de = detect_language_and_dialect_from_text(
        "Guten Tag, können Sie mir bitte draußen helfen?"
    )
    assert lang_de == "de-DE"
    assert dialect_de is None

    # French & Swiss French
    lang_fr, _ = detect_language_and_dialect_from_text(
        "Bonjour, comment allez-vous aujourd'hui ?"
    )
    assert lang_fr == "fr-FR"

    lang_fr_ch, _ = detect_language_and_dialect_from_text(
        "Bonjour depuis le Valais, il y a nonante minutes."
    )
    assert lang_fr_ch == "fr-CH"

    # English (auto & en-GB hint & short ISO-639-1 'en' acoustic hint)
    lang_en, dialect_en = detect_language_and_dialect_from_text(
        "Hello, can you tell me how the weather is today?"
    )
    assert lang_en == "en-US"
    assert dialect_en is None

    lang_en_gb, _ = detect_language_and_dialect_from_text(
        "Good morning, could you check the workspace?",
        hint_language="en-GB",
    )
    assert lang_en_gb == "en-GB"

    lang_short_en, _ = detect_language_and_dialect_from_text("Okay", hint_language="en")
    assert lang_short_en == "en-US"

    # Chirp 3 auto-mode partially normalized Swiss German still resolves to ('gsw', 'vs')
    lang_chirp_gsw, dial_chirp_gsw = detect_language_and_dialect_from_text(
        "Grüezi miteinander, wir sind heute in Leukerbad und das Wetter ist gut.",
        hint_language="de",
    )
    assert lang_chirp_gsw == "gsw"
    assert dial_chirp_gsw == "vs"

    # Mandarin Chinese (Simplified & Traditional, plus Chirp 3 zh-Hans / cmn-Hans-CN codes)
    assert normalize_language_code("zh-Hans") == "zh-CN"
    assert normalize_language_code("cmn-Hans-CN") == "zh-CN"
    assert normalize_language_code("zh-Hant") == "zh-TW"
    assert normalize_language_code("cmn-Hant-TW") == "zh-TW"

    spaced_cjk = clean_cjk_spacing("你 好 , 欢 迎 来 到 瑞 士 洛 依 克 巴 德 。 请 检 查 ROS 2 机 器 人 吗 ?")
    assert spaced_cjk == "你好,欢迎来到瑞士洛依克巴德。请检查 ROS 2 机器人吗?"

    lang_zh_cn, dial_zh_cn = detect_language_and_dialect_from_text(spaced_cjk)
    assert lang_zh_cn == "zh-CN"
    assert dial_zh_cn is None

    lang_zh_tw, dial_zh_tw = detect_language_and_dialect_from_text("請問今天機器人的狀態怎麼樣？")
    assert lang_zh_tw == "zh-TW"
    assert dial_zh_tw is None

    # Spanish (es-ES / es-US) & Czech (cs-CZ — accented & unaccented)
    assert normalize_language_code("es") == "es-ES"
    assert normalize_language_code("es-ES") == "es-ES"
    assert normalize_language_code("es-MX") == "es-US"
    assert normalize_language_code("es-419") == "es-US"
    assert normalize_language_code("cs") == "cs-CZ"
    assert normalize_language_code("cz") == "cs-CZ"
    assert normalize_language_code("cs-CZ") == "cs-CZ"

    lang_es, dial_es = detect_language_and_dialect_from_text(
        "Hola, ¿puedes avanzar cinco metros y darte la vuelta?"
    )
    assert lang_es == "es-ES"
    assert dial_es is None

    lang_es_us, _ = detect_language_and_dialect_from_text(
        "Hola, avanza 5 metros hacia adelante por favor.",
        hint_language="es-US",
    )
    assert lang_es_us == "es-US"

    lang_cs, dial_cs = detect_language_and_dialect_from_text(
        "Dobrý den, jdi pět metrů dopředu a otoč se."
    )
    assert lang_cs == "cs-CZ"
    assert dial_cs is None

    lang_cs_ascii, _ = detect_language_and_dialect_from_text(
        "Dobry den, jdi 5 metru dopredu a otoc se prosim."
    )
    assert lang_cs_ascii == "cs-CZ"


def test_non_destructive_swiss_german_gloss_preserves_original() -> None:
    original = "Grüezi mitenand, wie isch s Wätter hüt?"
    gloss = build_swiss_german_gloss(original)
    assert original == "Grüezi mitenand, wie isch s Wätter hüt?"
    assert gloss is not None
    assert "Guten Tag" in gloss
    assert "heute" in gloss


def test_provider_selection_and_dialect_voice_routing() -> None:
    settings = Settings(
        stt_default_provider="google",
        stt_swiss_german_provider="faster_whisper",
        stt_standard_de_provider="google",
        stt_fr_provider="google",
        tts_default_provider="google",
        tts_swiss_german_provider="piper",
        piper_default_dialect="vs",
    )
    mock_google_asr = MockASRProvider()
    mock_google_asr.provider_name = "google"
    mock_fw_asr = MockASRProvider()
    mock_fw_asr.provider_name = "faster_whisper"

    mock_google_tts = MockTTSProvider()
    mock_google_tts.provider_name = "google"
    mock_piper_tts = MockTTSProvider()
    mock_piper_tts.provider_name = "piper"

    router = SpeechRouter(
        settings=settings,
        asr_providers={"google": mock_google_asr, "faster_whisper": mock_fw_asr},
        tts_providers={"google": mock_google_tts, "piper": mock_piper_tts},
    )

    assert router.select_asr_provider("gsw").provider_name == "faster_whisper"
    assert router.select_asr_provider("auto", session_hint_language="gsw").provider_name == "google"
    assert router.select_asr_provider("de-DE").provider_name == "google"
    assert router.select_asr_provider("fr-FR").provider_name == "google"
    assert router.select_asr_provider("fr-CH").provider_name == "google"
    assert router.select_asr_provider("en-US").provider_name == "google"
    assert router.select_asr_provider("en-GB").provider_name == "google"

    gsw_tts = router.select_tts_route("gsw", dialect="vs")
    assert gsw_tts.provider.provider_name == "piper"
    assert gsw_tts.dialect == "vs"
    assert gsw_tts.voice == "de_CH-swissdial_vs-medium"

    de_tts = router.select_tts_route("de-DE")
    assert de_tts.provider.provider_name == "google"
    assert de_tts.language == "de-DE"

    fr_tts = router.select_tts_route("fr-FR")
    assert fr_tts.provider.provider_name == "google"
    assert fr_tts.language == "fr-FR"

    en_tts = router.select_tts_route("en-US")
    assert en_tts.provider.provider_name == "google"
    assert en_tts.language == "en-US"
    assert en_tts.voice == "en-US-Chirp3-HD-Aoede"

    assert router.select_asr_provider("zh-CN").provider_name == "google"
    assert router.select_asr_provider("zh-TW").provider_name == "google"

    zh_cn_tts = router.select_tts_route("zh-CN")
    assert zh_cn_tts.provider.provider_name == "google"
    assert zh_cn_tts.language == "zh-CN"
    assert zh_cn_tts.voice == "cmn-CN-Chirp3-HD-Aoede"

    zh_tw_tts = router.select_tts_route("zh-TW")
    assert zh_tw_tts.provider.provider_name == "google"
    assert zh_tw_tts.language == "zh-TW"
    assert zh_tw_tts.voice == "cmn-TW-Wavenet-A"

    assert router.select_asr_provider("es-ES").provider_name == "google"
    assert router.select_asr_provider("es-US").provider_name == "google"
    assert router.select_asr_provider("cs-CZ").provider_name == "google"

    es_es_tts = router.select_tts_route("es-ES")
    assert es_es_tts.provider.provider_name == "google"
    assert es_es_tts.language == "es-ES"
    assert es_es_tts.voice == "es-ES-Neural2-A"

    es_us_tts = router.select_tts_route("es-US")
    assert es_us_tts.provider.provider_name == "google"
    assert es_us_tts.language == "es-US"
    assert es_us_tts.voice == "es-US-Neural2-A"

    cs_tts = router.select_tts_route("cs-CZ")
    assert cs_tts.provider.provider_name == "google"
    assert cs_tts.language == "cs-CZ"
    assert cs_tts.voice == "cs-CZ-Wavenet-A"


def test_all_eight_swissdial_dialects_resolve() -> None:
    for code in ("ag", "be", "bs", "gr", "lu", "sg", "vs", "zh"):
        resolved_code, voice_key = resolve_swissdial_voice_key(code)
        assert resolved_code == code
        assert voice_key == f"de_CH-swissdial_{code}-medium"


def test_pcm16_wav_roundtrip_and_sentence_chunker() -> None:
    t = np.arange(1600, dtype=np.float32) / 16000.0
    pcm = (0.4 * np.sin(2.0 * math.pi * 440.0 * t) * 32767).astype(np.int16).tobytes()
    wav = pcm16_to_wav(pcm, sample_rate=16000)
    assert is_wav_bytes(wav)
    recovered_pcm, sr, ch = extract_pcm16_from_wav(wav)
    assert sr == 16000
    assert ch == 1
    assert recovered_pcm == pcm

    sentences = split_into_sentences(
        "Grüezi us St. Gallen! Mir sind parat z.B. für Tests. Chasch du mi ghöre?"
    )
    assert len(sentences) == 3
    assert "St. Gallen!" in sentences[0]
    assert "z.B." in sentences[1]

    zh_sentences = split_into_sentences("您好！欢迎来到瑞士洛伊克巴德。请问今天我可以帮您什么？")
    assert len(zh_sentences) == 3
    assert zh_sentences[0] == "您好！"
    assert zh_sentences[1] == "欢迎来到瑞士洛伊克巴德。"
    assert zh_sentences[2] == "请问今天我可以帮您什么？"


def test_streaming_audio_buffer_vad_and_end_of_utterance() -> None:
    buf = StreamingAudioBuffer(
        sample_rate=16000,
        energy_threshold=0.02,
        min_speech_ms=200.0,
        end_silence_ms=300.0,
        partial_interval_ms=250.0,
    )
    frame_100ms_samples = 1600
    silence_frame = np.zeros(frame_100ms_samples, dtype=np.int16).tobytes()
    t = np.arange(frame_100ms_samples, dtype=np.float32) / 16000.0
    speech_frame = (0.3 * np.sin(2.0 * math.pi * 300.0 * t) * 32767).astype(np.int16).tobytes()

    # Pre-roll silence
    ev0 = buf.push_pcm(silence_frame)
    assert not ev0.is_speech

    # Speech starts
    ev1 = buf.push_pcm(speech_frame)
    assert ev1.speech_started is True

    # Continued speech triggers partial window
    ev2 = buf.push_pcm(speech_frame)
    ev3 = buf.push_pcm(speech_frame)
    assert ev2.should_emit_partial or ev3.should_emit_partial

    # Trailing silence triggers end_of_utterance and flushes PCM
    buf.push_pcm(silence_frame)
    buf.push_pcm(silence_frame)
    ev_end = buf.push_pcm(silence_frame)
    assert ev_end.end_of_utterance is True
    assert len(ev_end.utterance_pcm) > len(speech_frame) * 3
