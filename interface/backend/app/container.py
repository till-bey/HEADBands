"""Dependency-injection service container wiring replaceable providers from Settings."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request

from app.assistant.memory import SessionMemoryStore
from app.assistant.orchestrator import (
    ConversationOrchestrator,
    GeminiLLMProvider,
    LLMProvider,
    MockLLMProvider,
)
from app.assistant.tools import AssistantToolExecutor
from app.core.config import Settings
from app.infrastructure.firestore import FirestoreRepository
from app.infrastructure.pubsub import TelemetryPubSubPublisher
from app.infrastructure.storage import AudioStorageService
from app.robotics.action_validator import ActionSafetyValidator
from app.robotics.er2_client import (
    GeminiRoboticsER2Client,
    MockRoboticsER2Client,
    RoboticsReasoningProvider,
)
from app.robotics.robot_gateway import RobotSafetyGateway
from app.speech.asr.base import ASRProvider, MockASRProvider
from app.speech.asr.faster_whisper import FasterWhisperProvider
from app.speech.asr.google_stt import GoogleASRProvider
from app.speech.language.routing import SpeechRouter
from app.speech.tts.base import MockTTSProvider, TTSProvider
from app.speech.tts.google_tts import GoogleTTSProvider
from app.speech.tts.piper import PiperTTSProvider


@dataclass
class ServiceContainer:
    """Holds singleton service instances for a FastAPI application lifespan."""

    settings: Settings
    speech_router: SpeechRouter
    asr_providers: dict[str, ASRProvider]
    tts_providers: dict[str, TTSProvider]
    llm_provider: LLMProvider
    memory: SessionMemoryStore
    orchestrator: ConversationOrchestrator
    audio_storage: AudioStorageService
    firestore_repo: FirestoreRepository
    pubsub_publisher: TelemetryPubSubPublisher
    safety_validator: ActionSafetyValidator
    robot_gateway: RobotSafetyGateway
    robotics_er2: RoboticsReasoningProvider

    async def shutdown(self) -> None:
        fw = self.asr_providers.get("faster_whisper")
        if isinstance(fw, FasterWhisperProvider):
            await fw.close()


def build_service_container(settings: Settings) -> ServiceContainer:
    """Instantiate modular providers cleanly without eager cloud network calls at boot."""
    asr_providers: dict[str, ASRProvider] = {
        "google": GoogleASRProvider(settings),
        "faster_whisper": FasterWhisperProvider(settings),
        "mock": MockASRProvider(),
    }

    tts_providers: dict[str, TTSProvider] = {
        "google": GoogleTTSProvider(settings),
        "piper": PiperTTSProvider(settings),
        "mock": MockTTSProvider(),
    }

    speech_router = SpeechRouter(
        settings=settings,
        asr_providers=asr_providers,
        tts_providers=tts_providers,
    )

    audio_storage = AudioStorageService(settings)
    firestore_repo = FirestoreRepository(settings)
    pubsub_publisher = TelemetryPubSubPublisher(settings)

    safety_validator = ActionSafetyValidator(settings)
    robot_gateway = RobotSafetyGateway(
        settings=settings,
        validator=safety_validator,
        telemetry_publisher=pubsub_publisher,
    )

    robotics_er2: RoboticsReasoningProvider = (
        GeminiRoboticsER2Client(settings)
        if settings.robotics_provider == "gemini_robotics_er"
        else MockRoboticsER2Client()
    )

    llm_provider: LLMProvider = (
        GeminiLLMProvider(settings)
        if settings.llm_provider == "google"
        else MockLLMProvider()
    )

    memory = SessionMemoryStore(
        firestore_repo=firestore_repo,
        default_dialect=settings.piper_default_dialect,
    )
    tool_executor = AssistantToolExecutor(
        robot_gateway=robot_gateway,
        max_speed=settings.robot_max_linear_speed_mps,
        max_radius=settings.robot_max_workspace_radius_m,
        robotics_er2=robotics_er2,
    )
    orchestrator = ConversationOrchestrator(
        settings=settings,
        llm_provider=llm_provider,
        memory=memory,
        tool_executor=tool_executor,
    )

    return ServiceContainer(
        settings=settings,
        speech_router=speech_router,
        asr_providers=asr_providers,
        tts_providers=tts_providers,
        llm_provider=llm_provider,
        memory=memory,
        orchestrator=orchestrator,
        audio_storage=audio_storage,
        firestore_repo=firestore_repo,
        pubsub_publisher=pubsub_publisher,
        safety_validator=safety_validator,
        robot_gateway=robot_gateway,
        robotics_er2=robotics_er2,
    )


def get_container(request: Request) -> ServiceContainer:
    """FastAPI dependency returning the active ServiceContainer."""
    return request.app.state.container
