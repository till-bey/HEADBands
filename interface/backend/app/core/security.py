"""Authentication, Cloud Run OIDC service-to-service tokens, and Secret Manager helpers."""

from __future__ import annotations

import hmac
from typing import Annotated

from fastapi import Depends, Header, HTTPException, WebSocket, status
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import id_token

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def verify_constant_time_token(provided: str | None, expected: str) -> bool:
    """Compare bearer/shared tokens in constant time."""
    if not expected:
        return True
    if not provided:
        return False
    cleaned = provided.removeprefix("Bearer ").removeprefix("bearer ").strip()
    return hmac.compare_digest(cleaned.encode("utf-8"), expected.encode("utf-8"))


async def require_api_auth(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    """Enforce optional API authentication when API_AUTH_TOKEN is configured."""
    if not settings.api_auth_token:
        return
    candidate = authorization or x_api_key
    if not verify_constant_time_token(candidate, settings.api_auth_token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials.",
        )


def verify_robot_bridge_ws_auth(websocket: WebSocket, settings: Settings) -> bool:
    """Authenticate persistent outbound connection from ROS 2 robot bridge."""
    expected = settings.robot_bridge_shared_token.strip()
    if not expected:
        if settings.app_env == "production":
            logger.error("Refusing unauthenticated robot bridge in production without ROBOT_BRIDGE_SHARED_TOKEN")
            return False
        return True
    header_token = websocket.headers.get("authorization") or websocket.headers.get("x-robot-token")
    query_token = websocket.query_params.get("token")
    return verify_constant_time_token(header_token or query_token, expected)


def mint_cloud_run_id_token(target_audience: str) -> str | None:
    """Fetch a Google-signed OIDC ID token for private Cloud Run service-to-service calls."""
    try:
        auth_req = GoogleAuthRequest()
        return id_token.fetch_id_token(auth_req, target_audience)
    except Exception as exc:
        logger.warning("Could not mint OIDC ID token for private service call", extra={"audience": target_audience, "reason": str(exc)})
        return None


def access_secret_version(project_id: str, secret_id: str, version_id: str = "latest") -> str:
    """Fetch secret payload from Google Cloud Secret Manager."""
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/{version_id}"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("utf-8").strip()
