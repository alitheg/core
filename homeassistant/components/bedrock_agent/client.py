"""Clients for talking to Amazon Bedrock."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aiobotocore.session import AioSession, ClientCreatorContext
from botocore.exceptions import BotoCoreError, ClientError

from homeassistant.const import CONF_REGION
from homeassistant.core import HomeAssistant
from homeassistant.helpers.singleton import singleton
from homeassistant.util.hass_dict import HassKey

from .const import CONF_ACCESS_KEY_ID, CONF_SECRET_ACCESS_KEY, DOMAIN

DATA_SESSION: HassKey[AioSession] = HassKey(f"{DOMAIN}_session")

SERVICES = ("bedrock", "bedrock-agent", "bedrock-agent-runtime", "bedrock-runtime")

# The access key or secret is wrong, expired or has been revoked
INVALID_CREDENTIALS_CODES = {
    "InvalidSignatureException",
    "UnrecognizedClientException",
    "ExpiredTokenException",
}


def _create_session() -> AioSession:
    """Create a session with the service definitions loaded.

    botocore reads the service definitions from disk when a client is created.
    They are cached by the session, so they are loaded here in the executor to
    avoid blocking the event loop later.
    """
    session = AioSession()
    loader = session.get_component("data_loader")
    loader.list_available_services("service-2")
    for service in SERVICES:
        loader.load_service_model(service, "service-2")
        loader.load_service_model(service, "endpoint-rule-set-1")
    for name in ("endpoints", "partitions", "_retry", "sdk-default-configuration"):
        loader.load_data(name)
    session.get_scoped_config()
    return session


@singleton(DATA_SESSION, async_=True)
async def async_get_session(hass: HomeAssistant) -> AioSession:
    """Return the session shared by all config entries."""
    return await hass.async_add_executor_job(_create_session)


def create_client(
    session: AioSession, data: Mapping[str, Any], service_name: str
) -> ClientCreatorContext:
    """Create a client for a Bedrock service, to be used as a context manager."""
    return session.create_client(
        service_name,
        region_name=data[CONF_REGION],
        aws_access_key_id=data[CONF_ACCESS_KEY_ID],
        aws_secret_access_key=data[CONF_SECRET_ACCESS_KEY],
    )


def is_invalid_credentials(err: ClientError | BotoCoreError) -> bool:
    """Return if an error is caused by invalid credentials."""
    return (
        isinstance(err, ClientError)
        and err.response.get("Error", {}).get("Code") in INVALID_CREDENTIALS_CODES
    )


def is_auth_error(err: ClientError) -> bool:
    """Return if a client error is caused by invalid credentials or permissions.

    At runtime an access denied error usually means access to the model has
    not been enabled, so it should only be treated as an authentication error
    when validating credentials.
    """
    return (
        is_invalid_credentials(err)
        or err.response.get("Error", {}).get("Code") == "AccessDeniedException"
    )


def error_message(err: ClientError | BotoCoreError) -> str:
    """Return a human readable message for a botocore error."""
    if isinstance(err, ClientError):
        return err.response.get("Error", {}).get("Message") or str(err)
    return str(err)
