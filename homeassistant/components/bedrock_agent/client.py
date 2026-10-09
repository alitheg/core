"""Clients for talking to Amazon Bedrock."""

from collections.abc import Mapping
from typing import Any

from aiobotocore.config import AioConfig
from aiobotocore.session import AioSession, ClientCreatorContext
from botocore.exceptions import BotoCoreError, ClientError

from homeassistant.const import CONF_REGION
from homeassistant.core import HomeAssistant
from homeassistant.helpers.selector import SelectOptionDict
from homeassistant.helpers.singleton import singleton
from homeassistant.util.hass_dict import HassKey

from .const import CONF_ACCESS_KEY_ID, CONF_SECRET_ACCESS_KEY, DOMAIN, LOGGER

DATA_SESSION: HassKey[AioSession] = HassKey(f"{DOMAIN}_session")

# botocore reads the service definitions from disk when a client is created,
# this makes aiobotocore do that in a thread instead of the event loop.
CLIENT_CONFIG = AioConfig(warm_up_loader_caches=True)

# The access key or secret is wrong, expired or has been revoked
INVALID_CREDENTIALS_CODES = {
    "InvalidSignatureException",
    "UnrecognizedClientException",
    "ExpiredTokenException",
}


def _create_session() -> AioSession:
    """Create a session.

    Reading the scoped config reads the AWS config files from disk, so the
    session is created in the executor.
    """
    session = AioSession()
    session.get_scoped_config()
    return session


@singleton(DATA_SESSION, async_=True)
async def async_get_session(hass: HomeAssistant) -> AioSession:
    """Return the session shared by all config entries."""
    return await hass.async_add_executor_job(_create_session)


async def async_create_client(
    hass: HomeAssistant, data: Mapping[str, Any], service_name: str
) -> ClientCreatorContext:
    """Create a client for a Bedrock service, to be used as a context manager."""
    session = await async_get_session(hass)
    return session.create_client(
        service_name,
        region_name=data[CONF_REGION],
        aws_access_key_id=data[CONF_ACCESS_KEY_ID],
        aws_secret_access_key=data[CONF_SECRET_ACCESS_KEY],
        config=CLIENT_CONFIG,
    )


async def async_validate_credentials(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> None:
    """Validate the credentials can be used with Amazon Bedrock."""
    async with await async_create_client(hass, data, "bedrock") as bedrock:
        await bedrock.list_foundation_models(byOutputModality="TEXT")


async def async_list_models(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> list[SelectOptionDict]:
    """List the models that can be used with the Converse API.

    Newer models are only available through cross-region inference profiles,
    so those are listed together with the models that can be used directly.
    """
    models: list[SelectOptionDict] = []
    async with await async_create_client(hass, data, "bedrock") as bedrock:
        response = await bedrock.list_foundation_models(
            byOutputModality="TEXT", byInferenceType="ON_DEMAND"
        )
        models.extend(
            SelectOptionDict(
                value=model["modelId"],
                label=f"{model['providerName']} {model['modelName']}",
            )
            for model in response["modelSummaries"]
            if model.get("responseStreamingSupported", True)
            and model.get("modelLifecycle", {}).get("status") != "LEGACY"
        )
        paginator = bedrock.get_paginator("list_inference_profiles")
        async for page in paginator.paginate(typeEquals="SYSTEM_DEFINED"):
            models.extend(
                SelectOptionDict(
                    value=profile["inferenceProfileId"],
                    label=profile["inferenceProfileName"],
                )
                for profile in page["inferenceProfileSummaries"]
                if profile.get("status", "ACTIVE") == "ACTIVE"
            )
    return sorted(models, key=lambda model: model["label"])


async def async_list_agents(
    hass: HomeAssistant, data: Mapping[str, Any]
) -> list[SelectOptionDict]:
    """List the Bedrock Agents in the account.

    Failures are not fatal, the user can still enter an agent ID by hand.
    """
    try:
        async with await async_create_client(
            hass, data, "bedrock-agent"
        ) as bedrock_agent:
            response = await bedrock_agent.list_agents()
    except (ClientError, BotoCoreError) as err:
        LOGGER.debug("Unable to list Amazon Bedrock agents: %s", err)
        return []
    return [
        SelectOptionDict(value=agent["agentId"], label=agent["agentName"])
        for agent in response["agentSummaries"]
    ]


def is_invalid_credentials(err: ClientError | BotoCoreError) -> bool:
    """Return if an error is caused by invalid credentials."""
    return (
        isinstance(err, ClientError)
        and err.response.get("Error", {}).get("Code") in INVALID_CREDENTIALS_CODES
    )


def is_auth_error(err: ClientError | BotoCoreError) -> bool:
    """Return if an error is caused by invalid credentials or permissions.

    At runtime an access denied error usually means access to the model has
    not been enabled, so it should only be treated as an authentication error
    when validating credentials.
    """
    return is_invalid_credentials(err) or (
        isinstance(err, ClientError)
        and err.response.get("Error", {}).get("Code") == "AccessDeniedException"
    )


def error_message(err: ClientError | BotoCoreError) -> str:
    """Return a human readable message for a botocore error."""
    if isinstance(err, ClientError):
        return err.response.get("Error", {}).get("Message") or str(err)
    return str(err)
