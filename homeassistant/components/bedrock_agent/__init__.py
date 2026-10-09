"""The Amazon Bedrock integration."""

from contextlib import AsyncExitStack
from dataclasses import dataclass

from aiobotocore.client import AioBaseClient
from botocore.exceptions import BotoCoreError, ClientError

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import Platform
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryAuthFailed, ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv

from .client import (
    async_create_client,
    async_validate_credentials,
    error_message,
    is_auth_error,
)
from .const import DOMAIN

PLATFORMS = (Platform.AI_TASK, Platform.CONVERSATION)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)


@dataclass
class BedrockData:
    """Clients for talking to Amazon Bedrock."""

    runtime: AioBaseClient
    agent_runtime: AioBaseClient


type BedrockConfigEntry = ConfigEntry[BedrockData]


async def async_setup_entry(hass: HomeAssistant, entry: BedrockConfigEntry) -> bool:
    """Set up Amazon Bedrock from a config entry."""
    stack = AsyncExitStack()

    try:
        await async_validate_credentials(hass, entry.data)
        entry.runtime_data = BedrockData(
            runtime=await stack.enter_async_context(
                await async_create_client(hass, entry.data, "bedrock-runtime")
            ),
            agent_runtime=await stack.enter_async_context(
                await async_create_client(hass, entry.data, "bedrock-agent-runtime")
            ),
        )
    except (ClientError, BotoCoreError) as err:
        await stack.aclose()
        if is_auth_error(err):
            raise ConfigEntryAuthFailed(
                translation_domain=DOMAIN,
                translation_key="invalid_auth",
                translation_placeholders={"error": error_message(err)},
            ) from err
        raise ConfigEntryNotReady(
            translation_domain=DOMAIN,
            translation_key="cannot_connect",
            translation_placeholders={"error": error_message(err)},
        ) from err

    entry.async_on_unload(stack.aclose)

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


async def _async_update_listener(
    hass: HomeAssistant, entry: BedrockConfigEntry
) -> None:
    """Reload the entry when it or its subentries are updated."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: BedrockConfigEntry) -> bool:
    """Unload Amazon Bedrock."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
