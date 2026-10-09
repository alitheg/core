"""The Amazon Bedrock integration."""

from __future__ import annotations

from contextlib import AsyncExitStack
from dataclasses import dataclass
import mimetypes
from pathlib import Path
from typing import Any

from aiobotocore.client import AioBaseClient
from botocore.exceptions import BotoCoreError, ClientError
import voluptuous as vol

from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_REGION, Platform
from homeassistant.core import (
    HomeAssistant,
    ServiceCall,
    ServiceResponse,
    SupportsResponse,
)
from homeassistant.exceptions import (
    ConfigEntryAuthFailed,
    ConfigEntryNotReady,
    HomeAssistantError,
    ServiceValidationError,
)
from homeassistant.helpers import config_validation as cv, selector
from homeassistant.helpers.typing import ConfigType

from .client import (
    async_get_session,
    create_client,
    error_message,
    is_auth_error,
    is_invalid_credentials,
)
from .const import CONF_CHAT_MODEL, CONF_PROMPT, DOMAIN, LOGGER, RECOMMENDED_CHAT_MODEL

SERVICE_GENERATE_CONTENT = "generate_content"
CONF_CONFIG_ENTRY = "config_entry"
CONF_IMAGE_FILENAME = "image_filename"
CONF_MODEL = "model"

PLATFORMS = (Platform.CONVERSATION,)
CONFIG_SCHEMA = cv.config_entry_only_config_schema(DOMAIN)

# Image formats accepted by the Bedrock Converse API, keyed by MIME type
IMAGE_FORMATS = {
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/gif": "gif",
    "image/webp": "webp",
}


@dataclass
class BedrockData:
    """Clients for talking to Amazon Bedrock."""

    runtime: AioBaseClient
    agent_runtime: AioBaseClient


type BedrockConfigEntry = ConfigEntry[BedrockData]


async def async_setup(hass: HomeAssistant, config: ConfigType) -> bool:
    """Set up Amazon Bedrock."""

    async def generate_content(call: ServiceCall) -> ServiceResponse:
        """Generate content from text and optionally images."""
        entry_id = call.data[CONF_CONFIG_ENTRY]
        entry: BedrockConfigEntry | None = hass.config_entries.async_get_entry(entry_id)

        if entry is None or entry.domain != DOMAIN:
            raise ServiceValidationError(
                translation_domain=DOMAIN,
                translation_key="invalid_config_entry",
                translation_placeholders={"config_entry": entry_id},
            )

        content: list[dict[str, Any]] = [{"text": call.data[CONF_PROMPT]}]

        def read_images() -> list[dict[str, Any]]:
            images = []
            for image_filename in call.data[CONF_IMAGE_FILENAME]:
                if not hass.config.is_allowed_path(image_filename):
                    raise HomeAssistantError(
                        f"Cannot read `{image_filename}`, no access to path; "
                        "`allowlist_external_dirs` may need to be adjusted in "
                        "`configuration.yaml`"
                    )
                path = Path(image_filename)
                if not path.exists():
                    raise HomeAssistantError(f"`{image_filename}` does not exist")
                mime_type, _ = mimetypes.guess_type(image_filename)
                if mime_type not in IMAGE_FORMATS:
                    raise HomeAssistantError(
                        f"`{image_filename}` is not a supported image "
                        "(jpeg, png, gif or webp)"
                    )
                images.append(
                    {
                        "image": {
                            "format": IMAGE_FORMATS[mime_type],
                            "source": {"bytes": path.read_bytes()},
                        }
                    }
                )
            return images

        content.extend(await hass.async_add_executor_job(read_images))

        model = call.data.get(CONF_MODEL) or entry.options.get(
            CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL
        )

        try:
            response = await entry.runtime_data.runtime.converse(
                modelId=model,
                messages=[{"role": "user", "content": content}],
            )
        except (ClientError, BotoCoreError) as err:
            if is_invalid_credentials(err):
                entry.async_start_reauth(hass)
            raise HomeAssistantError(
                f"Error generating content: {error_message(err)}"
            ) from err

        text = "".join(
            block["text"]
            for block in response["output"]["message"]["content"]
            if "text" in block
        )
        if not text:
            raise HomeAssistantError("Unknown error generating content")

        return {"text": text}

    hass.services.async_register(
        DOMAIN,
        SERVICE_GENERATE_CONTENT,
        generate_content,
        schema=vol.Schema(
            {
                vol.Required(CONF_CONFIG_ENTRY): selector.ConfigEntrySelector(
                    {"integration": DOMAIN}
                ),
                vol.Required(CONF_PROMPT): cv.string,
                vol.Optional(CONF_MODEL): cv.string,
                vol.Optional(CONF_IMAGE_FILENAME, default=[]): vol.All(
                    cv.ensure_list, [cv.string]
                ),
            }
        ),
        supports_response=SupportsResponse.ONLY,
    )
    return True


async def async_setup_entry(hass: HomeAssistant, entry: BedrockConfigEntry) -> bool:
    """Set up Amazon Bedrock from a config entry."""
    session = await async_get_session(hass)
    stack = AsyncExitStack()

    try:
        # Verify the credentials work before setting up the entities
        async with create_client(session, entry.data, "bedrock") as bedrock:
            await bedrock.list_foundation_models(byOutputModality="TEXT")

        entry.runtime_data = BedrockData(
            runtime=await stack.enter_async_context(
                create_client(session, entry.data, "bedrock-runtime")
            ),
            agent_runtime=await stack.enter_async_context(
                create_client(session, entry.data, "bedrock-agent-runtime")
            ),
        )
    except ClientError as err:
        await stack.aclose()
        if is_auth_error(err):
            raise ConfigEntryAuthFailed(
                f"Invalid AWS credentials: {error_message(err)}"
            ) from err
        raise ConfigEntryNotReady(error_message(err)) from err
    except BotoCoreError as err:
        await stack.aclose()
        raise ConfigEntryNotReady(error_message(err)) from err

    entry.async_on_unload(stack.aclose)

    LOGGER.debug("Connected to Amazon Bedrock in %s", entry.data[CONF_REGION])

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: BedrockConfigEntry) -> bool:
    """Unload Amazon Bedrock."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
