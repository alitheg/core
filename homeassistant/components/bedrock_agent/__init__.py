"""The Amazon Bedrock integration."""

from __future__ import annotations

from dataclasses import dataclass
import mimetypes
from pathlib import Path
from typing import Any

import boto3
from botocore.client import BaseClient
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

from .const import (
    CONF_ACCESS_KEY_ID,
    CONF_CHAT_MODEL,
    CONF_PROMPT,
    CONF_SECRET_ACCESS_KEY,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
)

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

# The access key or secret is wrong, expired or has been revoked
INVALID_CREDENTIALS_CODES = {
    "InvalidSignatureException",
    "UnrecognizedClientException",
    "ExpiredTokenException",
}


@dataclass
class BedrockData:
    """Clients for talking to Amazon Bedrock."""

    runtime: BaseClient
    agent_runtime: BaseClient


type BedrockConfigEntry = ConfigEntry[BedrockData]


def create_client(data: dict[str, Any] | Any, service_name: str) -> BaseClient:
    """Create a boto3 client for a Bedrock service.

    This does blocking I/O and must be run in the executor.
    """
    session = boto3.Session(
        aws_access_key_id=data[CONF_ACCESS_KEY_ID],
        aws_secret_access_key=data[CONF_SECRET_ACCESS_KEY],
        region_name=data[CONF_REGION],
    )
    return session.client(service_name)


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
    """Return a human readable message for a boto error."""
    if isinstance(err, ClientError):
        return err.response.get("Error", {}).get("Message") or str(err)
    return str(err)


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
            response = await hass.async_add_executor_job(
                lambda: entry.runtime_data.runtime.converse(
                    modelId=model,
                    messages=[{"role": "user", "content": content}],
                )
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

    def create_clients() -> BedrockData:
        bedrock = create_client(entry.data, "bedrock")
        # Verify the credentials work before setting up the entities
        bedrock.list_foundation_models(byOutputModality="TEXT")
        return BedrockData(
            runtime=create_client(entry.data, "bedrock-runtime"),
            agent_runtime=create_client(entry.data, "bedrock-agent-runtime"),
        )

    try:
        entry.runtime_data = await hass.async_add_executor_job(create_clients)
    except ClientError as err:
        if is_auth_error(err):
            raise ConfigEntryAuthFailed(
                f"Invalid AWS credentials: {error_message(err)}"
            ) from err
        raise ConfigEntryNotReady(error_message(err)) from err
    except BotoCoreError as err:
        raise ConfigEntryNotReady(error_message(err)) from err

    LOGGER.debug("Connected to Amazon Bedrock in %s", entry.data[CONF_REGION])

    await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)

    return True


async def async_unload_entry(hass: HomeAssistant, entry: BedrockConfigEntry) -> bool:
    """Unload Amazon Bedrock."""
    return await hass.config_entries.async_unload_platforms(entry, PLATFORMS)
