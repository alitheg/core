"""Tests for the Amazon Bedrock integration."""

from unittest.mock import AsyncMock, patch

from aiobotocore.session import AioSession
from botocore.exceptions import ClientError, EndpointConnectionError
import pytest

from homeassistant.components.bedrock_agent.client import (
    SERVICES,
    _create_session,
    create_client,
)
from homeassistant.components.bedrock_agent.const import (
    CONF_ACCESS_KEY_ID,
    CONF_CHAT_MODEL,
    CONF_SECRET_ACCESS_KEY,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
)
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_REGION
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError, ServiceValidationError
from homeassistant.setup import async_setup_component

from tests.common import MockConfigEntry


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "Boom"}}, "operation")


def _converse_response(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
    }


async def test_setup_and_unload(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
    mock_init_component,
) -> None:
    """Test setting up and unloading the entry."""
    assert mock_config_entry.state is ConfigEntryState.LOADED
    assert mock_config_entry.runtime_data.runtime is mock_clients["bedrock-runtime"]
    assert (
        mock_config_entry.runtime_data.agent_runtime
        is mock_clients["bedrock-agent-runtime"]
    )
    assert hass.states.get("conversation.amazon_bedrock")

    assert await hass.config_entries.async_unload(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.NOT_LOADED


@pytest.mark.parametrize(
    ("side_effect", "state"),
    [
        (_client_error("UnrecognizedClientException"), ConfigEntryState.SETUP_ERROR),
        (_client_error("AccessDeniedException"), ConfigEntryState.SETUP_ERROR),
        (_client_error("ThrottlingException"), ConfigEntryState.SETUP_RETRY),
        (
            EndpointConnectionError(endpoint_url="https://example.com"),
            ConfigEntryState.SETUP_RETRY,
        ),
    ],
)
async def test_setup_errors(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
    side_effect: Exception,
    state: ConfigEntryState,
) -> None:
    """Test errors while setting up the entry."""
    mock_clients["bedrock"] = AsyncMock()
    mock_clients["bedrock"].list_foundation_models.side_effect = side_effect

    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()

    assert mock_config_entry.state is state
    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert [flow["context"]["source"] for flow in flows] == (
        ["reauth"] if state is ConfigEntryState.SETUP_ERROR else []
    )


async def test_generate_content(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
    mock_init_component,
) -> None:
    """Test generating content from text."""
    converse = mock_clients["bedrock-runtime"].converse
    converse.return_value = _converse_response("A haiku about lights")

    response = await hass.services.async_call(
        DOMAIN,
        "generate_content",
        {"config_entry": mock_config_entry.entry_id, "prompt": "Write a haiku"},
        blocking=True,
        return_response=True,
    )

    assert response == {"text": "A haiku about lights"}
    converse.assert_called_once_with(
        modelId=RECOMMENDED_CHAT_MODEL,
        messages=[{"role": "user", "content": [{"text": "Write a haiku"}]}],
    )


async def test_generate_content_with_images(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
    mock_init_component,
) -> None:
    """Test generating content from text and images."""
    hass.config_entries.async_update_entry(
        mock_config_entry, options={CONF_CHAT_MODEL: "amazon.nova-lite-v1:0"}
    )
    await hass.async_block_till_done()
    converse = mock_clients["bedrock-runtime"].converse
    converse.return_value = _converse_response("A cat and a dog")

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_bytes", side_effect=[b"jpeg", b"png"]),
        patch.object(hass.config, "is_allowed_path", return_value=True),
    ):
        response = await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {
                "config_entry": mock_config_entry.entry_id,
                "prompt": "Describe these images",
                "image_filename": ["cat.jpg", "dog.png"],
            },
            blocking=True,
            return_response=True,
        )

    assert response == {"text": "A cat and a dog"}
    converse.assert_called_once_with(
        modelId="amazon.nova-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": "Describe these images"},
                    {"image": {"format": "jpeg", "source": {"bytes": b"jpeg"}}},
                    {"image": {"format": "png", "source": {"bytes": b"png"}}},
                ],
            }
        ],
    )


async def test_generate_content_model_override(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
    mock_init_component,
) -> None:
    """Test overriding the model when generating content."""
    converse = mock_clients["bedrock-runtime"].converse
    converse.return_value = _converse_response("Hello")

    await hass.services.async_call(
        DOMAIN,
        "generate_content",
        {
            "config_entry": mock_config_entry.entry_id,
            "prompt": "Hi",
            "model": "mistral.mistral-large-2402-v1:0",
        },
        blocking=True,
        return_response=True,
    )

    assert converse.call_args.kwargs["modelId"] == "mistral.mistral-large-2402-v1:0"


@pytest.mark.parametrize(
    ("allowed", "exists", "filename", "message"),
    [
        (False, True, "cat.jpg", "no access to path"),
        (True, False, "cat.jpg", "does not exist"),
        (True, True, "cat.txt", "is not a supported image"),
    ],
)
async def test_generate_content_invalid_images(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_init_component,
    allowed: bool,
    exists: bool,
    filename: str,
    message: str,
) -> None:
    """Test generating content with invalid images."""
    with (
        patch("pathlib.Path.exists", return_value=exists),
        patch.object(hass.config, "is_allowed_path", return_value=allowed),
        pytest.raises(HomeAssistantError, match=message),
    ):
        await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {
                "config_entry": mock_config_entry.entry_id,
                "prompt": "Describe this",
                "image_filename": filename,
            },
            blocking=True,
            return_response=True,
        )


async def test_generate_content_errors(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
    mock_init_component,
) -> None:
    """Test errors when generating content."""
    converse = mock_clients["bedrock-runtime"].converse
    converse.side_effect = _client_error("ValidationException")

    with pytest.raises(HomeAssistantError, match="Error generating content: Boom"):
        await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {"config_entry": mock_config_entry.entry_id, "prompt": "Hi"},
            blocking=True,
            return_response=True,
        )

    converse.side_effect = None
    converse.return_value = _converse_response("")
    with pytest.raises(HomeAssistantError, match="Unknown error generating content"):
        await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {"config_entry": mock_config_entry.entry_id, "prompt": "Hi"},
            blocking=True,
            return_response=True,
        )


async def test_generate_content_invalid_config_entry(
    hass: HomeAssistant,
    mock_init_component,
) -> None:
    """Test generating content with a config entry from another integration."""
    other_entry = MockConfigEntry(domain="other")
    other_entry.add_to_hass(hass)

    with pytest.raises(ServiceValidationError):
        await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {"config_entry": other_entry.entry_id, "prompt": "Hi"},
            blocking=True,
            return_response=True,
        )


@pytest.mark.parametrize(
    ("code", "reauth"),
    [("UnrecognizedClientException", True), ("AccessDeniedException", False)],
)
async def test_generate_content_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
    mock_init_component,
    code: str,
    reauth: bool,
) -> None:
    """Test reauth is only started when the credentials are invalid."""
    mock_clients["bedrock-runtime"].converse.side_effect = _client_error(code)

    with pytest.raises(HomeAssistantError):
        await hass.services.async_call(
            DOMAIN,
            "generate_content",
            {"config_entry": mock_config_entry.entry_id, "prompt": "Hi"},
            blocking=True,
            return_response=True,
        )
    await hass.async_block_till_done()

    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert bool(flows) is reauth


async def test_create_session_preloads_service_data() -> None:
    """Test creating clients from the session doesn't do blocking I/O.

    The session is created in the executor and should load everything the
    clients need, so creating them in the event loop doesn't touch the disk.
    """
    session = _create_session()
    assert isinstance(session, AioSession)

    blocked = AssertionError("Blocking I/O in the event loop")
    with (
        patch("builtins.open", side_effect=blocked),
        patch("os.listdir", side_effect=blocked),
        patch("os.scandir", side_effect=blocked),
    ):
        for service in SERVICES:
            async with create_client(
                session,
                {
                    CONF_REGION: "us-east-1",
                    CONF_ACCESS_KEY_ID: "AKIAEXAMPLE",
                    CONF_SECRET_ACCESS_KEY: "secret",
                },
                service,
            ) as client:
                assert client.meta.service_model.service_name == service
