"""Tests for setting up the Amazon Bedrock integration."""

import builtins
import os
import threading
from typing import Any
from unittest.mock import AsyncMock, patch

from botocore.exceptions import ClientError, EndpointConnectionError
import pytest

from homeassistant.components.bedrock_agent.client import (
    _create_session,
    async_create_client,
)
from homeassistant.components.bedrock_agent.const import DOMAIN
from homeassistant.config_entries import ConfigEntryState
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant

from .conftest import ENTRY_DATA

from tests.common import MockConfigEntry


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "Boom"}}, "operation")


async def test_setup_and_unload(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test setting up and unloading the entry."""
    assert mock_config_entry.state is ConfigEntryState.LOADED
    assert mock_config_entry.runtime_data.runtime is mock_clients["bedrock-runtime"]
    assert (
        mock_config_entry.runtime_data.agent_runtime
        is mock_clients["bedrock-agent-runtime"]
    )
    assert hass.states.get("conversation.claude")
    assert hass.states.get("conversation.my_agent")
    assert hass.states.get("ai_task.nova")

    assert await hass.config_entries.async_unload(mock_config_entry.entry_id)
    assert mock_config_entry.state is ConfigEntryState.NOT_LOADED
    mock_clients["bedrock-runtime.context"].__aexit__.assert_awaited_once()
    mock_clients["bedrock-agent-runtime.context"].__aexit__.assert_awaited_once()


@pytest.mark.parametrize(
    ("side_effect", "state", "reauth"),
    [
        (
            _client_error("UnrecognizedClientException"),
            ConfigEntryState.SETUP_ERROR,
            True,
        ),
        (_client_error("AccessDeniedException"), ConfigEntryState.SETUP_ERROR, True),
        (_client_error("ThrottlingException"), ConfigEntryState.SETUP_RETRY, False),
        (
            EndpointConnectionError(endpoint_url="https://example.com"),
            ConfigEntryState.SETUP_RETRY,
            False,
        ),
    ],
)
async def test_setup_errors(
    hass: HomeAssistant,
    setup_ha: None,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, Any],
    side_effect: Exception,
    state: ConfigEntryState,
    reauth: bool,
) -> None:
    """Test errors while setting up the entry."""
    mock_clients["bedrock"] = AsyncMock()
    mock_clients["bedrock"].list_foundation_models.side_effect = side_effect

    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()

    assert mock_config_entry.state is state
    flows = hass.config_entries.flow.async_progress_by_handler(DOMAIN)
    assert [flow["context"]["source"] for flow in flows] == (
        ["reauth"] if reauth else []
    )


async def test_reload_on_subentry_update(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_init_component: None,
) -> None:
    """Test the entry is reloaded when a subentry is updated."""
    subentry = mock_config_entry.subentries["ai_task_id"]

    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={CONF_MODEL: "amazon.nova-pro-v1:0"},
    )
    await hass.async_block_till_done()

    assert mock_config_entry.state is ConfigEntryState.LOADED
    assert mock_config_entry.runtime_data.runtime  # Clients were recreated


async def test_create_clients_without_blocking_io(hass: HomeAssistant) -> None:
    """Test creating the real clients doesn't do blocking I/O in the event loop.

    botocore reads the service definitions from disk when a client is created,
    which must happen in the executor instead of the event loop.
    """
    hass.data["bedrock_agent_session"] = await hass.async_add_executor_job(
        _create_session
    )
    event_loop_thread = threading.current_thread()

    def block_in_event_loop(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if threading.current_thread() is event_loop_thread:
                raise AssertionError(f"Blocking I/O in the event loop: {args}")
            return func(*args, **kwargs)

        return wrapper

    with (
        patch("builtins.open", block_in_event_loop(builtins.open)),
        patch("os.listdir", block_in_event_loop(os.listdir)),
        patch("os.scandir", block_in_event_loop(os.scandir)),
    ):
        for service in (
            "bedrock",
            "bedrock-agent",
            "bedrock-agent-runtime",
            "bedrock-runtime",
        ):
            async with await async_create_client(hass, ENTRY_DATA, service) as client:
                assert client.meta.service_model.service_name == service
