"""Fixtures for the Amazon Bedrock tests."""

from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.components.bedrock_agent.const import (
    CONF_ACCESS_KEY_ID,
    CONF_AGENT_ALIAS_ID,
    CONF_AGENT_ID,
    CONF_SECRET_ACCESS_KEY,
    DOMAIN,
)
from homeassistant.config_entries import ConfigSubentryData
from homeassistant.const import CONF_LLM_HASS_API, CONF_MODEL, CONF_PROMPT, CONF_REGION
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component

from . import async_iter, message, set_converse_stream, text_events

from tests.common import MockConfigEntry

ENTRY_DATA = {
    CONF_REGION: "us-east-1",
    CONF_ACCESS_KEY_ID: "AKIAEXAMPLE",
    CONF_SECRET_ACCESS_KEY: "secret",
}

MODEL_SUMMARIES = [
    {
        "modelId": "amazon.nova-lite-v1:0",
        "modelName": "Nova Lite",
        "providerName": "Amazon",
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "ACTIVE"},
    },
    {
        "modelId": "anthropic.claude-v2",
        "modelName": "Claude",
        "providerName": "Anthropic",
        "responseStreamingSupported": True,
        "modelLifecycle": {"status": "LEGACY"},
    },
    {
        "modelId": "example.no-streaming-v1:0",
        "modelName": "No streaming",
        "providerName": "Example",
        "responseStreamingSupported": False,
    },
]

INFERENCE_PROFILES = [
    {
        "inferenceProfileId": "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "inferenceProfileName": "US Anthropic Claude Haiku 4.5",
        "status": "ACTIVE",
    }
]


@pytest.fixture
def enable_assist() -> bool:
    """Return if the conversation agent can control Home Assistant."""
    return False


@pytest.fixture
def mock_config_entry(hass: HomeAssistant, enable_assist: bool) -> MockConfigEntry:
    """Mock a config entry with a conversation agent, AI task and Bedrock Agent."""
    conversation_data: dict[str, Any] = {
        CONF_MODEL: "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        CONF_PROMPT: "You are a helpful assistant.",
    }
    if enable_assist:
        conversation_data[CONF_LLM_HASS_API] = [llm.LLM_API_ASSIST]
    return MockConfigEntry(
        title="Amazon Bedrock (us-east-1)",
        domain=DOMAIN,
        data=ENTRY_DATA,
        subentries_data=[
            ConfigSubentryData(
                data=conversation_data,
                subentry_id="conversation_id",
                subentry_type="conversation",
                title="Claude",
                unique_id=None,
            ),
            ConfigSubentryData(
                data={CONF_MODEL: "amazon.nova-lite-v1:0"},
                subentry_id="ai_task_id",
                subentry_type="ai_task_data",
                title="Nova",
                unique_id=None,
            ),
            ConfigSubentryData(
                data={CONF_AGENT_ID: "AGENT123", CONF_AGENT_ALIAS_ID: "ALIAS"},
                subentry_id="agent_id",
                subentry_type="agent",
                title="My agent",
                unique_id=None,
            ),
        ],
    )


@pytest.fixture
def mock_clients() -> Generator[dict[str, AsyncMock]]:
    """Mock the aiobotocore clients, keyed by service name.

    The client contexts are stored with the key `<service>.context`.
    """
    clients: dict[str, Any] = {}

    def create_client(service_name: str, **kwargs: Any) -> MagicMock:
        client = clients.setdefault(service_name, _create_mock_client(service_name))
        context = MagicMock()
        context.__aenter__.return_value = client
        context.__aexit__.return_value = False
        clients[f"{service_name}.context"] = context
        return context

    with patch(
        "homeassistant.components.bedrock_agent.client._create_session"
    ) as mock_create_session:
        mock_create_session.return_value.create_client.side_effect = create_client
        yield clients


def _create_mock_client(service_name: str) -> AsyncMock:
    """Create a mock client with default responses."""
    client = AsyncMock()
    if service_name == "bedrock":
        client.list_foundation_models.return_value = {"modelSummaries": MODEL_SUMMARIES}
        client.get_paginator = MagicMock()
        client.get_paginator.return_value.paginate.side_effect = lambda **kwargs: (
            async_iter([{"inferenceProfileSummaries": INFERENCE_PROFILES}])
        )
    elif service_name == "bedrock-agent":
        client.list_agents.return_value = {
            "agentSummaries": [{"agentId": "AGENT123", "agentName": "My agent"}]
        }
    elif service_name == "bedrock-runtime":
        set_converse_stream(client, message(text_events("Hello!")))
    return client


@pytest.fixture(autouse=True)
async def setup_ha(hass: HomeAssistant) -> None:
    """Set up Home Assistant."""
    assert await async_setup_component(hass, "homeassistant", {})


@pytest.fixture
async def mock_init_component(
    hass: HomeAssistant,
    setup_ha: None,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
) -> None:
    """Set up the integration."""
    mock_config_entry.add_to_hass(hass)
    assert await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
