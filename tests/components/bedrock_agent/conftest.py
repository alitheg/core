"""Tests helpers."""

from collections.abc import Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from homeassistant.components.bedrock_agent.const import (
    CONF_ACCESS_KEY_ID,
    CONF_SECRET_ACCESS_KEY,
    DOMAIN,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_REGION
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.setup import async_setup_component

from tests.common import MockConfigEntry


@pytest.fixture
def mock_config_entry(hass: HomeAssistant) -> MockConfigEntry:
    """Mock a config entry."""
    entry = MockConfigEntry(
        title="Amazon Bedrock",
        domain=DOMAIN,
        data={
            CONF_REGION: "us-east-1",
            CONF_ACCESS_KEY_ID: "AKIAEXAMPLE",
            CONF_SECRET_ACCESS_KEY: "secret",
        },
    )
    entry.add_to_hass(hass)
    return entry


@pytest.fixture
def mock_config_entry_with_assist(
    hass: HomeAssistant, mock_config_entry: MockConfigEntry
) -> MockConfigEntry:
    """Mock a config entry with assist."""
    hass.config_entries.async_update_entry(
        mock_config_entry, options={CONF_LLM_HASS_API: llm.LLM_API_ASSIST}
    )
    return mock_config_entry


@pytest.fixture
def mock_clients() -> Generator[dict[str, AsyncMock]]:
    """Mock the aiobotocore clients, keyed by service name."""
    clients: dict[str, AsyncMock] = {}

    def create_client(service_name: str, **kwargs: str) -> MagicMock:
        context = MagicMock()
        context.__aenter__.return_value = clients.setdefault(service_name, AsyncMock())
        return context

    with patch(
        "homeassistant.components.bedrock_agent.client._create_session"
    ) as mock_create_session:
        mock_create_session.return_value.create_client.side_effect = create_client
        yield clients


@pytest.fixture
async def mock_init_component(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, AsyncMock],
) -> None:
    """Initialize integration."""
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()


@pytest.fixture(autouse=True)
async def setup_ha(hass: HomeAssistant) -> None:
    """Set up Home Assistant."""
    assert await async_setup_component(hass, "homeassistant", {})
