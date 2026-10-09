"""Tests helpers."""

from collections.abc import Generator
from unittest.mock import MagicMock, patch

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
def mock_clients() -> Generator[dict[str, MagicMock]]:
    """Mock the boto3 clients, keyed by service name."""
    clients: dict[str, MagicMock] = {}

    def client(service_name: str) -> MagicMock:
        return clients.setdefault(service_name, MagicMock())

    with patch("homeassistant.components.bedrock_agent.boto3.Session") as mock_session:
        mock_session.return_value.client.side_effect = client
        yield clients


@pytest.fixture
async def mock_init_component(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, MagicMock],
) -> None:
    """Initialize integration."""
    assert await async_setup_component(hass, DOMAIN, {})
    await hass.async_block_till_done()


@pytest.fixture(autouse=True)
async def setup_ha(hass: HomeAssistant) -> None:
    """Set up Home Assistant."""
    assert await async_setup_component(hass, "homeassistant", {})
