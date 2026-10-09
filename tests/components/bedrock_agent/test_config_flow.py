"""Test the Amazon Bedrock config flow."""

from typing import Any
from unittest.mock import AsyncMock, patch

from botocore.exceptions import ClientError, EndpointConnectionError
import pytest

from homeassistant import config_entries
from homeassistant.components.bedrock_agent.const import (
    CONF_ACCESS_KEY_ID,
    CONF_AGENT_ALIAS_ID,
    CONF_AGENT_ID,
    CONF_MAX_TOKENS,
    CONF_SECRET_ACCESS_KEY,
    CONF_TEMPERATURE,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DOMAIN,
)
from homeassistant.const import (
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_NAME,
    CONF_PROMPT,
    CONF_REGION,
)
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType
from homeassistant.helpers import llm

from .conftest import ENTRY_DATA

from tests.common import MockConfigEntry

USER_INPUT = {
    CONF_REGION: "eu-central-1",
    CONF_ACCESS_KEY_ID: "AKIAEXAMPLE",
    CONF_SECRET_ACCESS_KEY: "secret",
}
HAIKU = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "Boom"}}, "operation")


@pytest.fixture(autouse=True)
def mock_setup_entry():
    """Mock setting up the entry."""
    with patch(
        "homeassistant.components.bedrock_agent.async_setup_entry",
        return_value=True,
    ) as mock_setup_entry:
        yield mock_setup_entry


async def test_form(
    hass: HomeAssistant, mock_clients: dict[str, Any], mock_setup_entry: AsyncMock
) -> None:
    """Test creating an entry with the default agents."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "user"

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model"
    assert result["errors"] == {}
    model_selector = result["data_schema"].schema[CONF_MODEL]
    # Legacy models and models without streaming are left out
    assert model_selector.config["options"] == [
        {"value": "amazon.nova-lite-v1:0", "label": "Amazon Nova Lite"},
        {"value": HAIKU, "label": "US Anthropic Claude Haiku 4.5"},
    ]
    # The preferred model is selected by default
    assert result["data_schema"]({})[CONF_MODEL] == HAIKU

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: HAIKU}
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Amazon Bedrock (eu-central-1)"
    assert result["data"] == USER_INPUT
    subentries = [
        (subentry.subentry_type, subentry.title, dict(subentry.data))
        for subentry in result["result"].subentries.values()
    ]
    assert subentries == [
        (
            "conversation",
            DEFAULT_CONVERSATION_NAME,
            {
                CONF_MODEL: HAIKU,
                CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
                CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
            },
        ),
        ("ai_task_data", DEFAULT_AI_TASK_NAME, {CONF_MODEL: HAIKU}),
    ]
    assert len(mock_setup_entry.mock_calls) == 1


async def test_form_list_models_fails(
    hass: HomeAssistant, mock_clients: dict[str, Any]
) -> None:
    """Test a model ID can be entered when the models can't be listed."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    mock_clients[
        "bedrock"
    ].get_paginator.return_value.paginate.side_effect = _client_error(
        "AccessDeniedException"
    )
    result = await hass.config_entries.flow.async_configure(result["flow_id"])
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "model"
    assert result["errors"] == {"base": "list_models_failed"}
    assert result["data_schema"].schema[CONF_MODEL].config["options"] == []

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], {CONF_MODEL: "my.custom-model"}
    )
    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert all(
        subentry.data[CONF_MODEL] == "my.custom-model"
        for subentry in result["result"].subentries.values()
    )


@pytest.mark.parametrize(
    ("side_effect", "error"),
    [
        (_client_error("UnrecognizedClientException"), "invalid_auth"),
        (_client_error("AccessDeniedException"), "invalid_auth"),
        (_client_error("ThrottlingException"), "cannot_connect"),
        (EndpointConnectionError(endpoint_url="https://example.com"), "cannot_connect"),
        (Exception, "unknown"),
    ],
)
async def test_form_errors(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    side_effect: Exception,
    error: str,
) -> None:
    """Test we handle errors and can recover."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "homeassistant.components.bedrock_agent.config_flow.async_validate_credentials",
        side_effect=side_effect,
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"], USER_INPUT
        )

    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": error}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    assert result["step_id"] == "model"


async def test_form_already_configured(
    hass: HomeAssistant, mock_clients: dict[str, Any]
) -> None:
    """Test we abort if the account is already configured in the region."""
    MockConfigEntry(domain=DOMAIN, data=USER_INPUT).add_to_hass(hass)

    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "already_configured"


async def test_reauth(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, Any],
) -> None:
    """Test re-authenticating with new credentials."""
    mock_config_entry.add_to_hass(hass)
    result = await mock_config_entry.start_reauth_flow(hass)
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "reauth_confirm"
    assert result["description_placeholders"]["region"] == "us-east-1"

    with patch(
        "homeassistant.components.bedrock_agent.config_flow.async_validate_credentials",
        side_effect=_client_error("UnrecognizedClientException"),
    ):
        result = await hass.config_entries.flow.async_configure(
            result["flow_id"],
            {CONF_ACCESS_KEY_ID: "AKIANEW", CONF_SECRET_ACCESS_KEY: "wrong"},
        )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {"base": "invalid_auth"}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"],
        {CONF_ACCESS_KEY_ID: "AKIANEW", CONF_SECRET_ACCESS_KEY: "new-secret"},
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reauth_successful"
    assert mock_config_entry.data == {
        **ENTRY_DATA,
        CONF_ACCESS_KEY_ID: "AKIANEW",
        CONF_SECRET_ACCESS_KEY: "new-secret",
    }


@pytest.fixture
async def loaded_entry(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, Any],
    mock_setup_entry: AsyncMock,
) -> MockConfigEntry:
    """Return a config entry that is loaded, without setting up the platforms."""
    mock_config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(mock_config_entry.entry_id)
    await hass.async_block_till_done()
    assert mock_config_entry.state is config_entries.ConfigEntryState.LOADED
    return mock_config_entry


async def test_add_conversation_agent(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Test adding a conversation agent."""
    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "conversation"),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["step_id"] == "init"
    defaults = result["data_schema"]({})
    assert defaults[CONF_NAME] == DEFAULT_CONVERSATION_NAME
    assert defaults[CONF_MODEL] == HAIKU
    assert defaults[CONF_LLM_HASS_API] == [llm.LLM_API_ASSIST]

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_NAME: "Nova",
            CONF_MODEL: "amazon.nova-lite-v1:0",
            CONF_PROMPT: "Speak like a pirate",
            CONF_LLM_HASS_API: [],
            CONF_TEMPERATURE: 0.5,
            CONF_MAX_TOKENS: 500,
        },
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Nova"
    assert result["data"] == {
        CONF_MODEL: "amazon.nova-lite-v1:0",
        CONF_PROMPT: "Speak like a pirate",
        CONF_TEMPERATURE: 0.5,
        CONF_MAX_TOKENS: 500,
    }


async def test_reconfigure_conversation_agent(
    hass: HomeAssistant, loaded_entry: MockConfigEntry
) -> None:
    """Test reconfiguring a conversation agent."""
    result = await loaded_entry.start_subentry_reconfigure_flow(hass, "conversation_id")
    assert result["type"] is FlowResultType.FORM
    assert CONF_NAME not in result["data_schema"].schema
    assert result["data_schema"]({})[CONF_MODEL] == HAIKU

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {
            CONF_MODEL: "amazon.nova-lite-v1:0",
            CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        },
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "reconfigure_successful"
    assert loaded_entry.subentries["conversation_id"].data == {
        CONF_MODEL: "amazon.nova-lite-v1:0",
        CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
        CONF_TEMPERATURE: 1.0,
        CONF_MAX_TOKENS: 3000,
    }


async def test_add_ai_task(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    mock_clients: dict[str, Any],
) -> None:
    """Test adding an AI task, also when the models can't be listed."""
    mock_clients["bedrock"] = AsyncMock()
    mock_clients["bedrock"].list_foundation_models.side_effect = _client_error(
        "ThrottlingException"
    )
    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "ai_task_data"),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert CONF_PROMPT not in result["data_schema"].schema
    assert result["data_schema"].schema[CONF_MODEL].config["options"] == []

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_NAME: "Task", CONF_MODEL: "amazon.nova-pro-v1:0"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Task"
    assert result["data"] == {CONF_MODEL: "amazon.nova-pro-v1:0", CONF_MAX_TOKENS: 3000}


@pytest.mark.parametrize("list_agents_fails", [False, True])
async def test_add_bedrock_agent(
    hass: HomeAssistant,
    loaded_entry: MockConfigEntry,
    mock_clients: dict[str, Any],
    list_agents_fails: bool,
) -> None:
    """Test adding a Bedrock Agent."""
    mock_clients["bedrock-agent"] = AsyncMock()
    if list_agents_fails:
        mock_clients["bedrock-agent"].list_agents.side_effect = _client_error(
            "AccessDeniedException"
        )
    else:
        mock_clients["bedrock-agent"].list_agents.return_value = {
            "agentSummaries": [{"agentId": "AGENT123", "agentName": "My agent"}]
        }

    result = await hass.config_entries.subentries.async_init(
        (loaded_entry.entry_id, "agent"),
        context={"source": config_entries.SOURCE_USER},
    )
    assert result["type"] is FlowResultType.FORM
    assert result["data_schema"].schema[CONF_AGENT_ID].config["options"] == (
        [] if list_agents_fails else [{"value": "AGENT123", "label": "My agent"}]
    )

    result = await hass.config_entries.subentries.async_configure(
        result["flow_id"],
        {CONF_NAME: "Agent", CONF_AGENT_ID: "AGENT123", CONF_AGENT_ALIAS_ID: "A1"},
    )

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["data"] == {CONF_AGENT_ID: "AGENT123", CONF_AGENT_ALIAS_ID: "A1"}


@pytest.mark.parametrize("subentry_type", ["conversation", "ai_task_data", "agent"])
async def test_subentry_entry_not_loaded(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    subentry_type: str,
) -> None:
    """Test adding a subentry when the entry is not loaded."""
    mock_config_entry.add_to_hass(hass)

    result = await hass.config_entries.subentries.async_init(
        (mock_config_entry.entry_id, subentry_type),
        context={"source": config_entries.SOURCE_USER},
    )

    assert result["type"] is FlowResultType.ABORT
    assert result["reason"] == "entry_not_loaded"
