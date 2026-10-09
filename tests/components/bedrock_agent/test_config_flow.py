"""Test the Amazon Bedrock config flow."""

from unittest.mock import MagicMock, patch

from botocore.exceptions import ClientError, EndpointConnectionError
import pytest

from homeassistant import config_entries
from homeassistant.components.bedrock_agent.config_flow import RECOMMENDED_OPTIONS
from homeassistant.components.bedrock_agent.const import (
    CONF_ACCESS_KEY_ID,
    CONF_AGENT_ALIAS_ID,
    CONF_AGENT_ID,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_RECOMMENDED,
    CONF_SECRET_ACCESS_KEY,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TOP_P,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_REGION
from homeassistant.core import HomeAssistant
from homeassistant.data_entry_flow import FlowResultType

from tests.common import MockConfigEntry

USER_INPUT = {
    CONF_REGION: "eu-central-1",
    CONF_ACCESS_KEY_ID: "AKIAEXAMPLE",
    CONF_SECRET_ACCESS_KEY: "secret",
}


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "Boom"}}, "operation")


async def test_form(
    hass: HomeAssistant,
    mock_clients: dict[str, MagicMock],
) -> None:
    """Test we get the form and create an entry."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )
    assert result["type"] is FlowResultType.FORM
    assert result["errors"] == {}

    result = await hass.config_entries.flow.async_configure(
        result["flow_id"], USER_INPUT
    )
    await hass.async_block_till_done()

    assert result["type"] is FlowResultType.CREATE_ENTRY
    assert result["title"] == "Amazon Bedrock (eu-central-1)"
    assert result["data"] == USER_INPUT
    assert result["options"] == RECOMMENDED_OPTIONS
    assert result["result"].state is config_entries.ConfigEntryState.LOADED


@pytest.mark.parametrize(
    ("side_effect", "error"),
    [
        (_client_error("UnrecognizedClientException"), "invalid_auth"),
        (_client_error("InvalidSignatureException"), "invalid_auth"),
        (_client_error("ThrottlingException"), "cannot_connect"),
        (EndpointConnectionError(endpoint_url="https://example.com"), "cannot_connect"),
        (Exception, "unknown"),
    ],
)
async def test_form_errors(
    hass: HomeAssistant,
    mock_clients: dict[str, MagicMock],
    side_effect: Exception,
    error: str,
) -> None:
    """Test we handle errors and can recover."""
    result = await hass.config_entries.flow.async_init(
        DOMAIN, context={"source": config_entries.SOURCE_USER}
    )

    with patch(
        "homeassistant.components.bedrock_agent.config_flow.validate_input",
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
    assert result["type"] is FlowResultType.CREATE_ENTRY


async def test_form_already_configured(
    hass: HomeAssistant,
    mock_clients: dict[str, MagicMock],
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


async def test_options_recommended(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_init_component,
) -> None:
    """Test the options flow with recommended settings."""
    hass.config_entries.async_update_entry(
        mock_config_entry, options=RECOMMENDED_OPTIONS
    )
    options_flow = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )
    assert options_flow["type"] is FlowResultType.FORM
    assert CONF_CHAT_MODEL not in options_flow["data_schema"].schema

    options = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        {
            CONF_PROMPT: "Speak like a pirate",
            CONF_LLM_HASS_API: "none",
            CONF_RECOMMENDED: True,
        },
    )
    await hass.async_block_till_done()

    assert options["type"] is FlowResultType.CREATE_ENTRY
    assert options["data"] == {
        CONF_PROMPT: "Speak like a pirate",
        CONF_RECOMMENDED: True,
    }


async def test_options_advanced(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, MagicMock],
    mock_init_component,
) -> None:
    """Test switching to advanced options and selecting a model and agent."""
    mock_clients["bedrock"].list_foundation_models.return_value = {
        "modelSummaries": [
            {
                "modelId": "amazon.nova-lite-v1:0",
                "modelName": "Nova Lite",
                "providerName": "Amazon",
                "modelLifecycle": {"status": "ACTIVE"},
            },
            {
                "modelId": "anthropic.claude-v2",
                "modelName": "Claude",
                "providerName": "Anthropic",
                "modelLifecycle": {"status": "LEGACY"},
            },
        ]
    }
    mock_clients["bedrock"].list_inference_profiles.return_value = {
        "inferenceProfileSummaries": [
            {
                "inferenceProfileId": "us.anthropic.claude-3-5-haiku-20241022-v1:0",
                "inferenceProfileName": "US Anthropic Claude 3.5 Haiku",
            }
        ]
    }
    mock_clients["bedrock-agent"] = MagicMock()
    mock_clients["bedrock-agent"].list_agents.return_value = {
        "agentSummaries": [{"agentId": "AGENT123", "agentName": "My agent"}]
    }
    hass.config_entries.async_update_entry(
        mock_config_entry, options=RECOMMENDED_OPTIONS
    )

    options_flow = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )
    # Toggle recommended off to show the advanced options
    options_flow = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        {
            CONF_PROMPT: "Speak like a pirate",
            CONF_LLM_HASS_API: "assist",
            CONF_RECOMMENDED: False,
        },
    )
    assert options_flow["type"] is FlowResultType.FORM
    schema = options_flow["data_schema"].schema
    model_selector = schema[CONF_CHAT_MODEL]
    assert [option["value"] for option in model_selector.config["options"]] == [
        "amazon.nova-lite-v1:0",
        "us.anthropic.claude-3-5-haiku-20241022-v1:0",
    ]
    assert schema[CONF_AGENT_ID].config["options"] == [
        {"value": "AGENT123", "label": "My agent"}
    ]

    options = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        {
            CONF_PROMPT: "Speak like a pirate",
            CONF_LLM_HASS_API: "assist",
            CONF_RECOMMENDED: False,
            CONF_CHAT_MODEL: "amazon.nova-lite-v1:0",
            CONF_MAX_TOKENS: 500,
            CONF_TOP_P: 0.5,
            CONF_TEMPERATURE: 0.3,
            CONF_AGENT_ID: "AGENT123",
            CONF_AGENT_ALIAS_ID: "ALIAS",
        },
    )
    await hass.async_block_till_done()

    assert options["type"] is FlowResultType.CREATE_ENTRY
    assert options["data"] == {
        CONF_PROMPT: "Speak like a pirate",
        CONF_LLM_HASS_API: "assist",
        CONF_RECOMMENDED: False,
        CONF_CHAT_MODEL: "amazon.nova-lite-v1:0",
        CONF_MAX_TOKENS: 500,
        CONF_TOP_P: 0.5,
        CONF_TEMPERATURE: 0.3,
        CONF_AGENT_ID: "AGENT123",
        CONF_AGENT_ALIAS_ID: "ALIAS",
    }


async def test_options_listing_fails(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, MagicMock],
    mock_init_component,
) -> None:
    """Test the advanced options still work when models can't be listed."""
    mock_clients["bedrock"].list_foundation_models.side_effect = _client_error(
        "AccessDeniedException"
    )
    mock_clients["bedrock-agent"] = MagicMock()
    mock_clients["bedrock-agent"].list_agents.side_effect = _client_error(
        "AccessDeniedException"
    )
    hass.config_entries.async_update_entry(
        mock_config_entry, options={CONF_RECOMMENDED: False}
    )

    options_flow = await hass.config_entries.options.async_init(
        mock_config_entry.entry_id
    )
    assert options_flow["type"] is FlowResultType.FORM

    options = await hass.config_entries.options.async_configure(
        options_flow["flow_id"],
        {
            CONF_PROMPT: "",
            CONF_LLM_HASS_API: "none",
            CONF_RECOMMENDED: False,
            CONF_CHAT_MODEL: RECOMMENDED_CHAT_MODEL,
            CONF_MAX_TOKENS: RECOMMENDED_MAX_TOKENS,
            CONF_TOP_P: RECOMMENDED_TOP_P,
            CONF_TEMPERATURE: 1.0,
            CONF_AGENT_ID: "",
            CONF_AGENT_ALIAS_ID: "TSTALIASID",
        },
    )

    assert options["type"] is FlowResultType.CREATE_ENTRY
    assert CONF_AGENT_ID not in options["data"]
    assert CONF_AGENT_ALIAS_ID not in options["data"]
