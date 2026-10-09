"""Tests for the Amazon Bedrock conversation agent."""

from unittest.mock import AsyncMock, MagicMock, patch

from botocore.exceptions import ClientError
import voluptuous as vol

from homeassistant.components import conversation
from homeassistant.components.bedrock_agent.const import (
    CONF_AGENT_ALIAS_ID,
    CONF_AGENT_ID,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
)
from homeassistant.components.bedrock_agent.conversation import _convert_content
from homeassistant.const import CONF_LLM_HASS_API
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent, llm

from tests.common import MockConfigEntry


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "Boom"}}, "operation")


def _converse_response(*content: dict) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": list(content)}},
        "stopReason": (
            "tool_use" if any("toolUse" in block for block in content) else "end_turn"
        ),
    }


async def test_entity(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_init_component,
) -> None:
    """Test entity properties."""
    state = hass.states.get("conversation.amazon_bedrock")
    assert state
    assert state.attributes["supported_features"] == 0

    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={**mock_config_entry.options, CONF_LLM_HASS_API: "assist"},
    )
    await hass.async_block_till_done()

    state = hass.states.get("conversation.amazon_bedrock")
    assert state
    assert (
        state.attributes["supported_features"]
        == conversation.ConversationEntityFeature.CONTROL
    )

    # A Bedrock Agent brings its own tools, so it can't control Home Assistant
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={**mock_config_entry.options, CONF_AGENT_ID: "AGENT123"},
    )
    await hass.async_block_till_done()

    state = hass.states.get("conversation.amazon_bedrock")
    assert state
    assert state.attributes["supported_features"] == 0


async def test_conversation_agent(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, MagicMock],
    mock_init_component,
) -> None:
    """Test a conversation with history."""
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={
            CONF_PROMPT: "Speak like a pirate",
            CONF_CHAT_MODEL: "amazon.nova-lite-v1:0",
            CONF_MAX_TOKENS: 100,
            CONF_TEMPERATURE: 0.5,
            CONF_TOP_P: 0.9,
        },
    )
    await hass.async_block_till_done()
    converse = mock_clients["bedrock-runtime"].converse
    converse.return_value = _converse_response({"text": "Ahoy matey!"})

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.amazon_bedrock"
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Ahoy matey!"

    kwargs = converse.call_args.kwargs
    assert kwargs["modelId"] == "amazon.nova-lite-v1:0"
    assert "Speak like a pirate" in kwargs["system"][0]["text"]
    assert kwargs["inferenceConfig"] == {
        "maxTokens": 100,
        "temperature": 0.5,
        "topP": 0.9,
    }
    assert "toolConfig" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]

    converse.return_value = _converse_response({"text": "Arr, me name be Bedrock"})
    result = await conversation.async_converse(
        hass,
        "what is your name?",
        result.conversation_id,
        Context(),
        agent_id="conversation.amazon_bedrock",
    )

    assert result.response.speech["plain"]["speech"] == "Arr, me name be Bedrock"
    assert converse.call_args.kwargs["messages"] == [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "assistant", "content": [{"text": "Ahoy matey!"}]},
        {"role": "user", "content": [{"text": "what is your name?"}]},
    ]


async def test_error_handling(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, MagicMock],
    mock_init_component,
) -> None:
    """Test that we handle errors when calling the Converse API."""
    mock_clients["bedrock-runtime"].converse.side_effect = _client_error(
        "ThrottlingException"
    )

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id=mock_config_entry.entry_id
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR, result
    assert result.response.error_code == "unknown", result


async def test_template_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_init_component,
) -> None:
    """Test that template error handling works."""
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={
            CONF_PROMPT: "talk like a {% if True %}smarthome{% else %}pirate please."
        },
    )
    await hass.async_block_till_done()

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id=mock_config_entry.entry_id
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR, result
    assert result.response.error_code == "unknown", result


@patch(
    "homeassistant.components.bedrock_agent.conversation.llm.AssistAPI._async_get_tools"
)
async def test_function_call(
    mock_get_tools,
    hass: HomeAssistant,
    mock_config_entry_with_assist: MockConfigEntry,
    mock_clients: dict[str, MagicMock],
    mock_init_component,
) -> None:
    """Test function call from the assistant."""
    agent_id = mock_config_entry_with_assist.entry_id
    context = Context()

    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test function"
    mock_tool.parameters = vol.Schema(
        {vol.Optional("param1", description="Test parameters"): str}
    )
    mock_tool.async_call.return_value = "Test response"

    mock_get_tools.return_value = [mock_tool]

    converse = mock_clients["bedrock-runtime"].converse
    converse.side_effect = [
        _converse_response(
            {"text": "Let me check"},
            {
                "toolUse": {
                    "toolUseId": "tool_1",
                    "name": "test_tool",
                    "input": {"param1": "test_value"},
                }
            },
            {
                "toolUse": {
                    "toolUseId": "tool_2",
                    "name": "test_tool",
                    "input": {"param1": "other_value"},
                }
            },
        ),
        _converse_response({"text": "I have successfully called the function"}),
    ]

    result = await conversation.async_converse(
        hass, "Please call the test function", None, context, agent_id=agent_id
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert (
        result.response.speech["plain"]["speech"]
        == "I have successfully called the function"
    )
    assert converse.call_count == 2

    tool_config = converse.call_args.kwargs["toolConfig"]
    assert tool_config == {
        "tools": [
            {
                "toolSpec": {
                    "name": "test_tool",
                    "description": "Test function",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "param1": {
                                    "type": "string",
                                    "description": "Test parameters",
                                }
                            },
                            "required": [],
                        }
                    },
                }
            }
        ]
    }

    # Both tool results are sent back in a single user message
    assert converse.call_args.kwargs["messages"][1:] == [
        {
            "role": "assistant",
            "content": [
                {"text": "Let me check"},
                {
                    "toolUse": {
                        "toolUseId": "tool_1",
                        "name": "test_tool",
                        "input": {"param1": "test_value"},
                    }
                },
                {
                    "toolUse": {
                        "toolUseId": "tool_2",
                        "name": "test_tool",
                        "input": {"param1": "other_value"},
                    }
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tool_1",
                        "content": [{"json": "Test response"}],
                    }
                },
                {
                    "toolResult": {
                        "toolUseId": "tool_2",
                        "content": [{"json": "Test response"}],
                    }
                },
            ],
        },
    ]

    assert mock_tool.async_call.call_count == 2
    mock_tool.async_call.assert_any_call(
        hass,
        llm.ToolInput(
            id="tool_1", tool_name="test_tool", tool_args={"param1": "test_value"}
        ),
        llm.LLMContext(
            platform="bedrock_agent",
            context=context,
            user_prompt="Please call the test function",
            language="en",
            assistant="conversation",
            device_id=None,
        ),
    )


async def test_bedrock_agent(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, MagicMock],
    mock_init_component,
) -> None:
    """Test sending the conversation to a Bedrock Agent."""
    hass.config_entries.async_update_entry(
        mock_config_entry,
        options={CONF_AGENT_ID: "AGENT123", CONF_AGENT_ALIAS_ID: "ALIAS"},
    )
    await hass.async_block_till_done()
    invoke_agent = mock_clients["bedrock-agent-runtime"].invoke_agent
    invoke_agent.return_value = {
        "completion": [
            {"chunk": {"bytes": b"Hello "}},
            {"trace": {}},
            {"chunk": {"bytes": b"from the agent"}},
        ]
    }

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id=mock_config_entry.entry_id
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Hello from the agent"
    invoke_agent.assert_called_once_with(
        agentId="AGENT123",
        agentAliasId="ALIAS",
        sessionId=result.conversation_id,
        inputText="hello",
    )
    mock_clients["bedrock-runtime"].converse.assert_not_called()

    invoke_agent.side_effect = _client_error("ResourceNotFoundException")
    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id=mock_config_entry.entry_id
    )
    assert result.response.response_type == intent.IntentResponseType.ERROR, result


def test_convert_content_merges_roles() -> None:
    """Test consecutive content with the same role is merged."""
    assert _convert_content(
        [
            conversation.SystemContent(content="system"),
            conversation.UserContent(content="first"),
            conversation.UserContent(content="second"),
            conversation.AssistantContent(agent_id="agent", content=""),
            conversation.UserContent(content="third"),
        ]
    ) == [
        {
            "role": "user",
            "content": [{"text": "first"}, {"text": "second"}, {"text": "third"}],
        }
    ]
