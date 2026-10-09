"""Tests for the Amazon Bedrock conversation agents."""

from typing import Any
from unittest.mock import AsyncMock, patch

from botocore.exceptions import ClientError
from freezegun import freeze_time
import probatio
import pytest

from homeassistant.components import conversation
from homeassistant.components.bedrock_agent.entity import _convert_content
from homeassistant.components.conversation import DATA_COMPONENT
from homeassistant.components.llm import LLMTools
from homeassistant.core import Context, HomeAssistant
from homeassistant.helpers import intent, llm

from . import async_iter, message, set_converse_stream, text_events, tool_use_events

from tests.common import MockConfigEntry


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "Boom"}}, "operation")


async def test_entity_features(hass: HomeAssistant, mock_init_component: None) -> None:
    """Test the conversation entities only control Home Assistant when enabled."""
    for entity_id in ("conversation.claude", "conversation.my_agent"):
        state = hass.states.get(entity_id)
        assert state
        assert state.attributes["supported_features"] == 0
        entity = hass.data[DATA_COMPONENT].get_entity(entity_id)
        assert entity
        assert entity.supported_languages == "*"


@pytest.mark.parametrize("enable_assist", [True])
async def test_entity_features_with_assist(
    hass: HomeAssistant, mock_init_component: None
) -> None:
    """Test the conversation entity can control Home Assistant."""
    state = hass.states.get("conversation.claude")
    assert state
    assert (
        state.attributes["supported_features"]
        == conversation.ConversationEntityFeature.CONTROL
    )


async def test_conversation(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test a conversation with history."""
    client = mock_clients["bedrock-runtime"]
    set_converse_stream(
        client,
        message(text_events("Ahoy "), text_events("matey!", index=1)),
        message(text_events("Arr")),
    )

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.claude"
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Ahoy matey!"
    kwargs = client.converse_stream.call_args.kwargs
    assert kwargs["modelId"] == "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    assert "You are a helpful assistant." in kwargs["system"][0]["text"]
    assert kwargs["inferenceConfig"] == {"maxTokens": 3000, "temperature": 1.0}
    assert "toolConfig" not in kwargs
    assert "outputConfig" not in kwargs
    assert kwargs["messages"] == [{"role": "user", "content": [{"text": "hello"}]}]

    result = await conversation.async_converse(
        hass,
        "what is your name?",
        result.conversation_id,
        Context(),
        agent_id="conversation.claude",
    )

    assert result.response.speech["plain"]["speech"] == "Arr"
    assert client.converse_stream.call_args.kwargs["messages"] == [
        {"role": "user", "content": [{"text": "hello"}]},
        {"role": "assistant", "content": [{"text": "Ahoy matey!"}]},
        {"role": "user", "content": [{"text": "what is your name?"}]},
    ]


async def test_reasoning_content(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test reasoning is added to the chat log but not sent back."""
    set_converse_stream(
        mock_clients["bedrock-runtime"],
        message(
            [
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"reasoningContent": {"text": "Thinking"}},
                    }
                },
                {
                    "contentBlockDelta": {
                        "contentBlockIndex": 0,
                        "delta": {"reasoningContent": {"signature": "abc"}},
                    }
                },
                {"contentBlockStop": {"contentBlockIndex": 0}},
            ],
            text_events("Done", index=1),
        ),
    )

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.claude"
    )

    assert result.response.speech["plain"]["speech"] == "Done"
    chat_log = hass.data[conversation.chat_log.DATA_CHAT_LOGS][result.conversation_id]
    assert chat_log.content[-1].thinking_content == "Thinking"
    assert _convert_content(chat_log.content)[-1] == {
        "role": "assistant",
        "content": [{"text": "Done"}],
    }


async def test_max_tokens(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test a warning is logged when the response is cut off."""
    set_converse_stream(
        mock_clients["bedrock-runtime"],
        message(text_events("Cut"), stop_reason="max_tokens"),
    )

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.claude"
    )

    assert result.response.speech["plain"]["speech"] == "Cut"
    assert "cut off at the maximum tokens" in caplog.text


@pytest.mark.parametrize(
    ("code", "reauth"),
    [("ExpiredTokenException", True), ("AccessDeniedException", False)],
)
async def test_error_handling(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
    code: str,
    reauth: bool,
) -> None:
    """Test errors are handled and only invalid credentials start reauth."""
    mock_clients["bedrock-runtime"].converse_stream.side_effect = _client_error(code)

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.claude"
    )
    await hass.async_block_till_done()

    assert result.response.response_type == intent.IntentResponseType.ERROR
    assert result.response.error_code == "unknown"
    flows = hass.config_entries.flow.async_progress_by_handler("bedrock_agent")
    assert bool(flows) is reauth


async def test_error_in_stream(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test errors that are raised while reading the stream."""

    async def failing_stream():
        yield {"messageStart": {"role": "assistant"}}
        raise _client_error("ModelStreamErrorException")

    mock_clients["bedrock-runtime"].converse_stream.side_effect = [
        {"stream": failing_stream()}
    ]

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.claude"
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR


async def test_template_error(
    hass: HomeAssistant,
    mock_config_entry: MockConfigEntry,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test that template errors are handled."""
    subentry = mock_config_entry.subentries["conversation_id"]
    hass.config_entries.async_update_subentry(
        mock_config_entry,
        subentry,
        data={**subentry.data, "prompt": "{% if True %}broken"},
    )
    await hass.async_block_till_done()

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.claude"
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR
    mock_clients["bedrock-runtime"].converse_stream.assert_not_called()


@pytest.mark.parametrize("enable_assist", [True])
@patch("homeassistant.components.llm.async_get_tools", new_callable=AsyncMock)
async def test_function_call(
    mock_get_tools: AsyncMock,
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test streamed tool calls are run and their results sent back."""
    mock_tool = AsyncMock()
    mock_tool.name = "test_tool"
    mock_tool.description = "Test function"
    mock_tool.parameters = probatio.Schema(
        {probatio.Optional("param1", description="Test parameters"): str}
    )
    mock_tool.async_call.side_effect = [
        llm.ToolResult(data={"answer": 42}),
        llm.ToolResult(data="Test response"),
    ]
    mock_get_tools.return_value = LLMTools(tools=[mock_tool])

    client = mock_clients["bedrock-runtime"]
    set_converse_stream(
        client,
        message(
            text_events("Let me check"),
            tool_use_events(1, "tool_1", "test_tool", {"param1": "test_value"}),
            tool_use_events(2, "tool_2", "test_tool", {}),
            stop_reason="tool_use",
        ),
        message(text_events("I have successfully called the function")),
    )

    context = Context()
    with freeze_time("2024-06-03 23:00:00"):
        result = await conversation.async_converse(
            hass,
            "Please call the test function",
            None,
            context,
            agent_id="conversation.claude",
        )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert (
        result.response.speech["plain"]["speech"]
        == "I have successfully called the function"
    )
    assert client.converse_stream.call_count == 2

    tool_specs = client.converse_stream.call_args.kwargs["toolConfig"]["tools"]
    assert {
        "toolSpec": {
            "name": "test_tool",
            "description": "Test function",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Test parameters"}
                    },
                    "additionalProperties": False,
                }
            },
        }
    } in tool_specs

    # Both tool results are sent back in a single user message
    assert client.converse_stream.call_args.kwargs["messages"][1:] == [
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
                {"toolUse": {"toolUseId": "tool_2", "name": "test_tool", "input": {}}},
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "tool_1",
                        "content": [{"json": {"answer": 42}}],
                        "status": "success",
                    }
                },
                {
                    "toolResult": {
                        "toolUseId": "tool_2",
                        "content": [{"json": {"result": "Test response"}}],
                        "status": "success",
                    }
                },
            ],
        },
    ]
    assert mock_tool.async_call.call_count == 2
    assert mock_tool.async_call.call_args_list[0].args[1] == llm.ToolInput(
        id="tool_1", tool_name="test_tool", tool_args={"param1": "test_value"}
    )


@pytest.mark.parametrize("enable_assist", [True])
async def test_invalid_tool_input(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test invalid JSON for the tool input is handled."""
    events = tool_use_events(0, "tool_1", "test_tool", {})
    events[1]["contentBlockDelta"]["delta"]["toolUse"]["input"] = "{not json"
    set_converse_stream(mock_clients["bedrock-runtime"], message(events))

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.claude"
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR


def test_convert_content() -> None:
    """Test converting the chat log to Converse API messages."""
    assert _convert_content(
        [
            conversation.SystemContent(content="system"),
            conversation.UserContent(content="first"),
            conversation.UserContent(content="second"),
            conversation.AssistantContent(
                agent_id="agent",
                content="",
                tool_calls=[
                    llm.ToolInput(
                        id="external",
                        tool_name="external_tool",
                        tool_args={},
                        external=True,
                    )
                ],
            ),
            conversation.ToolResultContent(
                agent_id="agent",
                tool_call_id="failed",
                tool_name="tool",
                result=llm.ToolResult(data={"error": "Boom"}, error=True),
            ),
            conversation.UserContent(content=""),
        ]
    ) == [
        {
            "role": "user",
            "content": [
                {"text": "first"},
                {"text": "second"},
                {
                    "toolResult": {
                        "toolUseId": "failed",
                        "content": [{"json": {"error": "Boom"}}],
                        "status": "error",
                    }
                },
            ],
        }
    ]


async def test_bedrock_agent(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test sending the conversation to a Bedrock Agent."""
    invoke_agent = mock_clients["bedrock-agent-runtime"].invoke_agent
    invoke_agent.return_value = {
        "completion": async_iter(
            [
                {"chunk": {"bytes": b"Hello "}},
                {"trace": {}},
                {"chunk": {"bytes": b"from the agent"}},
            ]
        )
    }

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.my_agent"
    )

    assert result.response.response_type == intent.IntentResponseType.ACTION_DONE
    assert result.response.speech["plain"]["speech"] == "Hello from the agent"
    invoke_agent.assert_awaited_once_with(
        agentId="AGENT123",
        agentAliasId="ALIAS",
        sessionId=result.conversation_id,
        inputText="hello",
        streamingConfigurations={"streamFinalResponse": True},
    )
    mock_clients["bedrock-runtime"].converse_stream.assert_not_called()


async def test_bedrock_agent_error(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test errors from a Bedrock Agent are handled."""
    mock_clients["bedrock-agent-runtime"].invoke_agent.side_effect = _client_error(
        "ResourceNotFoundException"
    )

    result = await conversation.async_converse(
        hass, "hello", None, Context(), agent_id="conversation.my_agent"
    )

    assert result.response.response_type == intent.IntentResponseType.ERROR
