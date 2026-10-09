"""Conversation support for Amazon Bedrock."""

from collections.abc import Callable
from functools import partial
from typing import Any, Literal

from botocore.exceptions import BotoCoreError, ClientError
from voluptuous_openapi import convert

from homeassistant.components import assist_pipeline, conversation
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_LLM_HASS_API, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import chat_session, device_registry as dr, intent, llm
from homeassistant.helpers.entity_platform import AddEntitiesCallback

from . import BedrockConfigEntry, error_message
from .const import (
    CONF_AGENT_ALIAS_ID,
    CONF_AGENT_ID,
    CONF_CHAT_MODEL,
    CONF_MAX_TOKENS,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_TOP_P,
    DEFAULT_AGENT_ALIAS_ID,
    DOMAIN,
    LOGGER,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
)

PARALLEL_UPDATES = 0

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: BedrockConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    async_add_entities([BedrockConversationEntity(config_entry)])


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format tool specification for the Converse API."""
    tool_spec: dict[str, Any] = {
        "name": tool.name,
        "inputSchema": {
            "json": convert(tool.parameters, custom_serializer=custom_serializer)
        },
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"toolSpec": tool_spec}


def _convert_content(
    chat_content: list[conversation.Content],
) -> list[dict[str, Any]]:
    """Convert the chat log to Converse API messages.

    The Converse API requires messages to alternate between the user and the
    assistant, with tool results sent as user messages, so consecutive content
    with the same role is merged into a single message.
    """
    messages: list[dict[str, Any]] = []

    for content in chat_content:
        role: str
        blocks: list[dict[str, Any]] = []

        if isinstance(content, conversation.SystemContent):
            # The system prompt is passed separately
            continue
        if isinstance(content, conversation.ToolResultContent):
            role = "user"
            blocks.append(
                {
                    "toolResult": {
                        "toolUseId": content.tool_call_id,
                        "content": [{"json": content.tool_result}],
                    }
                }
            )
        elif isinstance(content, conversation.UserContent):
            role = "user"
            if content.content:
                blocks.append({"text": content.content})
        else:
            role = "assistant"
            if content.content:
                blocks.append({"text": content.content})
            blocks.extend(
                {
                    "toolUse": {
                        "toolUseId": tool_call.id,
                        "name": tool_call.tool_name,
                        "input": tool_call.tool_args,
                    }
                }
                for tool_call in content.tool_calls or ()
            )

        if not blocks:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    return messages


class BedrockConversationEntity(
    conversation.ConversationEntity, conversation.AbstractConversationAgent
):
    """Amazon Bedrock conversation agent."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: BedrockConfigEntry) -> None:
        """Initialize the agent."""
        self.entry = entry
        self._attr_unique_id = entry.entry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, entry.entry_id)},
            name=entry.title,
            manufacturer="Amazon",
            model=(
                "Bedrock Agent"
                if entry.options.get(CONF_AGENT_ID)
                else entry.options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL)
            ),
            entry_type=dr.DeviceEntryType.SERVICE,
        )
        if entry.options.get(CONF_LLM_HASS_API) and not entry.options.get(
            CONF_AGENT_ID
        ):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    async def async_added_to_hass(self) -> None:
        """When entity is added to Home Assistant."""
        await super().async_added_to_hass()
        assist_pipeline.async_migrate_engine(
            self.hass, "conversation", self.entry.entry_id, self.entity_id
        )
        conversation.async_set_agent(self.hass, self.entry, self)
        self.entry.async_on_unload(
            self.entry.add_update_listener(self._async_entry_update_listener)
        )

    async def async_will_remove_from_hass(self) -> None:
        """When entity will be removed from Home Assistant."""
        conversation.async_unset_agent(self.hass, self.entry)
        await super().async_will_remove_from_hass()

    async def async_process(
        self, user_input: conversation.ConversationInput
    ) -> conversation.ConversationResult:
        """Process a sentence."""
        with (
            chat_session.async_get_chat_session(
                self.hass, user_input.conversation_id
            ) as session,
            conversation.async_get_chat_log(self.hass, session, user_input) as chat_log,
        ):
            if self.entry.options.get(CONF_AGENT_ID):
                return await self._async_handle_agent(user_input, chat_log)
            return await self._async_handle_message(user_input, chat_log)

    async def _async_handle_agent(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Send the message to a Bedrock Agent.

        Bedrock Agents keep the conversation history server side, keyed by the
        session ID, and bring their own tools and knowledge bases.
        """
        assert user_input.agent_id
        options = self.entry.options
        client = self.entry.runtime_data.agent_runtime

        def invoke_agent() -> str:
            response = client.invoke_agent(
                agentId=options[CONF_AGENT_ID],
                agentAliasId=options.get(CONF_AGENT_ALIAS_ID) or DEFAULT_AGENT_ALIAS_ID,
                sessionId=chat_log.conversation_id,
                inputText=user_input.text,
            )
            # The completion is an event stream that is read while iterating
            return "".join(
                event["chunk"]["bytes"].decode()
                for event in response["completion"]
                if "chunk" in event
            )

        try:
            text = await self.hass.async_add_executor_job(invoke_agent)
        except (ClientError, BotoCoreError) as err:
            LOGGER.error("Error talking to Bedrock Agent: %s", error_message(err))
            raise HomeAssistantError("Error talking to Amazon Bedrock") from err

        chat_log.async_add_assistant_content_without_tools(
            conversation.AssistantContent(agent_id=user_input.agent_id, content=text)
        )

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(text)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=chat_log.conversation_id
        )

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the Converse API."""
        assert user_input.agent_id
        options = self.entry.options

        try:
            await chat_log.async_update_llm_data(
                DOMAIN,
                user_input,
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        system_prompt = chat_log.content[0]
        assert isinstance(system_prompt, conversation.SystemContent)

        model_args: dict[str, Any] = {
            "modelId": options.get(CONF_CHAT_MODEL, RECOMMENDED_CHAT_MODEL),
            "system": [{"text": system_prompt.content}],
            "inferenceConfig": {
                "maxTokens": int(options.get(CONF_MAX_TOKENS, RECOMMENDED_MAX_TOKENS)),
                "temperature": options.get(CONF_TEMPERATURE, RECOMMENDED_TEMPERATURE),
                "topP": options.get(CONF_TOP_P, RECOMMENDED_TOP_P),
            },
        }
        if chat_log.llm_api:
            model_args["toolConfig"] = {
                "tools": [
                    _format_tool(tool, chat_log.llm_api.custom_serializer)
                    for tool in chat_log.llm_api.tools
                ]
            }

        client = self.entry.runtime_data.runtime

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            messages = _convert_content(chat_log.content)
            try:
                response = await self.hass.async_add_executor_job(
                    partial(client.converse, messages=messages, **model_args)
                )
            except (ClientError, BotoCoreError) as err:
                LOGGER.error("Error talking to Amazon Bedrock: %s", error_message(err))
                raise HomeAssistantError("Error talking to Amazon Bedrock") from err

            LOGGER.debug("Response %s", response)

            text = ""
            tool_calls: list[llm.ToolInput] = []
            for block in response["output"]["message"]["content"]:
                if "text" in block:
                    text += block["text"]
                elif "toolUse" in block:
                    tool_calls.append(
                        llm.ToolInput(
                            id=block["toolUse"]["toolUseId"],
                            tool_name=block["toolUse"]["name"],
                            tool_args=block["toolUse"]["input"],
                        )
                    )

            # Run the tools, the results are added to the chat log
            async for _ in chat_log.async_add_assistant_content(
                conversation.AssistantContent(
                    agent_id=user_input.agent_id,
                    content=text,
                    tool_calls=tool_calls or None,
                )
            ):
                pass

            if not tool_calls:
                break

        intent_response = intent.IntentResponse(language=user_input.language)
        intent_response.async_set_speech(text)
        return conversation.ConversationResult(
            response=intent_response, conversation_id=chat_log.conversation_id
        )

    async def _async_entry_update_listener(
        self, hass: HomeAssistant, entry: ConfigEntry
    ) -> None:
        """Handle options update."""
        # Reload as we update device info + entity name + supported features
        await hass.config_entries.async_reload(entry.entry_id)
