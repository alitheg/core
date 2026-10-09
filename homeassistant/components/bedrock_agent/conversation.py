"""Conversation support for Amazon Bedrock."""

from collections.abc import AsyncGenerator, AsyncIterable
from typing import Any, Literal, override

from botocore.exceptions import BotoCoreError, ClientError

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT, MATCH_ALL
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback

from . import BedrockConfigEntry
from .client import error_message
from .const import (
    CONF_AGENT_ALIAS_ID,
    CONF_AGENT_ID,
    DEFAULT_AGENT_ALIAS_ID,
    DOMAIN,
    SUBENTRY_AGENT,
    SUBENTRY_CONVERSATION,
)
from .entity import BedrockBaseEntity, BedrockBaseLLMEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: BedrockConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up conversation entities."""
    for subentry in config_entry.subentries.values():
        entity: conversation.ConversationEntity
        if subentry.subentry_type == SUBENTRY_CONVERSATION:
            entity = BedrockConversationEntity(config_entry, subentry)
        elif subentry.subentry_type == SUBENTRY_AGENT:
            entity = BedrockAgentConversationEntity(config_entry, subentry)
        else:
            continue
        async_add_entities([entity], config_subentry_id=subentry.subentry_id)


class BedrockConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    BedrockBaseLLMEntity,
):
    """Amazon Bedrock conversation agent using the Converse API."""

    _attr_supports_streaming = True

    def __init__(self, entry: BedrockConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the agent."""
        super().__init__(entry, subentry)
        if self.subentry.data.get(CONF_LLM_HASS_API):
            self._attr_supported_features = (
                conversation.ConversationEntityFeature.CONTROL
            )

    @property
    @override
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @override
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Call the Converse API."""
        options = self.subentry.data

        try:
            await chat_log.async_provide_llm_data(
                user_input.as_llm_context(DOMAIN),
                options.get(CONF_LLM_HASS_API),
                options.get(CONF_PROMPT),
                user_input.extra_system_prompt,
            )
        except conversation.ConverseError as err:
            return err.as_conversation_result()

        await self._async_handle_chat_log(chat_log)

        return conversation.async_get_result_from_chat_log(user_input, chat_log)


async def _transform_agent_stream(
    completion: AsyncIterable[dict[str, Any]],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the Bedrock Agent completion stream to the chat log format."""
    yield {"role": "assistant"}
    async for event in completion:
        if chunk := event.get("chunk"):
            yield {"content": chunk["bytes"].decode()}


class BedrockAgentConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
    BedrockBaseEntity,
):
    """Conversation agent backed by a Bedrock Agent.

    Bedrock Agents keep the conversation history server side, keyed by the
    session ID, and bring their own instructions, tools and knowledge bases.
    """

    _attr_supports_streaming = True

    @property
    @override
    def supported_languages(self) -> list[str] | Literal["*"]:
        """Return a list of supported languages."""
        return MATCH_ALL

    @override
    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Send the message to the Bedrock Agent."""
        options = self.subentry.data
        client = self.entry.runtime_data.agent_runtime

        try:
            response = await client.invoke_agent(
                agentId=options[CONF_AGENT_ID],
                agentAliasId=options.get(CONF_AGENT_ALIAS_ID) or DEFAULT_AGENT_ALIAS_ID,
                sessionId=chat_log.conversation_id,
                inputText=user_input.text,
                streamingConfigurations={"streamFinalResponse": True},
            )
            async for _ in chat_log.async_add_delta_content_stream(
                self.entity_id, _transform_agent_stream(response["completion"])
            ):
                pass
        except (ClientError, BotoCoreError) as err:
            self._async_handle_error(err)
            raise HomeAssistantError(
                translation_domain=DOMAIN,
                translation_key="api_error",
                translation_placeholders={"error": error_message(err)},
            ) from err

        return conversation.async_get_result_from_chat_log(user_input, chat_log)
