"""AI Task integration for Amazon Bedrock."""

from json import JSONDecodeError
from typing import override

from homeassistant.components import ai_task, conversation
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers.entity_platform import AddConfigEntryEntitiesCallback
from homeassistant.util.json import json_loads

from . import BedrockConfigEntry
from .const import DOMAIN, LOGGER, SUBENTRY_AI_TASK
from .entity import BedrockBaseLLMEntity

PARALLEL_UPDATES = 0


async def async_setup_entry(
    hass: HomeAssistant,
    config_entry: BedrockConfigEntry,
    async_add_entities: AddConfigEntryEntitiesCallback,
) -> None:
    """Set up AI Task entities."""
    for subentry in config_entry.subentries.values():
        if subentry.subentry_type != SUBENTRY_AI_TASK:
            continue

        async_add_entities(
            [BedrockTaskEntity(config_entry, subentry)],
            config_subentry_id=subentry.subentry_id,
        )


class BedrockTaskEntity(ai_task.AITaskEntity, BedrockBaseLLMEntity):
    """Amazon Bedrock AI Task entity."""

    _attr_supported_features = (
        ai_task.AITaskEntityFeature.GENERATE_DATA
        | ai_task.AITaskEntityFeature.SUPPORT_ATTACHMENTS
    )

    @override
    async def _async_generate_data(
        self,
        task: ai_task.GenDataTask,
        chat_log: conversation.ChatLog,
    ) -> ai_task.GenDataTaskResult:
        """Handle a generate data task."""
        await self._async_handle_chat_log(chat_log, task.name, task.structure)

        if not isinstance(chat_log.content[-1], conversation.AssistantContent):
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="response_not_found"
            )

        text = chat_log.content[-1].content or ""

        if not task.structure:
            return ai_task.GenDataTaskResult(
                conversation_id=chat_log.conversation_id,
                data=text,
            )
        try:
            data = json_loads(text)
        except JSONDecodeError as err:
            LOGGER.error("Failed to parse JSON response: %s. Response: %s", err, text)
            raise HomeAssistantError(
                translation_domain=DOMAIN, translation_key="json_parse_error"
            ) from err

        return ai_task.GenDataTaskResult(
            conversation_id=chat_log.conversation_id,
            data=data,
        )
