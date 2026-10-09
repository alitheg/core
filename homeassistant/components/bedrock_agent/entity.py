"""Base entity for Amazon Bedrock."""

from collections.abc import AsyncGenerator, AsyncIterable, Callable
import json
from mimetypes import guess_file_type
from pathlib import Path
import re
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
import probatio

from homeassistant.components import conversation
from homeassistant.config_entries import ConfigSubentry
from homeassistant.const import CONF_MODEL
from homeassistant.core import HomeAssistant, callback
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import device_registry as dr, llm
from homeassistant.helpers.entity import Entity
from homeassistant.util import slugify

from . import BedrockConfigEntry
from .client import error_message, is_invalid_credentials
from .const import (
    CONF_MAX_TOKENS,
    CONF_TEMPERATURE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    LOGGER,
)

# Max number of back and forth with the LLM to generate a response
MAX_TOOL_ITERATIONS = 10

# Formats accepted by the Converse API, keyed by MIME type
IMAGE_FORMATS = {
    "image/gif": "gif",
    "image/jpeg": "jpeg",
    "image/png": "png",
    "image/webp": "webp",
}
DOCUMENT_FORMATS = {
    "application/msword": "doc",
    "application/pdf": "pdf",
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "text/csv": "csv",
    "text/html": "html",
    "text/markdown": "md",
    "text/plain": "txt",
}


def _format_schema(
    schema: probatio.Schema, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Convert a schema to a JSON schema."""
    return probatio.to_openapi(
        schema, custom_serializer=custom_serializer, openapi_version="3.1.0"
    )


def _format_tool(
    tool: llm.Tool, custom_serializer: Callable[[Any], Any] | None
) -> dict[str, Any]:
    """Format a tool specification for the Converse API."""
    tool_spec: dict[str, Any] = {
        "name": tool.name,
        "inputSchema": {"json": _format_schema(tool.parameters, custom_serializer)},
    }
    if tool.description:
        tool_spec["description"] = tool.description
    return {"toolSpec": tool_spec}


def _disallow_additional_properties(schema: Any) -> None:
    """Structured output requires objects to not allow additional properties."""
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            schema.setdefault("additionalProperties", False)
        for value in schema.values():
            _disallow_additional_properties(value)
    elif isinstance(schema, list):
        for value in schema:
            _disallow_additional_properties(value)


def _format_structured_output(
    name: str, schema: probatio.Schema, llm_api: llm.APIInstance | None
) -> dict[str, Any]:
    """Format the output config for a structured response."""
    json_schema = _format_schema(
        schema, llm_api.custom_serializer if llm_api else llm.selector_serializer
    )
    _disallow_additional_properties(json_schema)
    return {
        "textFormat": {
            "type": "json_schema",
            "structure": {
                "jsonSchema": {
                    "name": slugify(name)[:64] or "response",
                    "schema": json.dumps(json_schema),
                }
            },
        }
    }


def _tool_result_json(result: llm.ToolResult) -> dict[str, Any]:
    """Return the tool result as a JSON object, as the Converse API expects."""
    if isinstance(result.data, dict):
        return result.data
    return {"result": result.data}


def _convert_content(
    chat_content: list[conversation.Content],
    attachments: dict[int, list[dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    """Convert the chat log to Converse API messages.

    The Converse API requires messages to alternate between the user and the
    assistant, with tool results sent as user messages, so consecutive content
    with the same role is merged into a single message.

    Attachments are keyed by the id() of the user content they belong to.
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
                        "content": [{"json": _tool_result_json(content.result)}],
                        "status": "error" if content.result.error else "success",
                    }
                }
            )
        elif isinstance(content, conversation.UserContent):
            role = "user"
            if content.content:
                blocks.append({"text": content.content})
            if attachments:
                blocks.extend(attachments.get(id(content), ()))
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
                if not tool_call.external
            )

        if not blocks:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    return messages


def _document_name(path: Path) -> str:
    """Return a document name that is accepted by the Converse API.

    Names may only contain alphanumeric characters, whitespace, hyphens,
    parentheses and square brackets, without consecutive whitespace.
    """
    name = re.sub(r"[^A-Za-z0-9\s\-()\[\]]", "-", path.stem)
    return re.sub(r"\s+", " ", name).strip() or "document"


async def async_prepare_attachments(
    hass: HomeAssistant, attachments: list[conversation.Attachment]
) -> list[dict[str, Any]]:
    """Convert attachments to Converse API content blocks.

    Caller needs to ensure that the files are allowed.
    """

    def read_attachments() -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for attachment in attachments:
            path = attachment.path
            if not path.exists():
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="attachment_not_found",
                    translation_placeholders={"path": str(path)},
                )
            mime_type = attachment.mime_type or guess_file_type(path)[0]
            if mime_type in IMAGE_FORMATS:
                blocks.append(
                    {
                        "image": {
                            "format": IMAGE_FORMATS[mime_type],
                            "source": {"bytes": path.read_bytes()},
                        }
                    }
                )
            elif mime_type in DOCUMENT_FORMATS:
                blocks.append(
                    {
                        "document": {
                            "format": DOCUMENT_FORMATS[mime_type],
                            "name": _document_name(path),
                            "source": {"bytes": path.read_bytes()},
                        }
                    }
                )
            else:
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="unsupported_attachment",
                    translation_placeholders={"path": str(path)},
                )
        return blocks

    return await hass.async_add_executor_job(read_attachments)


async def _transform_stream(
    stream: AsyncIterable[dict[str, Any]],
) -> AsyncGenerator[conversation.AssistantContentDeltaDict]:
    """Transform the Converse API event stream to the chat log format."""
    # Tool use blocks stream their input as JSON fragments, keyed by block index
    tool_uses: dict[int, dict[str, str]] = {}

    async for event in stream:
        if "messageStart" in event:
            yield {"role": "assistant"}
        elif "contentBlockStart" in event:
            start = event["contentBlockStart"]
            if tool_use := start.get("start", {}).get("toolUse"):
                tool_uses[start["contentBlockIndex"]] = {
                    "id": tool_use["toolUseId"],
                    "name": tool_use["name"],
                    "input": "",
                }
        elif "contentBlockDelta" in event:
            block_delta = event["contentBlockDelta"]
            delta = block_delta["delta"]
            if "text" in delta:
                yield {"content": delta["text"]}
            elif "toolUse" in delta:
                tool_uses[block_delta["contentBlockIndex"]]["input"] += delta[
                    "toolUse"
                ].get("input", "")
            elif reasoning := delta.get("reasoningContent", {}).get("text"):
                yield {"thinking_content": reasoning}
        elif "contentBlockStop" in event:
            index = event["contentBlockStop"]["contentBlockIndex"]
            if tool_use_data := tool_uses.pop(index, None):
                try:
                    tool_args = json.loads(tool_use_data["input"] or "{}")
                except json.JSONDecodeError as err:
                    raise HomeAssistantError(
                        translation_domain=DOMAIN,
                        translation_key="invalid_tool_input",
                    ) from err
                yield {
                    "tool_calls": [
                        llm.ToolInput(
                            id=tool_use_data["id"],
                            tool_name=tool_use_data["name"],
                            tool_args=tool_args,
                        )
                    ]
                }
        elif "messageStop" in event:
            if event["messageStop"].get("stopReason") == "max_tokens":
                LOGGER.warning("The response was cut off at the maximum tokens")


class BedrockBaseEntity(Entity):
    """Base entity for Amazon Bedrock."""

    _attr_has_entity_name = True
    _attr_name = None

    def __init__(self, entry: BedrockConfigEntry, subentry: ConfigSubentry) -> None:
        """Initialize the entity."""
        self.entry = entry
        self.subentry = subentry
        self._attr_unique_id = subentry.subentry_id
        self._attr_device_info = dr.DeviceInfo(
            identifiers={(DOMAIN, subentry.subentry_id)},
            name=subentry.title,
            manufacturer="Amazon",
            model=subentry.data.get(CONF_MODEL, "Bedrock Agent"),
            entry_type=dr.DeviceEntryType.SERVICE,
        )

    @callback
    def _async_handle_error(self, err: ClientError | BotoCoreError) -> None:
        """Log an error and start reauth if the credentials are invalid."""
        LOGGER.error("Error talking to Amazon Bedrock: %s", error_message(err))
        if is_invalid_credentials(err):
            self.entry.async_start_reauth(self.hass)


class BedrockBaseLLMEntity(BedrockBaseEntity):
    """Base entity for models used through the Converse API."""

    async def _async_handle_chat_log(
        self,
        chat_log: conversation.ChatLog,
        structure_name: str | None = None,
        structure: probatio.Schema | None = None,
    ) -> None:
        """Generate an answer for the chat log."""
        options = self.subentry.data

        system_prompt = chat_log.content[0]
        assert isinstance(system_prompt, conversation.SystemContent)

        model_args: dict[str, Any] = {
            "modelId": options[CONF_MODEL],
            "inferenceConfig": {
                "maxTokens": int(options.get(CONF_MAX_TOKENS, DEFAULT_MAX_TOKENS)),
                "temperature": options.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
            },
        }
        if system_prompt.content:
            model_args["system"] = [{"text": system_prompt.content}]
        if chat_log.llm_api:
            model_args["toolConfig"] = {
                "tools": [
                    _format_tool(tool, chat_log.llm_api.custom_serializer)
                    for tool in chat_log.llm_api.tools
                ]
            }
        if structure:
            model_args["outputConfig"] = _format_structured_output(
                structure_name or "response", structure, chat_log.llm_api
            )

        # Attachments are only sent with the message they were added to
        attachments: dict[int, list[dict[str, Any]]] = {}
        last_content = chat_log.content[-1]
        if isinstance(last_content, conversation.UserContent) and (
            last_content.attachments
        ):
            attachments[id(last_content)] = await async_prepare_attachments(
                self.hass, last_content.attachments
            )

        client = self.entry.runtime_data.runtime

        # To prevent infinite loops, we limit the number of iterations
        for _iteration in range(MAX_TOOL_ITERATIONS):
            messages = _convert_content(chat_log.content, attachments)

            try:
                response = await client.converse_stream(messages=messages, **model_args)
                async for _ in chat_log.async_add_delta_content_stream(
                    self.entity_id, _transform_stream(response["stream"])
                ):
                    pass
            except (ClientError, BotoCoreError) as err:
                self._async_handle_error(err)
                raise HomeAssistantError(
                    translation_domain=DOMAIN,
                    translation_key="api_error",
                    translation_placeholders={"error": error_message(err)},
                ) from err

            if not chat_log.unresponded_tool_results:
                break
