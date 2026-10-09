"""Tests for the Amazon Bedrock integration."""

from collections.abc import AsyncIterator, Iterable
import json
from typing import Any
from unittest.mock import AsyncMock

from homeassistant.core import HomeAssistant

from tests.common import MockConfigEntry


async def setup_integration(hass: HomeAssistant, config_entry: MockConfigEntry) -> None:
    """Set up the integration."""
    config_entry.add_to_hass(hass)
    await hass.config_entries.async_setup(config_entry.entry_id)
    await hass.async_block_till_done()


async def async_iter(items: Iterable[Any]) -> AsyncIterator[Any]:
    """Return an async iterator over the items."""
    for item in items:
        yield item


def text_events(text: str, index: int = 0) -> list[dict[str, Any]]:
    """Return the stream events for a text block."""
    return [
        {"contentBlockDelta": {"contentBlockIndex": index, "delta": {"text": text}}},
        {"contentBlockStop": {"contentBlockIndex": index}},
    ]


def tool_use_events(
    index: int, tool_use_id: str, name: str, tool_input: dict[str, Any]
) -> list[dict[str, Any]]:
    """Return the stream events for a tool use block, with the input split up."""
    encoded = json.dumps(tool_input)
    middle = len(encoded) // 2
    return [
        {
            "contentBlockStart": {
                "contentBlockIndex": index,
                "start": {"toolUse": {"toolUseId": tool_use_id, "name": name}},
            }
        },
        *(
            {
                "contentBlockDelta": {
                    "contentBlockIndex": index,
                    "delta": {"toolUse": {"input": part}},
                }
            }
            for part in (encoded[:middle], encoded[middle:])
        ),
        {"contentBlockStop": {"contentBlockIndex": index}},
    ]


def message(
    *blocks: list[dict[str, Any]], stop_reason: str = "end_turn"
) -> list[dict[str, Any]]:
    """Return the stream events for a message made of the given blocks."""
    return [
        {"messageStart": {"role": "assistant"}},
        *(event for block in blocks for event in block),
        {"messageStop": {"stopReason": stop_reason}},
        {"metadata": {"usage": {"inputTokens": 10, "outputTokens": 5}}},
    ]


def set_converse_stream(client: AsyncMock, *messages: list[dict[str, Any]]) -> None:
    """Make converse_stream return the messages, one per call."""
    client.converse_stream.side_effect = [
        {"stream": async_iter(events)} for events in messages
    ]
