"""Tests for the Amazon Bedrock AI Task entity."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import probatio
import pytest

from homeassistant.components import ai_task, media_source
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import HomeAssistantError
from homeassistant.helpers import selector

from . import message, set_converse_stream, text_events

ENTITY_ID = "ai_task.nova"


async def test_generate_data(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test generating text."""
    client = mock_clients["bedrock-runtime"]
    set_converse_stream(client, message(text_events("The test data")))

    result = await ai_task.async_generate_data(
        hass,
        task_name="Test Task",
        entity_id=ENTITY_ID,
        instructions="Generate test data",
    )

    assert result.data == "The test data"
    kwargs = client.converse_stream.call_args.kwargs
    assert kwargs["modelId"] == "amazon.nova-lite-v1:0"
    assert kwargs["messages"] == [
        {"role": "user", "content": [{"text": "Generate test data"}]}
    ]
    assert "outputConfig" not in kwargs


async def test_generate_structured_data(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test generating structured data."""
    client = mock_clients["bedrock-runtime"]
    set_converse_stream(
        client, message(text_events('{"characters": ["Mario", "Luigi"]}'))
    )

    result = await ai_task.async_generate_data(
        hass,
        task_name="Test Task",
        entity_id=ENTITY_ID,
        instructions="Generate test data",
        structure=probatio.Schema(
            {
                probatio.Required("characters"): selector.selector(
                    {"text": {"multiple": True}}
                )
            }
        ),
    )

    assert result.data == {"characters": ["Mario", "Luigi"]}
    output_config = client.converse_stream.call_args.kwargs["outputConfig"]
    json_schema = output_config["textFormat"]["structure"]["jsonSchema"]
    assert output_config["textFormat"]["type"] == "json_schema"
    assert json_schema["name"] == "test_task"
    assert json_schema["schema"] == (
        '{"type": "object", "properties": {"characters": {"type": "array", '
        '"items": {"type": "string"}}}, "required": ["characters"], '
        '"additionalProperties": false}'
    )


async def test_generate_invalid_structured_data(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test an invalid structured response."""
    set_converse_stream(
        mock_clients["bedrock-runtime"], message(text_events("Not JSON"))
    )

    with pytest.raises(HomeAssistantError, match="Failed to parse"):
        await ai_task.async_generate_data(
            hass,
            task_name="Test Task",
            entity_id=ENTITY_ID,
            instructions="Generate test data",
            structure=probatio.Schema({probatio.Required("characters"): str}),
        )


async def test_generate_data_no_response(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test the model not returning a response."""
    set_converse_stream(mock_clients["bedrock-runtime"], [])

    with pytest.raises(HomeAssistantError, match="did not return a response"):
        await ai_task.async_generate_data(
            hass,
            task_name="Test Task",
            entity_id=ENTITY_ID,
            instructions="Generate test data",
        )


async def test_generate_data_with_attachments(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
) -> None:
    """Test generating data with an image and a document attached."""
    client = mock_clients["bedrock-runtime"]
    set_converse_stream(client, message(text_events("A person at the door")))

    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            side_effect=[
                media_source.PlayMedia(
                    url="http://example.com/doorbell_snapshot.jpg",
                    mime_type="image/jpeg",
                    path=Path("doorbell_snapshot.jpg"),
                ),
                media_source.PlayMedia(
                    url="http://example.com/house rules (v2).pdf",
                    mime_type="application/pdf",
                    path=Path("house  rules_v2.pdf"),
                ),
            ],
        ),
        patch("pathlib.Path.exists", return_value=True),
        patch("pathlib.Path.read_bytes", side_effect=[b"jpeg", b"pdf"]),
    ):
        result = await ai_task.async_generate_data(
            hass,
            task_name="Test Task",
            entity_id=ENTITY_ID,
            instructions="Who is at the door?",
            attachments=[
                {"media_content_id": "media-source://media/doorbell_snapshot.jpg"},
                {"media_content_id": "media-source://media/house_rules.pdf"},
            ],
        )

    assert result.data == "A person at the door"
    assert client.converse_stream.call_args.kwargs["messages"] == [
        {
            "role": "user",
            "content": [
                {"text": "Who is at the door?"},
                {"image": {"format": "jpeg", "source": {"bytes": b"jpeg"}}},
                {
                    "document": {
                        "format": "pdf",
                        "name": "house rules-v2",
                        "source": {"bytes": b"pdf"},
                    }
                },
            ],
        }
    ]


@pytest.mark.parametrize(
    ("exists", "mime_type", "message_match"),
    [
        (False, "image/jpeg", "does not exist"),
        (True, "video/mp4", "is not a supported image or document"),
    ],
)
async def test_generate_data_invalid_attachments(
    hass: HomeAssistant,
    mock_clients: dict[str, Any],
    mock_init_component: None,
    exists: bool,
    mime_type: str,
    message_match: str,
) -> None:
    """Test attachments that can't be sent."""
    with (
        patch(
            "homeassistant.components.media_source.async_resolve_media",
            return_value=media_source.PlayMedia(
                url="http://example.com/file",
                mime_type=mime_type,
                path=Path("file"),
            ),
        ),
        patch("pathlib.Path.exists", return_value=exists),
        pytest.raises(HomeAssistantError, match=message_match),
    ):
        await ai_task.async_generate_data(
            hass,
            task_name="Test Task",
            entity_id=ENTITY_ID,
            instructions="Describe this",
            attachments=[{"media_content_id": "media-source://media/file"}],
        )
    mock_clients["bedrock-runtime"].converse_stream.assert_not_called()
