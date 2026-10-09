"""Config flow for Amazon Bedrock integration."""

from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any

from botocore.exceptions import BotoCoreError, ClientError
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.const import CONF_LLM_HASS_API, CONF_REGION
from homeassistant.core import HomeAssistant
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)
from homeassistant.helpers.typing import VolDictType

from . import create_client, is_auth_error
from .const import (
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
    DEFAULT_AGENT_ALIAS_ID,
    DOMAIN,
    RECOMMENDED_CHAT_MODEL,
    RECOMMENDED_MAX_TOKENS,
    RECOMMENDED_TEMPERATURE,
    RECOMMENDED_TOP_P,
    REGIONS,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_REGION, default="us-east-1"): SelectSelector(
            SelectSelectorConfig(
                options=REGIONS, custom_value=True, mode=SelectSelectorMode.DROPDOWN
            )
        ),
        vol.Required(CONF_ACCESS_KEY_ID): str,
        vol.Required(CONF_SECRET_ACCESS_KEY): TextSelector(
            TextSelectorConfig(type=TextSelectorType.PASSWORD)
        ),
    }
)

RECOMMENDED_OPTIONS = {
    CONF_RECOMMENDED: True,
    CONF_LLM_HASS_API: llm.LLM_API_ASSIST,
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}


def validate_input(data: dict[str, Any]) -> None:
    """Validate the user input allows us to connect.

    Data has the keys from STEP_USER_DATA_SCHEMA with values provided by the user.
    This does blocking I/O and must be run in the executor.
    """
    create_client(data, "bedrock").list_foundation_models(byOutputModality="TEXT")


class BedrockConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Amazon Bedrock."""

    VERSION = 1

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Handle the initial step."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._async_abort_entries_match(
                {
                    CONF_REGION: user_input[CONF_REGION],
                    CONF_ACCESS_KEY_ID: user_input[CONF_ACCESS_KEY_ID],
                }
            )
            try:
                await self.hass.async_add_executor_job(validate_input, user_input)
            except ClientError as err:
                if is_auth_error(err):
                    errors["base"] = "invalid_auth"
                else:
                    _LOGGER.debug("Error connecting to Amazon Bedrock: %s", err)
                    errors["base"] = "cannot_connect"
            except BotoCoreError as err:
                _LOGGER.debug("Error connecting to Amazon Bedrock: %s", err)
                errors["base"] = "cannot_connect"
            except Exception:
                _LOGGER.exception("Unexpected exception")
                errors["base"] = "unknown"
            else:
                return self.async_create_entry(
                    title=f"Amazon Bedrock ({user_input[CONF_REGION]})",
                    data=user_input,
                    options=RECOMMENDED_OPTIONS,
                )

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA, user_input
            ),
            errors=errors,
        )

    @staticmethod
    def async_get_options_flow(
        config_entry: ConfigEntry,
    ) -> OptionsFlow:
        """Create the options flow."""
        return BedrockOptionsFlow(config_entry)


class BedrockOptionsFlow(OptionsFlow):
    """Amazon Bedrock config flow options handler."""

    def __init__(self, config_entry: ConfigEntry) -> None:
        """Initialize options flow."""
        self.last_rendered_recommended = config_entry.options.get(
            CONF_RECOMMENDED, False
        )
        self.models: list[SelectOptionDict] | None = None
        self.agents: list[SelectOptionDict] | None = None

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Manage the options."""
        options: dict[str, Any] | MappingProxyType[str, Any] = self.config_entry.options

        if user_input is not None:
            if user_input[CONF_RECOMMENDED] == self.last_rendered_recommended:
                if user_input[CONF_LLM_HASS_API] == "none":
                    user_input.pop(CONF_LLM_HASS_API)
                if not user_input.get(CONF_AGENT_ID):
                    user_input.pop(CONF_AGENT_ID, None)
                    user_input.pop(CONF_AGENT_ALIAS_ID, None)
                return self.async_create_entry(title="", data=user_input)

            # Re-render the options again, now with the recommended options shown/hidden
            self.last_rendered_recommended = user_input[CONF_RECOMMENDED]

            options = {
                CONF_RECOMMENDED: user_input[CONF_RECOMMENDED],
                CONF_PROMPT: user_input[CONF_PROMPT],
                CONF_LLM_HASS_API: user_input[CONF_LLM_HASS_API],
            }

        if not options.get(CONF_RECOMMENDED) and self.models is None:
            self.models, self.agents = await self.hass.async_add_executor_job(
                _fetch_models_and_agents, dict(self.config_entry.data)
            )

        schema = bedrock_config_option_schema(
            self.hass, options, self.models or [], self.agents or []
        )
        return self.async_show_form(
            step_id="init",
            data_schema=vol.Schema(schema),
        )


def _fetch_models_and_agents(
    data: dict[str, Any],
) -> tuple[list[SelectOptionDict], list[SelectOptionDict]]:
    """Fetch the available models and agents.

    Failures are not fatal, the user can still enter a model ID by hand.
    This does blocking I/O and must be run in the executor.
    """
    models: list[SelectOptionDict] = []
    agents: list[SelectOptionDict] = []

    try:
        bedrock = create_client(data, "bedrock")
        response = bedrock.list_foundation_models(
            byOutputModality="TEXT", byInferenceType="ON_DEMAND"
        )
        models.extend(
            SelectOptionDict(
                value=model["modelId"],
                label=f"{model['providerName']} {model['modelName']}",
            )
            for model in response["modelSummaries"]
            if model.get("modelLifecycle", {}).get("status") != "LEGACY"
        )
        # Newer models are only available through cross-region inference profiles
        if hasattr(bedrock, "list_inference_profiles"):
            response = bedrock.list_inference_profiles()
            models.extend(
                SelectOptionDict(
                    value=profile["inferenceProfileId"],
                    label=profile["inferenceProfileName"],
                )
                for profile in response["inferenceProfileSummaries"]
            )
    except (ClientError, BotoCoreError) as err:
        _LOGGER.warning("Unable to list Amazon Bedrock models: %s", err)

    try:
        bedrock_agent = create_client(data, "bedrock-agent")
        response = bedrock_agent.list_agents()
        agents.extend(
            SelectOptionDict(value=agent["agentId"], label=agent["agentName"])
            for agent in response["agentSummaries"]
        )
    except (ClientError, BotoCoreError) as err:
        _LOGGER.debug("Unable to list Amazon Bedrock agents: %s", err)

    return sorted(models, key=lambda model: model["label"]), agents


def bedrock_config_option_schema(
    hass: HomeAssistant,
    options: dict[str, Any] | MappingProxyType[str, Any],
    models: list[SelectOptionDict],
    agents: list[SelectOptionDict],
) -> VolDictType:
    """Return a schema for Amazon Bedrock completion options."""
    hass_apis: list[SelectOptionDict] = [
        SelectOptionDict(
            label="No control",
            value="none",
        )
    ]
    hass_apis.extend(
        SelectOptionDict(
            label=api.name,
            value=api.id,
        )
        for api in llm.async_get_apis(hass)
    )

    schema: VolDictType = {
        vol.Optional(
            CONF_PROMPT,
            description={
                "suggested_value": options.get(
                    CONF_PROMPT, llm.DEFAULT_INSTRUCTIONS_PROMPT
                )
            },
        ): TemplateSelector(),
        vol.Optional(
            CONF_LLM_HASS_API,
            description={"suggested_value": options.get(CONF_LLM_HASS_API)},
            default="none",
        ): SelectSelector(SelectSelectorConfig(options=hass_apis)),
        vol.Required(
            CONF_RECOMMENDED, default=options.get(CONF_RECOMMENDED, False)
        ): bool,
    }

    if options.get(CONF_RECOMMENDED):
        return schema

    schema.update(
        {
            vol.Optional(
                CONF_CHAT_MODEL,
                description={"suggested_value": options.get(CONF_CHAT_MODEL)},
                default=RECOMMENDED_CHAT_MODEL,
            ): SelectSelector(
                SelectSelectorConfig(
                    options=models,
                    custom_value=True,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_MAX_TOKENS,
                description={"suggested_value": options.get(CONF_MAX_TOKENS)},
                default=RECOMMENDED_MAX_TOKENS,
            ): int,
            vol.Optional(
                CONF_TOP_P,
                description={"suggested_value": options.get(CONF_TOP_P)},
                default=RECOMMENDED_TOP_P,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_TEMPERATURE,
                description={"suggested_value": options.get(CONF_TEMPERATURE)},
                default=RECOMMENDED_TEMPERATURE,
            ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
            vol.Optional(
                CONF_AGENT_ID,
                description={"suggested_value": options.get(CONF_AGENT_ID)},
            ): SelectSelector(
                SelectSelectorConfig(
                    options=agents,
                    custom_value=True,
                    mode=SelectSelectorMode.DROPDOWN,
                )
            ),
            vol.Optional(
                CONF_AGENT_ALIAS_ID,
                description={
                    "suggested_value": options.get(
                        CONF_AGENT_ALIAS_ID, DEFAULT_AGENT_ALIAS_ID
                    )
                },
            ): str,
        }
    )
    return schema
