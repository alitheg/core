"""Config flow for Amazon Bedrock integration."""

from collections.abc import Mapping
import logging
from typing import Any, override

from botocore.exceptions import BotoCoreError, ClientError
import probatio

from homeassistant.config_entries import (
    SOURCE_USER,
    ConfigEntry,
    ConfigEntryState,
    ConfigFlow,
    ConfigFlowResult,
    ConfigSubentryFlow,
    SubentryFlowResult,
)
from homeassistant.const import (
    CONF_LLM_HASS_API,
    CONF_MODEL,
    CONF_NAME,
    CONF_PROMPT,
    CONF_REGION,
)
from homeassistant.core import callback
from homeassistant.helpers import llm
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectOptionDict,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TemplateSelector,
)

from .client import (
    async_list_agents,
    async_list_models,
    async_validate_credentials,
    is_auth_error,
)
from .const import (
    CONF_ACCESS_KEY_ID,
    CONF_AGENT_ALIAS_ID,
    CONF_AGENT_ID,
    CONF_MAX_TOKENS,
    CONF_SECRET_ACCESS_KEY,
    CONF_TEMPERATURE,
    DEFAULT_AGENT_ALIAS_ID,
    DEFAULT_AI_TASK_NAME,
    DEFAULT_CONVERSATION_NAME,
    DEFAULT_CONVERSATION_OPTIONS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DOMAIN,
    PREFERRED_MODELS,
    REGIONS,
    SUBENTRY_AGENT,
    SUBENTRY_AI_TASK,
    SUBENTRY_CONVERSATION,
)

_LOGGER = logging.getLogger(__name__)

STEP_USER_DATA_SCHEMA = probatio.Schema(
    {
        probatio.Required(CONF_REGION, default="us-east-1"): SelectSelector(
            SelectSelectorConfig(
                options=REGIONS, custom_value=True, mode=SelectSelectorMode.DROPDOWN
            )
        ),
        probatio.Required(CONF_ACCESS_KEY_ID): str,
        probatio.Required(probatio.Secret(CONF_SECRET_ACCESS_KEY)): str,
    }
)

STEP_REAUTH_DATA_SCHEMA = probatio.Schema(
    {
        probatio.Required(CONF_ACCESS_KEY_ID): str,
        probatio.Required(probatio.Secret(CONF_SECRET_ACCESS_KEY)): str,
    }
)


def _model_selector(models: list[SelectOptionDict]) -> SelectSelector:
    """Return a selector for a model, allowing any model ID to be entered."""
    return SelectSelector(
        SelectSelectorConfig(
            options=models, custom_value=True, mode=SelectSelectorMode.DROPDOWN
        )
    )


def _default_model(models: list[SelectOptionDict]) -> str | None:
    """Return the preferred model that is available."""
    for preferred in PREFERRED_MODELS:
        for model in models:
            if preferred in model["value"]:
                return model["value"]
    return models[0]["value"] if models else None


class BedrockConfigFlow(ConfigFlow, domain=DOMAIN):
    """Handle a config flow for Amazon Bedrock."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialize the config flow."""
        self.data: dict[str, Any] = {}
        self.models: list[SelectOptionDict] = []

    @classmethod
    @callback
    @override
    def async_get_supported_subentry_types(
        cls, config_entry: ConfigEntry
    ) -> dict[str, type[ConfigSubentryFlow]]:
        """Return subentries supported by this handler."""
        return {
            SUBENTRY_CONVERSATION: LLMSubentryFlowHandler,
            SUBENTRY_AI_TASK: LLMSubentryFlowHandler,
            SUBENTRY_AGENT: AgentSubentryFlowHandler,
        }

    async def _async_validate(self, data: Mapping[str, Any]) -> dict[str, str]:
        """Validate the credentials and return any errors."""
        try:
            await async_validate_credentials(self.hass, data)
        except (ClientError, BotoCoreError) as err:
            if is_auth_error(err):
                return {"base": "invalid_auth"}
            _LOGGER.debug("Error connecting to Amazon Bedrock: %s", err)
            return {"base": "cannot_connect"}
        except Exception:
            _LOGGER.exception("Unexpected exception")
            return {"base": "unknown"}
        return {}

    @override
    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for the region and credentials."""
        errors: dict[str, str] = {}

        if user_input is not None:
            self._async_abort_entries_match(
                {
                    CONF_REGION: user_input[CONF_REGION],
                    CONF_ACCESS_KEY_ID: user_input[CONF_ACCESS_KEY_ID],
                }
            )
            if not (errors := await self._async_validate(user_input)):
                self.data = user_input
                return await self.async_step_model()

        return self.async_show_form(
            step_id="user",
            data_schema=self.add_suggested_values_to_schema(
                STEP_USER_DATA_SCHEMA, user_input
            ),
            errors=errors,
        )

    async def async_step_model(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask which model to use for the default agents."""
        if user_input is not None:
            model = user_input[CONF_MODEL]
            return self.async_create_entry(
                title=f"Amazon Bedrock ({self.data[CONF_REGION]})",
                data=self.data,
                subentries=[
                    {
                        "subentry_type": SUBENTRY_CONVERSATION,
                        "data": {**DEFAULT_CONVERSATION_OPTIONS, CONF_MODEL: model},
                        "title": DEFAULT_CONVERSATION_NAME,
                        "unique_id": None,
                    },
                    {
                        "subentry_type": SUBENTRY_AI_TASK,
                        "data": {CONF_MODEL: model},
                        "title": DEFAULT_AI_TASK_NAME,
                        "unique_id": None,
                    },
                ],
            )

        errors: dict[str, str] = {}
        try:
            self.models = await async_list_models(self.hass, self.data)
        except (ClientError, BotoCoreError) as err:
            # A model ID can still be entered by hand
            _LOGGER.warning("Unable to list Amazon Bedrock models: %s", err)
            self.models = []
            errors["base"] = "list_models_failed"

        return self.async_show_form(
            step_id="model",
            data_schema=probatio.Schema(
                {
                    probatio.Required(
                        CONF_MODEL, default=_default_model(self.models)
                    ): _model_selector(self.models),
                }
            ),
            errors=errors,
        )

    async def async_step_reauth(
        self, entry_data: Mapping[str, Any]
    ) -> ConfigFlowResult:
        """Handle the credentials becoming invalid."""
        return await self.async_step_reauth_confirm()

    async def async_step_reauth_confirm(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Ask for new credentials."""
        errors: dict[str, str] = {}
        reauth_entry = self._get_reauth_entry()

        if user_input is not None:
            data = {**reauth_entry.data, **user_input}
            if not (errors := await self._async_validate(data)):
                return self.async_update_reload_and_abort(reauth_entry, data=data)

        return self.async_show_form(
            step_id="reauth_confirm",
            data_schema=self.add_suggested_values_to_schema(
                STEP_REAUTH_DATA_SCHEMA,
                user_input
                or {CONF_ACCESS_KEY_ID: reauth_entry.data[CONF_ACCESS_KEY_ID]},
            ),
            description_placeholders={"region": reauth_entry.data[CONF_REGION]},
            errors=errors,
        )


class BedrockSubentryFlowHandler(ConfigSubentryFlow):
    """Base class for Amazon Bedrock subentry flows."""

    options: dict[str, Any]

    @property
    def _is_new(self) -> bool:
        """Return if this is a new subentry."""
        return self.source == SOURCE_USER

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Add a subentry."""
        self.options = {}
        return await self.async_step_init()

    async def async_step_reconfigure(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Reconfigure a subentry."""
        self.options = dict(self._get_reconfigure_subentry().data)
        return await self.async_step_init()

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure the subentry."""
        raise NotImplementedError

    def _async_save(self, user_input: dict[str, Any]) -> SubentryFlowResult:
        """Create or update the subentry."""
        if self._is_new:
            return self.async_create_entry(
                title=user_input.pop(CONF_NAME), data=user_input
            )
        return self.async_update_and_abort(
            self._get_entry(), self._get_reconfigure_subentry(), data=user_input
        )

    def _name_schema(self, default_name: str) -> dict[Any, Any]:
        """Return the schema for the name, only shown for new subentries."""
        if not self._is_new:
            return {}
        return {probatio.Required(CONF_NAME, default=default_name): str}


class LLMSubentryFlowHandler(BedrockSubentryFlowHandler):
    """Flow for conversation agents and AI tasks using a model."""

    @override
    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure the model and its settings."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            if not user_input.get(CONF_LLM_HASS_API):
                user_input.pop(CONF_LLM_HASS_API, None)
            return self._async_save(user_input)

        try:
            models = await async_list_models(self.hass, entry.data)
        except (ClientError, BotoCoreError) as err:
            _LOGGER.warning("Unable to list Amazon Bedrock models: %s", err)
            models = []

        is_conversation = self._subentry_type == SUBENTRY_CONVERSATION
        if self._is_new and is_conversation:
            self.options = dict(DEFAULT_CONVERSATION_OPTIONS)

        schema: dict[Any, Any] = self._name_schema(
            DEFAULT_CONVERSATION_NAME if is_conversation else DEFAULT_AI_TASK_NAME
        )
        schema[
            probatio.Required(
                CONF_MODEL, default=self.options.get(CONF_MODEL, _default_model(models))
            )
        ] = _model_selector(models)

        if is_conversation:
            hass_apis = [
                SelectOptionDict(label=api.name, value=api.id)
                for api in llm.async_get_apis(self.hass)
            ]
            valid_api_ids = {api["value"] for api in hass_apis}
            schema.update(
                {
                    probatio.Optional(CONF_PROMPT): TemplateSelector(),
                    probatio.Optional(
                        CONF_LLM_HASS_API,
                        default=[
                            api
                            for api in self.options.get(CONF_LLM_HASS_API, [])
                            if api in valid_api_ids
                        ],
                    ): SelectSelector(
                        SelectSelectorConfig(options=hass_apis, multiple=True)
                    ),
                    probatio.Optional(
                        CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                    ): NumberSelector(NumberSelectorConfig(min=0, max=1, step=0.05)),
                }
            )

        schema[probatio.Optional(CONF_MAX_TOKENS, default=DEFAULT_MAX_TOKENS)] = (
            NumberSelector(
                NumberSelectorConfig(min=1, step=1, mode=NumberSelectorMode.BOX)
            )
        )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                probatio.Schema(schema), self.options
            ),
        )


class AgentSubentryFlowHandler(BedrockSubentryFlowHandler):
    """Flow for conversation agents backed by a Bedrock Agent."""

    @override
    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> SubentryFlowResult:
        """Configure the Bedrock Agent."""
        entry = self._get_entry()
        if entry.state is not ConfigEntryState.LOADED:
            return self.async_abort(reason="entry_not_loaded")

        if user_input is not None:
            return self._async_save(user_input)

        agents = await async_list_agents(self.hass, entry.data)
        schema: dict[Any, Any] = self._name_schema("Bedrock Agent")
        schema.update(
            {
                probatio.Required(CONF_AGENT_ID): SelectSelector(
                    SelectSelectorConfig(
                        options=agents,
                        custom_value=True,
                        mode=SelectSelectorMode.DROPDOWN,
                    )
                ),
                probatio.Required(
                    CONF_AGENT_ALIAS_ID, default=DEFAULT_AGENT_ALIAS_ID
                ): str,
            }
        )

        return self.async_show_form(
            step_id="init",
            data_schema=self.add_suggested_values_to_schema(
                probatio.Schema(schema), self.options
            ),
        )
