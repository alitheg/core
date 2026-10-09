"""Constants for the Amazon Bedrock integration."""

import logging

from homeassistant.const import CONF_LLM_HASS_API, CONF_PROMPT
from homeassistant.helpers import llm

DOMAIN = "bedrock_agent"
LOGGER = logging.getLogger(__package__)

CONF_ACCESS_KEY_ID = "access_key_id"
CONF_SECRET_ACCESS_KEY = "secret_access_key"
CONF_MAX_TOKENS = "max_tokens"
CONF_TEMPERATURE = "temperature"
CONF_AGENT_ID = "agent_id"
CONF_AGENT_ALIAS_ID = "agent_alias_id"

SUBENTRY_CONVERSATION = "conversation"
SUBENTRY_AI_TASK = "ai_task_data"
SUBENTRY_AGENT = "agent"

DEFAULT_CONVERSATION_NAME = "Amazon Bedrock conversation"
DEFAULT_AI_TASK_NAME = "Amazon Bedrock AI Task"
DEFAULT_AGENT_ALIAS_ID = "TSTALIASID"
DEFAULT_MAX_TOKENS = 3000
DEFAULT_TEMPERATURE = 1.0

DEFAULT_CONVERSATION_OPTIONS = {
    CONF_LLM_HASS_API: [llm.LLM_API_ASSIST],
    CONF_PROMPT: llm.DEFAULT_INSTRUCTIONS_PROMPT,
}

# Preferred default models, the first one that is available in the region is used.
# Matched against the model or inference profile ID.
PREFERRED_MODELS = (
    "anthropic.claude-haiku-4-5",
    "anthropic.claude-sonnet-4-5",
    "amazon.nova-2-lite",
    "amazon.nova-pro",
    "amazon.nova-lite",
)

REGIONS = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "sa-east-1",
    "eu-central-1",
    "eu-central-2",
    "eu-north-1",
    "eu-south-1",
    "eu-south-2",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "ap-northeast-1",
    "ap-northeast-2",
    "ap-northeast-3",
    "ap-south-1",
    "ap-south-2",
    "ap-southeast-1",
    "ap-southeast-2",
]
