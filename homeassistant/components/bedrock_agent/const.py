"""Constants for the Amazon Bedrock integration."""

import logging

DOMAIN = "bedrock_agent"
LOGGER = logging.getLogger(__package__)

CONF_ACCESS_KEY_ID = "access_key_id"
CONF_SECRET_ACCESS_KEY = "secret_access_key"

CONF_RECOMMENDED = "recommended"
CONF_PROMPT = "prompt"
CONF_CHAT_MODEL = "chat_model"
RECOMMENDED_CHAT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"
CONF_MAX_TOKENS = "max_tokens"
RECOMMENDED_MAX_TOKENS = 1024
CONF_TEMPERATURE = "temperature"
RECOMMENDED_TEMPERATURE = 1.0
CONF_TOP_P = "top_p"
RECOMMENDED_TOP_P = 1.0

CONF_AGENT_ID = "agent_id"
CONF_AGENT_ALIAS_ID = "agent_alias_id"
DEFAULT_AGENT_ALIAS_ID = "TSTALIASID"

REGIONS = [
    "us-east-1",
    "us-east-2",
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
