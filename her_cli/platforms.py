"""
Shared platform registry for Hermes Agent.

Single source of truth for platform metadata consumed by both
skills_config (label display) and tools_config (default toolset
resolution).  Import ``PLATFORMS`` from here instead of maintaining
duplicate dicts in each module.
"""

from collections import OrderedDict
from typing import NamedTuple


class PlatformInfo(NamedTuple):
    """Metadata for a single platform entry."""
    label: str
    default_toolset: str


# Ordered so that TUI menus are deterministic.
PLATFORMS: OrderedDict[str, PlatformInfo] = OrderedDict([
    ("cli",            PlatformInfo(label="🖥️  CLI",            default_toolset="her_agent-cli")),
    ("telegram",       PlatformInfo(label="📱 Telegram",        default_toolset="her_agent-telegram")),
    ("discord",        PlatformInfo(label="💬 Discord",         default_toolset="her_agent-discord")),
    ("slack",          PlatformInfo(label="💼 Slack",           default_toolset="her_agent-slack")),
    ("whatsapp",       PlatformInfo(label="📱 WhatsApp",        default_toolset="her_agent-whatsapp")),
    ("signal",         PlatformInfo(label="📡 Signal",          default_toolset="her_agent-signal")),
    ("bluebubbles",    PlatformInfo(label="💙 BlueBubbles",     default_toolset="her_agent-bluebubbles")),
    ("email",          PlatformInfo(label="📧 Email",           default_toolset="her_agent-email")),
    ("homeassistant",  PlatformInfo(label="🏠 Home Assistant",  default_toolset="her_agent-homeassistant")),
    ("mattermost",     PlatformInfo(label="💬 Mattermost",      default_toolset="her_agent-mattermost")),
    ("matrix",         PlatformInfo(label="💬 Matrix",          default_toolset="her_agent-matrix")),
    ("dingtalk",       PlatformInfo(label="💬 DingTalk",        default_toolset="her_agent-dingtalk")),
    ("feishu",         PlatformInfo(label="🪽 Feishu",          default_toolset="her_agent-feishu")),
    ("wecom",          PlatformInfo(label="💬 WeCom",           default_toolset="her_agent-wecom")),
    ("wecom_callback", PlatformInfo(label="💬 WeCom Callback",  default_toolset="her_agent-wecom-callback")),
    ("weixin",         PlatformInfo(label="💬 Weixin",          default_toolset="her_agent-weixin")),
    ("qqbot",          PlatformInfo(label="💬 QQBot",           default_toolset="her_agent-qqbot")),
    ("webhook",        PlatformInfo(label="🔗 Webhook",         default_toolset="her_agent-webhook")),
    ("api_server",     PlatformInfo(label="🌐 API Server",      default_toolset="her_agent-api-server")),
])


def platform_label(key: str, default: str = "") -> str:
    """Return the display label for a platform key, or *default*."""
    info = PLATFORMS.get(key)
    return info.label if info is not None else default
