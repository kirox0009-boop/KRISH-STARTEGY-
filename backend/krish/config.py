"""Configuration: environment + YAML config files.

Two sources on purpose:
  * secrets / infra  -> environment (.env)
  * factory behaviour -> config/*.yaml, so it is reviewable, diffable and
    editable live from the control room without a redeploy.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# repo_root/backend/krish/config.py -> repo_root
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.getenv("KRISH_CONFIG_DIR", REPO_ROOT / "config"))
DATA_DIR = Path(os.getenv("KRISH_DATA_DIR", REPO_ROOT / "var"))

CACHE_DIR = DATA_DIR / "cache"
ARTIFACT_DIR = DATA_DIR / "artifacts"
PACKAGE_DIR = DATA_DIR / "packages"
LOG_DIR = DATA_DIR / "logs"

for _d in (DATA_DIR, CACHE_DIR, ARTIFACT_DIR, PACKAGE_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    """Infra + secrets. Never contains strategy logic knobs."""

    env: str = field(default_factory=lambda: _env("KRISH_ENV", "dev"))
    # bus: "memory" needs nothing installed, "redis" is what the VPS uses
    bus_backend: str = field(default_factory=lambda: _env("KRISH_BUS", "memory"))
    redis_url: str = field(default_factory=lambda: _env("REDIS_URL", "redis://localhost:6379/0"))
    database_url: str = field(
        default_factory=lambda: _env("DATABASE_URL", f"sqlite:///{DATA_DIR / 'krish.db'}")
    )
    api_host: str = field(default_factory=lambda: _env("KRISH_API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(_env("KRISH_API_PORT", "8000")))
    log_level: str = field(default_factory=lambda: _env("KRISH_LOG_LEVEL", "INFO"))

    # LLM (optional: agents fall back to deterministic generation without it)
    llm_provider: str = field(default_factory=lambda: _env("KRISH_LLM_PROVIDER", "none"))
    llm_model: str = field(default_factory=lambda: _env("KRISH_LLM_MODEL", ""))
    llm_api_key: str = field(default_factory=lambda: _env("KRISH_LLM_API_KEY", ""))

    # delivery
    telegram_bot_token: str = field(default_factory=lambda: _env("TELEGRAM_BOT_TOKEN", ""))
    telegram_chat_id: str = field(default_factory=lambda: _env("TELEGRAM_CHAT_ID", ""))
    gdrive_credentials_file: str = field(
        default_factory=lambda: _env("GDRIVE_CREDENTIALS_FILE", "")
    )
    gdrive_folder_id: str = field(default_factory=lambda: _env("GDRIVE_FOLDER_ID", ""))

    # safety rails
    allow_live_trading: bool = field(default_factory=lambda: _env_bool("KRISH_ALLOW_LIVE", False))

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none" and bool(self.llm_api_key)


@lru_cache(maxsize=1)
def settings() -> Settings:
    return Settings()


# --------------------------------------------------------------------------- #
# YAML configs (hot-reloadable)
# --------------------------------------------------------------------------- #

_yaml_lock = threading.Lock()
_yaml_cache: dict[str, dict[str, Any]] = {}


def load_yaml(name: str, *, refresh: bool = False) -> dict[str, Any]:
    """Load ``config/<name>.yaml``. Cached unless ``refresh`` is set."""
    with _yaml_lock:
        if refresh or name not in _yaml_cache:
            path = CONFIG_DIR / f"{name}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"config file not found: {path}")
            with path.open("r", encoding="utf-8") as fh:
                _yaml_cache[name] = yaml.safe_load(fh) or {}
        return _yaml_cache[name]


#: Written to the top of any config file the UI rewrites. PyYAML cannot preserve
#: comments on a round trip, so instead of silently deleting the guidance that was
#: in the file, we say plainly what happened and where the original lives.
_REWRITE_HEADER = """\
# This file was last written by KRISH (control room / API).
# Comments are not preserved when the UI saves it - see the original, fully
# commented version in git history or in the repository's config/ directory.
"""


def save_yaml(name: str, data: dict[str, Any]) -> None:
    """Persist a config file and invalidate the cache (used by the UI)."""
    path = CONFIG_DIR / f"{name}.yaml"
    with _yaml_lock:
        with path.open("w", encoding="utf-8") as fh:
            fh.write(_REWRITE_HEADER)
            yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
        _yaml_cache[name] = data


def factory_config(*, refresh: bool = False) -> dict[str, Any]:
    return load_yaml("factory", refresh=refresh)


def factory_section(section: str) -> dict[str, Any]:
    return dict(factory_config().get(section) or {})
