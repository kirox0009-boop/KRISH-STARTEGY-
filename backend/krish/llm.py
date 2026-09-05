"""Optional LLM hook.

The factory is designed to keep producing with **no** LLM configured: every agent
that can use one has a deterministic fallback. When a key is present, the LLM is
used for the parts where language actually helps — phrasing hypotheses, writing
strategy dossiers, and explaining verdicts — never for arithmetic or for deciding
whether a strategy passed. Numbers stay with the backtester.
"""

from __future__ import annotations

import logging

import httpx

from .config import settings

log = logging.getLogger("krish.llm")

_TIMEOUT = httpx.Timeout(60.0, connect=10.0)


def available() -> bool:
    return settings().llm_enabled


async def complete(
    prompt: str,
    *,
    system: str | None = None,
    max_tokens: int = 800,
    temperature: float = 0.7,
) -> str | None:
    """Return model text, or ``None`` if no LLM is configured or the call fails."""
    cfg = settings()
    if not cfg.llm_enabled:
        return None
    try:
        if cfg.llm_provider == "anthropic":
            return await _anthropic(cfg, prompt, system, max_tokens, temperature)
        if cfg.llm_provider in {"openai", "openai_compatible"}:
            return await _openai(cfg, prompt, system, max_tokens, temperature)
        log.warning("unknown llm provider '%s'", cfg.llm_provider)
    except Exception:
        log.exception("llm call failed; falling back to deterministic path")
    return None


async def _anthropic(cfg, prompt: str, system: str | None, max_tokens: int, temperature: float):
    payload = {
        "model": cfg.llm_model or "claude-sonnet-4-20250514",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["system"] = system
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            json=payload,
            headers={
                "x-api-key": cfg.llm_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        resp.raise_for_status()
        blocks = resp.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks).strip() or None


async def _openai(cfg, prompt: str, system: str | None, max_tokens: int, temperature: float):
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            json={
                "model": cfg.llm_model or "gpt-4o-mini",
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            headers={"Authorization": f"Bearer {cfg.llm_api_key}"},
        )
        resp.raise_for_status()
        choices = resp.json().get("choices", [])
        return (choices[0]["message"]["content"].strip() if choices else None) or None
