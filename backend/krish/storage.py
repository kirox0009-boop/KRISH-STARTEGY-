"""Object storage: keep the VPS disk small by pushing bulk to the cloud.

A VPS disk is the scarcest resource on a box that also has to do other work, and
the factory produces two kinds of bulk: the Parquet price cache and the delivered
ZIPs. Neither needs to live locally.

    price cache  -> uploaded once, re-downloaded on demand, safe to prune locally
    packages     -> uploaded, then optionally removed locally; the download
                    endpoint redirects to the object store

Backends:
    ``local``  (default) everything stays on disk. Zero configuration.
    ``s3``     any S3-compatible store — Cloudflare R2, Backblaze B2, Wasabi,
               MinIO, AWS S3. Only the endpoint URL differs.

The database is handled separately and needs no code: point ``DATABASE_URL`` at a
managed Postgres (Neon, Supabase, Railway) and the blackboard leaves the box too.

Everything here degrades to a no-op when unconfigured, so a plain install keeps
working exactly as before.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import factory_section

log = logging.getLogger("krish.storage")


class ObjectStore(ABC):
    """Minimal contract. Every method is allowed to fail softly and return False."""

    enabled: bool = False
    backend: str = "local"

    @abstractmethod
    def put(self, local_path: Path, key: str) -> str | None:
        """Upload. Returns a URL when the store can produce one."""

    @abstractmethod
    def get(self, key: str, local_path: Path) -> bool:
        """Download into ``local_path``. Returns False if the key is absent."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete(self, key: str) -> bool: ...

    def url_for(self, key: str) -> str | None:
        return None

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend, "enabled": self.enabled}


class LocalStore(ObjectStore):
    """No remote at all — the default, and a perfectly good choice."""

    enabled = False
    backend = "local"

    def put(self, local_path: Path, key: str) -> str | None:
        return None

    def get(self, key: str, local_path: Path) -> bool:
        return False

    def exists(self, key: str) -> bool:
        return False

    def delete(self, key: str) -> bool:
        return False


class S3Store(ObjectStore):
    """S3-compatible object storage.

    Cloudflare R2 is the recommended pairing: no egress fees, 10 GB free, and it
    speaks plain S3 so nothing here is vendor-specific.
    """

    backend = "s3"

    def __init__(self) -> None:
        self.bucket = os.getenv("S3_BUCKET", "")
        self.endpoint = os.getenv("S3_ENDPOINT_URL", "") or None
        self.region = os.getenv("S3_REGION", "auto")
        self.public_base = os.getenv("S3_PUBLIC_BASE_URL", "").rstrip("/")
        self.prefix = str(factory_section("storage").get("prefix", "krish")).strip("/")
        self._client: Any = None
        self.enabled = bool(
            self.bucket and os.getenv("S3_ACCESS_KEY_ID") and os.getenv("S3_SECRET_ACCESS_KEY")
        )
        if not self.enabled:
            log.warning(
                "storage.backend is 's3' but S3_BUCKET / S3_ACCESS_KEY_ID / "
                "S3_SECRET_ACCESS_KEY are not all set — staying local"
            )

    # ------------------------------------------------------------------ #

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3  # imported lazily: only needed with the cloud extra
                from botocore.config import Config
            except ImportError as exc:  # pragma: no cover
                self.enabled = False
                raise RuntimeError(
                    "object storage needs the optional extra: pip install -e 'backend[cloud]'"
                ) from exc
            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                region_name=self.region,
                aws_access_key_id=os.getenv("S3_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY"),
                config=Config(
                    signature_version="s3v4",
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        return self._client

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key.lstrip('/')}" if self.prefix else key.lstrip("/")

    def put(self, local_path: Path, key: str) -> str | None:
        if not self.enabled or not local_path.exists():
            return None
        full = self._key(key)
        try:
            self.client.upload_file(str(local_path), self.bucket, full)
        except Exception as exc:
            log.warning("upload failed for %s: %s", full, exc)
            return None
        log.info("uploaded %s (%.1f KB)", full, local_path.stat().st_size / 1024)
        return self.url_for(key)

    def get(self, key: str, local_path: Path) -> bool:
        if not self.enabled:
            return False
        full = self._key(key)
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(self.bucket, full, str(local_path))
        except Exception:
            return False
        log.info("restored %s from object storage", full)
        return True

    def exists(self, key: str) -> bool:
        if not self.enabled:
            return False
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        if not self.enabled:
            return False
        try:
            self.client.delete_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception as exc:
            log.warning("delete failed for %s: %s", key, exc)
            return False

    def url_for(self, key: str) -> str | None:
        if not self.enabled:
            return None
        full = self._key(key)
        if self.public_base:
            return f"{self.public_base}/{full}"
        # No public base configured: hand back a time-limited signed link so the
        # bucket can stay private, which is the right default for a trading repo.
        try:
            return self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": full},
                ExpiresIn=7 * 24 * 3600,
            )
        except Exception:
            return None

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "s3",
            "enabled": self.enabled,
            "bucket": self.bucket,
            "endpoint": self.endpoint or "aws",
            "prefix": self.prefix,
            "public_base": self.public_base or "(presigned links)",
        }


@lru_cache(maxsize=1)
def store() -> ObjectStore:
    backend = str(factory_section("storage").get("backend", "local")).lower()
    if backend in {"s3", "r2", "b2", "minio"}:
        s3 = S3Store()
        return s3 if s3.enabled else LocalStore()
    return LocalStore()


def reset_store() -> None:
    """Drop the cached instance so a config change takes effect."""
    store.cache_clear()


# --------------------------------------------------------------------------- #
# key naming — stable and human-browsable in the bucket
# --------------------------------------------------------------------------- #


def cache_key(asset: str, timeframe: str) -> str:
    return f"cache/{asset.upper()}_{timeframe.upper()}.parquet"


def package_key(package_name: str) -> str:
    return f"packages/{package_name}.zip"


def offload_price_cache() -> bool:
    cfg = factory_section("storage")
    return bool(cfg.get("offload_price_cache", True)) and store().enabled


def upload_packages() -> bool:
    cfg = factory_section("storage")
    return bool(cfg.get("upload_packages", True)) and store().enabled


def keep_local_packages() -> bool:
    """When False, the local ZIP is removed once it is safely in the cloud."""
    return bool(factory_section("storage").get("keep_local_packages", True))


__all__ = [
    "LocalStore",
    "ObjectStore",
    "S3Store",
    "cache_key",
    "keep_local_packages",
    "offload_price_cache",
    "package_key",
    "reset_store",
    "store",
    "upload_packages",
]
