"""Storage abstraction supporting Google Cloud Storage and Local filesystem."""
from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional


class Storage(ABC):
    """Abstract storage interface."""

    @abstractmethod
    def put_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> None:
        """Write raw bytes to the given storage path."""
        pass

    @abstractmethod
    def get_bytes(self, path: str) -> bytes:
        """Read raw bytes from the given storage path. Raises FileNotFoundError if missing."""
        pass

    def put_json(self, path: str, obj: Any) -> None:
        """Serialize an object to JSON and write to storage."""
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.put_bytes(path, data, content_type="application/json")

    def get_json(self, path: str) -> Any:
        """Read and parse JSON from storage."""
        data = self.get_bytes(path)
        return json.loads(data.decode("utf-8"))

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Return True if path exists in storage."""
        pass

    @abstractmethod
    def list(self, prefix: str) -> list[str]:
        """List all paths matching the prefix in storage."""
        pass


class LocalStorage(Storage):
    """Local filesystem storage backend for local dev and tests."""

    def __init__(self, base_dir: str | Path):
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        clean = path.strip("/")
        resolved = (self.base_dir / clean).resolve()
        if not str(resolved).startswith(str(self.base_dir)):
            raise ValueError(f"Path traversal detected: {path}")
        return resolved

    def put_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> None:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_file = target.with_name(f".{target.name}.tmp.{os.getpid()}")
        temp_file.write_bytes(data)
        temp_file.replace(target)

    def get_bytes(self, path: str) -> bytes:
        target = self._resolve(path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return target.read_bytes()

    def exists(self, path: str) -> bool:
        target = self._resolve(path)
        return target.is_file()

    def list(self, prefix: str) -> list[str]:
        clean_prefix = prefix.strip("/")
        results: list[str] = []
        if not self.base_dir.exists():
            return results

        for root, _, files in os.walk(self.base_dir):
            for file in files:
                full_path = Path(root) / file
                rel_path = str(full_path.relative_to(self.base_dir)).replace(os.sep, "/")
                if not clean_prefix or rel_path.startswith(clean_prefix):
                    results.append(rel_path)
        return sorted(results)


class GCSStorage(Storage):
    """Google Cloud Storage backend for production Cloud Run deployment."""

    def __init__(self, bucket_name: str, client: Any = None):
        self.bucket_name = bucket_name
        if client is None:
            from google.cloud import storage as gcs_module
            self.client = gcs_module.Client()
        else:
            self.client = client
        self._bucket = None

    @property
    def bucket(self):
        if self._bucket is None:
            self._bucket = self.client.bucket(self.bucket_name)
        return self._bucket

    def put_bytes(self, path: str, data: bytes, content_type: Optional[str] = None) -> None:
        clean = path.strip("/")
        blob = self.bucket.blob(clean)
        blob.upload_from_string(data, content_type=content_type)

    def get_bytes(self, path: str) -> bytes:
        clean = path.strip("/")
        blob = self.bucket.blob(clean)
        if not blob.exists():
            raise FileNotFoundError(f"File not found in GCS: {path}")
        return blob.download_as_bytes()

    def exists(self, path: str) -> bool:
        clean = path.strip("/")
        blob = self.bucket.blob(clean)
        return blob.exists()

    def list(self, prefix: str) -> list[str]:
        clean_prefix = prefix.strip("/")
        blobs = self.client.list_blobs(self.bucket_name, prefix=clean_prefix)
        return sorted([b.name for b in blobs])


_storage_instance: Optional[Storage] = None


def get_storage() -> Storage:
    """Factory returning GCSStorage if BUCKET env is set, else LocalStorage."""
    global _storage_instance
    if _storage_instance is not None:
        return _storage_instance

    bucket = os.environ.get("BUCKET")
    if bucket:
        _storage_instance = GCSStorage(bucket)
    else:
        local_dir = os.environ.get("LOCAL_STORAGE_DIR", "out/storage")
        _storage_instance = LocalStorage(local_dir)
    return _storage_instance


def set_storage(storage: Optional[Storage]) -> None:
    """Set or reset storage backend instance (useful for unit tests)."""
    global _storage_instance
    _storage_instance = storage
