"""Store Nova's local Civitai credential without returning it to the browser."""

import os
import tempfile
from pathlib import Path


class CivitaiKeyStore:
    def __init__(self, directory):
        self.path = directory / ".civitai-api-key"

    def exists(self):
        return self.path.is_file()

    def read(self):
        try:
            payload = self.path.read_bytes()
        except FileNotFoundError:
            return ""
        if os.name == "nt":
            import pywintypes
            import win32crypt

            try:
                _, payload = win32crypt.CryptUnprotectData(payload, None, None, None, 1)
            except pywintypes.error as exc:
                raise OSError("Could not unlock the saved Civitai key.") from exc
        return payload.decode("utf-8")

    def save(self, key):
        payload = key.encode("utf-8")
        if os.name == "nt":
            import pywintypes
            import win32crypt

            try:
                payload = win32crypt.CryptProtectData(payload, "Nova Stasis Civitai key", None, None, None, 1)
            except pywintypes.error as exc:
                raise OSError("Could not protect the Civitai key.") from exc
        with tempfile.NamedTemporaryFile(dir=self.path.parent, prefix=".civitai-key-", delete=False) as output:
            temporary = Path(output.name)
            try:
                output.write(payload)
                output.close()
                temporary.replace(self.path)
            finally:
                output.close()
                temporary.unlink(missing_ok=True)

    def clear(self):
        self.path.unlink(missing_ok=True)
