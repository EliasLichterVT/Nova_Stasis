import os
from urllib.parse import urlsplit

import httpx


class ComfyError(Exception):
    pass


class Comfy:
    def __init__(self):
        url = os.environ.get("NOVA_COMFY_URL", "http://127.0.0.1:8188").rstrip("/")
        parsed = urlsplit(url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
                or parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment):
            raise RuntimeError("NOVA_COMFY_URL must be a loopback HTTP origin.")
        self.client = httpx.AsyncClient(base_url=url, timeout=30, trust_env=False)

    async def close(self):
        await self.client.aclose()

    async def request(self, path, **kwargs):
        try:
            response = await self.client.request("POST" if "json" in kwargs or "files" in kwargs
                                                 else "GET", path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            if path == "/prompt":
                try:
                    errors = exc.response.json().get("node_errors", {})
                    messages = [
                        f"{node.get('class_type', 'Node')}: {error.get('message', 'Invalid input')}"
                        for node in errors.values() for error in node.get("errors", [])
                    ]
                    if messages:
                        raise ComfyError("; ".join(messages[:5])) from exc
                except ValueError:
                    pass
            raise ComfyError(f"ComfyUI returned HTTP {exc.response.status_code} for {path}.") from exc
        except httpx.RequestError as exc:
            raise ComfyError("ComfyUI did not respond. Check that it is running on the configured port.") from exc

    async def upload(self, name, content):
        result = (await self.request("/upload/image", files={
            "image": (f"{name}.png", content, "image/png"),
        }, data={"subfolder": "nova_stasis", "type": "input", "overwrite": "true"})).json()
        return "/".join(filter(None, [result.get("subfolder"), result["name"]]))
