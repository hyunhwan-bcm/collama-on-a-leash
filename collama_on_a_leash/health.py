from __future__ import annotations

import json
import urllib.error
import urllib.request


def check_health(address: str, port: int, *, timeout: float = 10.0) -> dict[str, object]:
    base_url = f"http://{address}:{port}"
    health = _get_json_or_text(f"{base_url}/health", timeout)
    models = _get_json_or_text(f"{base_url}/v1/models", timeout)
    return {
        "base_url": base_url,
        "health": health,
        "models": models,
    }


def _get_json_or_text(url: str, timeout: float) -> object:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body)
            except json.JSONDecodeError:
                return {"status": resp.status, "body": body}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{url} is not reachable: {exc}") from exc
