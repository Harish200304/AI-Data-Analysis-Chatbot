import json
import os
import urllib.error
import urllib.request


DEFAULT_MODEL = "abacusai/dracarys-llama-3.1-70b-instruct"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


class AIClientError(RuntimeError):
    pass


def chat_completion(messages, api_key=None, model=DEFAULT_MODEL, temperature=0.2, max_tokens=900):
    token = api_key or os.getenv("NVIDIA_API_KEY") or os.getenv("NVAPI_KEY")
    if not token:
        raise AIClientError("Missing API key. Set NVIDIA_API_KEY or enter it in the app.")

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": 0.8,
    }
    request = urllib.request.Request(
        f"{os.getenv('NVIDIA_BASE_URL', DEFAULT_BASE_URL).rstrip('/')}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise AIClientError(f"NVIDIA API returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise AIClientError(f"Could not reach NVIDIA API: {exc.reason}") from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AIClientError(f"Unexpected NVIDIA API response: {data}") from exc
