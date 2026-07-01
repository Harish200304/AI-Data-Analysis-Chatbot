import http.client
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request


DEFAULT_MODEL = "meta/llama-3.3-70b-instruct"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
# A general instruct model responds in a few seconds normally; keep enough
# headroom for longer document-generation prompts without masking real hangs.
DEFAULT_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

# Any of these mean "the network/socket hiccuped", not "the request is bad".
# They're all safe to retry.
RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    socket.timeout,
    ConnectionError,
    http.client.HTTPException,
    OSError,
)

logger = logging.getLogger(__name__)


class AIClientError(RuntimeError):
    pass


def chat_completion(messages, api_key=None, model=DEFAULT_MODEL, temperature=0.2, max_tokens=900, timeout=DEFAULT_TIMEOUT_SECONDS, retries=MAX_RETRIES):
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
    body = json.dumps(payload).encode("utf-8")
    url = f"{os.getenv('NVIDIA_BASE_URL', DEFAULT_BASE_URL).rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    data = None
    last_error = None
    for attempt in range(1, retries + 1):
        # Build a fresh Request each attempt instead of reusing one object.
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            # 429 (rate limited) and 5xx are worth retrying; other 4xx are not.
            if exc.code == 429 or exc.code >= 500:
                last_error = AIClientError(f"NVIDIA API returned HTTP {exc.code}: {detail}")
                if attempt < retries:
                    logger.warning("NVIDIA API HTTP %s on attempt %s/%s, retrying", exc.code, attempt, retries)
                    time.sleep(min(2 ** attempt, 10))
                    continue
                raise last_error from exc
            raise AIClientError(f"NVIDIA API returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            reason = exc.reason
            last_error = exc
            if attempt < retries:
                logger.warning("NVIDIA API network error on attempt %s/%s: %s", attempt, retries, reason)
                time.sleep(min(2 ** attempt, 10))
                continue
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise AIClientError(
                    "NVIDIA API request timed out after several attempts. "
                    "The request may be too large, or the service may be under load. "
                    "Try again, or shorten the document/question."
                ) from exc
            raise AIClientError(f"Could not reach NVIDIA API: {reason}") from exc
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt < retries:
                logger.warning("NVIDIA API error on attempt %s/%s: %s", attempt, retries, exc)
                time.sleep(min(2 ** attempt, 10))
                continue
            raise AIClientError(
                "NVIDIA API request timed out after several attempts. "
                "The request may be too large, or the service may be under load. "
                "Try again, or shorten the document/question."
            ) from exc

    if data is None:
        raise AIClientError("NVIDIA API request failed with no response.") from last_error

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AIClientError(f"Unexpected NVIDIA API response: {data}") from exc