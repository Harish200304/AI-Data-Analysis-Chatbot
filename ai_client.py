import http.client
import json
import logging
import os
import socket
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

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


def _diagnose_host(url):
    """Best-effort low-level connectivity check, used only to enrich error messages."""
    try:
        host = urlparse(url).hostname
    except Exception:
        return "could not parse target host from URL"
    if not host:
        return "could not parse target host from URL"
    try:
        start = time.monotonic()
        socket.getaddrinfo(host, 443)
        dns_time = time.monotonic() - start
    except Exception as exc:
        return f"DNS lookup for {host} failed: {exc}. Outbound DNS may be blocked from this host."
    try:
        start = time.monotonic()
        with socket.create_connection((host, 443), timeout=8):
            connect_time = time.monotonic() - start
        return (
            f"DNS resolved {host} in {dns_time:.2f}s and a raw TCP connection to port 443 "
            f"succeeded in {connect_time:.2f}s, so the host is reachable. The API/proxy in "
            f"between is accepting the connection but not sending a response within the "
            f"timeout — this points at an egress proxy, firewall, or rate limiter silently "
            f"dropping the request, not a slow model."
        )
    except Exception as exc:
        return (
            f"DNS resolved {host} in {dns_time:.2f}s but a raw TCP connection to port 443 "
            f"failed: {exc}. Outbound network access to {host} appears to be blocked from "
            f"this deployment environment — check your platform's egress/firewall settings."
        )


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
            if exc.code in (401, 403):
                raise AIClientError(
                    f"NVIDIA API rejected the request (HTTP {exc.code}): {detail}. "
                    f"This usually means the API key is invalid/expired, or the key does "
                    f"not have access to the model '{model}'. NVIDIA API keys are often "
                    f"scoped to specific models — try switching back to the model this key "
                    f"was issued for, or check https://build.nvidia.com for which models "
                    f"your key can access."
                ) from exc
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
                diagnosis = _diagnose_host(url)
                raise AIClientError(
                    "NVIDIA API request timed out after several attempts. "
                    f"Diagnosis: {diagnosis}"
                ) from exc
            raise AIClientError(f"Could not reach NVIDIA API: {reason}") from exc
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            if attempt < retries:
                logger.warning("NVIDIA API error on attempt %s/%s: %s", attempt, retries, exc)
                time.sleep(min(2 ** attempt, 10))
                continue
            diagnosis = _diagnose_host(url)
            raise AIClientError(
                "NVIDIA API request timed out after several attempts. "
                f"Diagnosis: {diagnosis}"
            ) from exc

    if data is None:
        raise AIClientError("NVIDIA API request failed with no response.") from last_error

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AIClientError(f"Unexpected NVIDIA API response: {data}") from exc