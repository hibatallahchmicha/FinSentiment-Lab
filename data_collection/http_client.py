"""
data_collection/http_client.py
-------------------------------
Thin wrapper around `requests` that adds:
  - automatic retries with exponential back-off
  - consistent timeout enforcement
  - structured error logging

Every network call in the pipeline goes through this client so retry
behaviour is centralised and easy to tune in config/settings.py.
"""

import time
from typing import Any, Dict, Optional

import requests
from requests import Response

from config.logger import get_logger
from config.settings import HTTP_TIMEOUT, MAX_RETRIES, RETRY_BACKOFF

log = get_logger(__name__)


class HTTPClient:
    """Reusable session-based HTTP client with retry logic."""

    def __init__(self, base_url: str = "", default_params: Optional[Dict] = None):
        self.base_url = base_url
        self.default_params = default_params or {}
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get(self, endpoint: str = "", params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        GET request with retries.

        Parameters
        ----------
        endpoint : str
            Path appended to base_url (or a full URL if base_url is empty).
        params : dict
            Query parameters merged with default_params.

        Returns
        -------
        dict
            Parsed JSON body.

        Raises
        ------
        requests.HTTPError
            When all retries are exhausted or a non-retryable status is returned.
        """
        url = f"{self.base_url}{endpoint}" if endpoint else self.base_url
        merged_params = {**self.default_params, **(params or {})}

        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                log.debug("GET %s | attempt %d/%d | params=%s",
                          url, attempt, MAX_RETRIES, list(merged_params.keys()))

                response: Response = self._session.get(
                    url, params=merged_params, timeout=HTTP_TIMEOUT
                )

                # 429 = rate-limited → always retry
                if response.status_code == 429:
                    wait = RETRY_BACKOFF ** attempt
                    log.warning("Rate-limited (429). Waiting %.1fs before retry.", wait)
                    time.sleep(wait)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout as exc:
                last_exc = exc
                log.warning("Timeout on attempt %d/%d for %s", attempt, MAX_RETRIES, url)

            except requests.exceptions.ConnectionError as exc:
                last_exc = exc
                log.warning("Connection error on attempt %d/%d for %s", attempt, MAX_RETRIES, url)

            except requests.exceptions.HTTPError as exc:
                # 4xx (except 429) are not worth retrying
                if response.status_code < 500:
                    log.error("Non-retryable HTTP %d for %s: %s",
                              response.status_code, url, response.text[:200])
                    raise
                last_exc = exc
                log.warning("Server error %d on attempt %d/%d",
                            response.status_code, attempt, MAX_RETRIES)

            # Exponential back-off before next attempt
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF ** attempt
                log.info("Retrying in %.1fs…", wait)
                time.sleep(wait)

        raise requests.exceptions.RetryError(
            f"All {MAX_RETRIES} retries exhausted for {url}"
        ) from last_exc

    def close(self):
        self._session.close()

    # Context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()