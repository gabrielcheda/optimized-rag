"""
Retry utilities with exponential backoff for API calls
Improves reliability by handling transient failures
"""

import logging

from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def create_retry_decorator(
    max_attempts: int = 3, min_wait: int = 2, max_wait: int = 10, multiplier: int = 1
):
    """
    Create a retry decorator for API calls with exponential backoff

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time in seconds
        max_wait: Maximum wait time in seconds
        multiplier: Multiplier for exponential backoff

    Returns:
        Configured retry decorator

    Example:
        @create_retry_decorator(max_attempts=3)
        def api_call():
            return client.embeddings.create(...)
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(
            (
                APIError,
                APIConnectionError,
                RateLimitError,
                APITimeoutError,
                ConnectionError,
                TimeoutError,
            )
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


# Pre-configured decorator for embedding API calls
retry_embedding_call = create_retry_decorator(max_attempts=3, min_wait=2, max_wait=15)
