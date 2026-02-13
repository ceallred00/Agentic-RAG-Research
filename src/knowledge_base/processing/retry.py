import time
import logging
from typing import Callable, Tuple, Type, TypeVar

T = TypeVar("T")

logger = logging.getLogger(__name__)

def retry_with_backoff(
        fn: Callable[[], T], 
        max_retries: int, 
        initial_delay: int, 
        max_delay: int, 
        retryable_on: Tuple[Type[Exception], ...]) -> T:
    """
    Executes a callable with exponential backoff retry logic for rate-limited APIs.

    On each retryable failure, the delay doubles: initial_delay * 2^attempt,
    capped at max_delay. Non-retryable exceptions propagate immediately.

    Args:
        fn (Callable): A zero-argument callable wrapping the API call (e.g., lambda: client.embed(batch)).
        max_retries (int): Maximum number of retry attempts after the initial call.
            Total attempts = max_retries + 1.
        initial_delay (int): Base delay in seconds before the first retry.
        max_delay (int): Upper bound on delay in seconds to prevent excessive waits.
        retryable_on (Tuple[Type[Exception], ...]): Exception types that should trigger a retry
            (e.g., ResourceExhausted, PineconeApiException). All other exceptions propagate immediately.

    Returns:
        The return value of fn() on success.

    Raises:
        The original retryable exception if all retry attempts are exhausted.
        Any non-retryable exception immediately on first occurrence.

    Example:
        >>> from google.api_core.exceptions import ResourceExhausted
        >>> result = retry_with_backoff(
        ...     fn=lambda: embedding_model.embed_documents(batch),
        ...     max_retries=5,
        ...     initial_delay=2,
        ...     max_delay=60,
        ...     retryable_on=(ResourceExhausted,)
        ... )
    """
    for attempt in range(max_retries + 1): # 1 initial attempt + max_retries
        try: 
            return fn()
        except retryable_on as e:
            if attempt == max_retries:
                raise
            delay = min(initial_delay * (2**attempt), max_delay)
            logger.warning(f"Rate limited. Attempt {attempt + 1}/{max_retries}. Retrying in {delay}s.")
            time.sleep(delay)
        
    # Unreachable: loop always returns or raises, but satisfies type checker
    raise RuntimeError("retry_with_backoff exhausted all attempts without returning or raising")