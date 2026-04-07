import time
import functools
from groq import BadRequestError, RateLimitError, APIConnectionError, APITimeoutError

def with_retry(max_retries: int = 2, backoff: float = 2.0):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except RateLimitError as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = backoff * (attempt + 1)
                        print(f"[RETRY] RateLimitError, wait {wait}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                except APITimeoutError as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = backoff * (attempt + 1)
                        print(f"[RETRY] TimeoutError, wait {wait}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                except APIConnectionError as e:
                    last_error = e
                    if attempt < max_retries:
                        wait = backoff * (attempt + 1)
                        print(f"[RETRY] ConnectionError, wait {wait}s (attempt {attempt+1}/{max_retries})")
                        time.sleep(wait)
                except BadRequestError:
                    raise
                except Exception:
                    raise
            raise last_error
        return wrapper
    return decorator