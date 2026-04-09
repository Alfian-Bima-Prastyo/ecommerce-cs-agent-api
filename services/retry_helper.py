import time
import functools
from openai import RateLimitError, APIConnectionError, APITimeoutError, BadRequestError

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

                except APITimeoutError as e:
                    last_error = e

                except APIConnectionError as e:
                    last_error = e

                except BadRequestError:
                    raise

                except Exception:
                    raise

                if attempt < max_retries:
                    wait = backoff * (attempt + 1)
                    print(f"[RETRY] {type(last_error).__name__}, wait {wait}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait)

            raise last_error
        return wrapper
    return decorator