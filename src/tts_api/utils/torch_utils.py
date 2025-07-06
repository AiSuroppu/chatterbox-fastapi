import torch
import logging
from typing import Callable, TypeVar, Iterator

# Define a generic TypeVar for robust type hinting of the return value.
T = TypeVar('T')

def _cuda_oom_batch_size_generator(initial_batch_size: int, model_name: str) -> Iterator[int]:
    """
    A private generator for the batch size reduction strategy.

    - First attempt: uses the initial_batch_size.
    - On first OOM: reduces batch size by 20% (or at least 1).
    - On subsequent OOMs: halves the batch size.
    - The final attempt is always with a batch size of 1.
    """
    batch_size = initial_batch_size
    if batch_size <= 0:
        return

    # Attempt 1: Initial Size
    yield batch_size

    # Subsequent attempts use a smaller batch size
    while batch_size > 1:
        if batch_size == initial_batch_size:
            # First failure: 20% reduction
            reduction = max(1, int(batch_size * 0.2))
            next_batch_size = batch_size - reduction
        else:
            # Subsequent failures: Halving
            next_batch_size = batch_size // 2

        # Ensure we don't propose a batch size of 0 or get stuck
        next_batch_size = max(1, next_batch_size)
        if next_batch_size < batch_size:
            batch_size = next_batch_size
            yield batch_size
        else:
            # Stop if reduction doesn't decrease the batch size
            break

def run_with_oom_retry(
    fn: Callable[..., T],
    initial_batch_size: int,
    model_name: str,
    device: str,
    **kwargs,
) -> T:
    """
    Executes a function with a robust CUDA OOM retry mechanism.

    This higher-order function abstracts the entire retry loop. It calls the
    provided function `fn`, passing it a `batch_size` from a dynamically
    reducing sequence. If `fn` throws a `torch.cuda.OutOfMemoryError`, it
    catches it, reduces the batch size, and retries.
    """
    if 'cuda' not in device.lower():
        return fn(batch_size=initial_batch_size, **kwargs)

    last_oom_error = None
    for batch_size in _cuda_oom_batch_size_generator(initial_batch_size, model_name):
        try:
            logging.debug(f"Attempting '{model_name}' with batch size {batch_size}")
            # The provided function `fn` must accept `batch_size` as a keyword argument.
            result = fn(batch_size=batch_size, **kwargs)
            return result  # Success!
        except torch.cuda.OutOfMemoryError as e:
            last_oom_error = e
            torch.cuda.empty_cache()
            logging.warning(
                f"CUDA OOM detected for {model_name} with batch size {batch_size}. Retrying..."
            )
            continue

    logging.error(f"'{model_name}' failed with CUDA OOM even at batch size 1.")
    raise RuntimeError(f"Operation '{model_name}' failed due to CUDA out of memory.") from last_oom_error