import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import os
from tqdm import tqdm


def _worker_index_wrapper(item):
    """Top-level worker wrapper so it is pickleable.
    Receives (idx, eval_fn, coords) and returns (idx, result or Exception).
    """
    idx, eval_fn, coords = item
    try:
        return idx, eval_fn(*coords)
    except Exception as e:
        return idx, e

def evaluate_multithread(eval_fn, eval_list, show_pbar=True, max_workers=None, chunksize=8, start_method=None, set_single_thread_blas=False):
    """
    Evaluate a function over a list of inputs using multiprocessing with ordered results.

    This function distributes work across multiple processes, evaluates each input,
    and returns results in the original input order. Failed evaluations are returned
    as Exception objects rather than raising, allowing partial results to be collected.

    Args:
        eval_fn (callable): Function to apply to each element. Should accept a single
            argument from eval_list and return a result.
        eval_list (list): List of inputs to process. Each element will be passed to eval_fn.
        show_pbar (bool, optional): Whether to display a tqdm progress bar. Defaults to True.
        max_workers (int, optional): Maximum number of worker processes. If None, defaults
            to the minimum of physical CPU cores and length of eval_list. Defaults to None.
        chunksize (int, optional): Number of items to send to each worker at once for
            processing. Larger chunks reduce overhead but may reduce load balancing.
            Defaults to 8.
        start_method (str, optional): Multiprocessing start method ('fork', 'spawn', or
            'forkserver'). If None, uses the platform default. Defaults to None.
        set_single_thread_blas (bool, optional): If True, sets environment variables to
            limit BLAS/OpenMP libraries to single-threaded mode within workers, reducing
            thread contention. Defaults to False.

    Returns:
        list: Results in the same order as eval_list. Failed evaluations are returned
            as Exception objects at their corresponding positions.

    Notes:
        - Results maintain input order despite using imap_unordered internally
        - Exceptions during evaluation are caught and stored rather than propagated
        - If errors occur, a summary is printed showing count and first error
        - Returns empty list if max_workers would be < 1
    """

    if max_workers is None:
        # Cap at physical cores available but never exceed length of work
        max_workers = min(cpu_count(), len(eval_list))
    if max_workers < 1:
        return []

    # Optional: reduce nested thread contention from BLAS/OpenMP libs
    if set_single_thread_blas:
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ.setdefault(var, "1")

    # Package arguments with index so we can restore ordering after unordered completion
    packaged = [(i, eval_fn, coords) for i, coords in enumerate(eval_list)]

    results = [None] * len(eval_list)
    errors = 0
    if show_pbar:
        pbar = tqdm(total=len(eval_list))

    # Streaming consumption
    ctx = mp.get_context(start_method) if start_method else mp.get_context()
    with ctx.Pool(processes=max_workers) as pool:
        for idx, value in pool.imap_unordered(_worker_index_wrapper, packaged, chunksize):
            if isinstance(value, Exception):
                errors += 1
                results[idx] = value
            else:
                results[idx] = value
            if show_pbar:
                pbar.update(1)

    if show_pbar:
        pbar.close()

    if errors:
        print(f"Parallel evaluation: {errors} tasks raised exceptions (stored as Exception).")
        print("First error:", next(r for r in results if isinstance(r, Exception)))

    return results