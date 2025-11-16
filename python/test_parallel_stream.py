#!/usr/bin/env python3
"""Minimal streaming parallel test for evaluate_multithread.
Run to verify no deadlock and progress advances.
"""
import sys, time, math
sys.path.insert(0, "./lensing_and_precession/")

from helper_functions import evaluate_multithread

# Simple CPU-bound function (simulate heavier work)
def heavy(x, y):
    acc = 0.0
    for i in range(50_000):  # small loop to burn time
        acc += math.sin(x) * math.cos(y) * 0.00001
    return acc + x * y


def main():
    coords = [(i, j) for i in range(6) for j in range(6)]  # 36 tasks
    start = time.time()
    results = evaluate_multithread(heavy, coords, show_pbar=True, max_workers=None, chunksize=4)
    elapsed = time.time() - start
    print("\nCompleted", len(results), "tasks in", f"{elapsed:.2f}s")
    print("First 5 results:", results[:5])
    if any(r is None for r in results):
        missing = sum(r is None for r in results)
        print(f"WARNING: {missing} tasks returned None (errors captured).")

if __name__ == "__main__":
    main()
