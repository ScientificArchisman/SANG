"""Task functions for the sang.data.pooled tests.

Kept in its own module with no heavy imports: spawn workers import the module that defines the
mapped function, so putting these in test_data.py would pull torch/decord into every worker.
"""
import os
import time


def crash_on_three(x):
    """Dies without writing a result frame -- the job 162848 failure mode."""
    if x == 3:
        time.sleep(1.0)  # let the instant tasks land first, so the casualty is unambiguous
        os._exit(1)
    return x


def wedge_on_three(x):
    if x == 3:
        time.sleep(3600)
    return x


def worker_pid(x):
    return os.getpid()
