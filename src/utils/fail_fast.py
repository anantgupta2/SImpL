"""Fail-fast guard: if training throws, TERMINATE the job instead of letting
launchpad hang with a live actor/GPU (which silently wastes compute until a
manual scancel).

Wrap the hot entry points (learner.run, actor.step) with @fail_fast(...). On an
uncaught Exception it prints the traceback, cancels the whole SLURM job (so every
launchpad node / the GPU is released), and hard-exits the current process.

We catch ``Exception`` only -- NOT ``BaseException`` -- so we never interfere
with launchpad's normal SystemExit/KeyboardInterrupt-based shutdown of a run that
finished cleanly.
"""
from __future__ import annotations

import functools
import os
import subprocess
import sys
import traceback


def _terminate_job(where: str) -> None:
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    print(f"[fail_fast] uncaught exception in {where}; terminating job to free compute.", flush=True)
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    # Dump the traceback to a DURABLE file too: in launchpad the actor's stdout goes to a
    # child stream that scancel kills before it flushes, so the real error otherwise never
    # surfaces in the slurm log (you only see the bare SIGTERM). This file survives.
    try:
        fname = f"fail_fast_{job_id or 'nojob'}_{os.getpid()}.log"
        with open(fname, "w") as f:
            f.write(f"[fail_fast] uncaught exception in {where} (pid {os.getpid()}, job {job_id})\n")
            traceback.print_exc(file=f)
        print(f"[fail_fast] traceback written to {os.path.abspath(fname)}", flush=True)
    except Exception:
        pass
    if job_id:
        # Cancel the entire job -> releases the GPU and every launchpad node,
        # regardless of which node raised.
        try:
            subprocess.call(["scancel", str(job_id)])
        except Exception:
            pass
    # Hard-exit this process immediately as a fallback (skips atexit/cleanup).
    os._exit(1)


def fail_fast(where: str):
    """Decorator: on an uncaught Exception, kill the job and exit."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception:
                _terminate_job(where)
        return wrapper
    return decorator
