# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Bounded, attached experiment execution shared by staged studies and LR sweeps.

The backend supplies the existing GPU/PID/cgroup policy functions. Discovery can
append stage-continuation jobs while other trajectories are still running.
"""

import os
from pathlib import Path
import subprocess
import sys
import time

from .xview2 import read_json, write_json


def durable_json(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable artifact conflict: {path}")
        return
    write_json(path, value)
    sync_file(path)


def sync_file(path):
    path = Path(path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def execute_jobs(root, state, section, jobs, gpu_map, max_concurrent, validate,
                 command_builder, backend, gpu_wait_seconds=60, memory_policy=None,
                 discover=None, cpu_threads=2):
    """Execute jobs; a successful exit is insufficient without artifact validation.

    ``command_builder(job, uuid)`` returns (argv, output_directory). ``discover``
    receives the controller state and returns newly eligible jobs. IDs are
    immutable and unique across a section; a continuation needs a new job ID.
    """
    if not 1 <= max_concurrent <= len(gpu_map) or cpu_threads not in (1, 2):
        raise ValueError("Invalid concurrency or CPU-thread budget")
    pending, active, known = [], {}, {}
    blocked_since, blocked_reason = None, "No allowed GPU is available"
    baseline_memory = backend.memory_snapshot() if memory_policy else None

    def enroll(new_jobs):
        for job in new_jobs:
            if job["id"] in known:
                if known[job["id"]] != job:
                    raise ValueError(f"Job definition changed: {job['id']}")
                continue
            known[job["id"]] = job
            record = state[section].setdefault(
                job["id"], {**job, "status": "pending", "attempts": []},
            )
            if record["status"] == "complete":
                valid = False
                try:
                    validate(job)
                    valid = True
                finally:
                    if not valid:
                        error = sys.exc_info()[1]
                        record.update({"status": "failed", "error": str(error)})
                        write_json(root / "state.json", state)
                continue
            try:
                validate(job)
            except (FileNotFoundError, backend.IncompleteTrialError):
                pending.append(job)
            else:
                record["status"] = "complete"

    enroll(jobs)
    write_json(root / "state.json", state)
    try:
        while True:
            if discover:
                enroll(discover(state))
            if not pending and not active:
                break
            if memory_policy:
                snapshot = backend.memory_snapshot()
                state["memory"] = snapshot
                backend.check_pressure(snapshot, memory_policy, baseline_memory)
            for uuid in gpu_map.values():
                if not pending or len(active) >= max_concurrent:
                    break
                if uuid in active or not backend.available_gpu(uuid, gpu_map):
                    continue
                decision = None
                if memory_policy:
                    measured = [backend.process_tree_pss(item["process"].pid) for item in active.values()]
                    snapshot = backend.memory_snapshot()
                    backend.check_pressure(snapshot, memory_policy, baseline_memory)
                    decision = backend.admission(snapshot, measured, memory_policy)
                    state["memory_admission"] = decision
                    if not decision["allowed"]:
                        blocked_reason = "Cgroup memory admission has insufficient reserved headroom"
                        continue
                blocked_since = None
                job = pending.pop(0)
                record = state[section][job["id"]]
                prepared = False
                try:
                    command, output = command_builder(job, uuid)
                    prepared = True
                finally:
                    if not prepared:
                        error = sys.exc_info()[1]
                        record.update({"status": "failed", "error": f"Command preparation: {error}"})
                output = Path(output)
                output.mkdir(parents=True, exist_ok=True)
                log_path = output / f"attempt_{len(record['attempts']) + 1:03d}.log"
                # Distinct stages share a trajectory directory, not a log namespace.
                if log_path.exists():
                    log_path = output / f"{job['id'].replace('/', '_')}_attempt_{len(record['attempts']) + 1:03d}.log"
                log = log_path.open("x")
                attempt = {"started_at": time.time(), "uuid": uuid, "command": command,
                           "log": str(log_path), "status": "running"}
                if decision is not None:
                    attempt["memory_admission"] = decision
                record["status"] = "running"
                record["attempts"].append(attempt)
                write_json(root / "state.json", state)
                parent_pid = os.getpid()
                environment = backend.child_environment(uuid, gpu_map)
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
                    environment[key] = str(cpu_threads)
                runtime = output / "runtime"
                runtime.mkdir(exist_ok=True)
                environment["TMPDIR"] = str(runtime)
                process = None
                try:
                    process = subprocess.Popen(
                        command, cwd=backend.REPO, env=environment, stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        preexec_fn=lambda: backend.arm_parent_death_signal(parent_pid),
                    )
                finally:
                    if process is None:
                        error = sys.exc_info()[1]
                        log.close()
                        record["status"] = attempt["status"] = "failed"
                        attempt.update({"error": str(error), "finished_at": time.time()})
                attempt.update({"pid": process.pid,
                                "process_start_ticks": backend.process_start_ticks(process.pid)})
                active[uuid] = {"process": process, "log": log, "job": job, "record": record,
                                "attempt": attempt, "start": time.monotonic()}
                write_json(root / "state.json", state)
                print(f"Started {section}/{job['id']} on {uuid}, pid={process.pid}", flush=True)
            if pending and not active:
                if blocked_since is None:
                    blocked_since = time.monotonic()
                if time.monotonic() - blocked_since >= gpu_wait_seconds:
                    raise RuntimeError(f"{blocked_reason}; pending jobs retained for explicit resume")
            for uuid, item in list(active.items()):
                code = item["process"].poll()
                if code is None:
                    continue
                item["log"].close()
                attempt, record = item["attempt"], item["record"]
                attempt.update({"exit_code": code, "finished_at": time.time(),
                                "seconds": time.monotonic() - item["start"],
                                "status": "complete" if code == 0 else "failed"})
                record["status"] = attempt["status"]
                del active[uuid]
                if code == 0:
                    try:
                        validate(item["job"])
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        record["status"] = attempt["status"] = "failed"
                        record["error"] = str(error)
                write_json(root / "state.json", state)
                if record["status"] != "complete":
                    raise RuntimeError(f"{section}/{item['job']['id']} failed; see {attempt['log']}")
                print(f"Completed {section}/{item['job']['id']}", flush=True)
            if active or pending:
                state["updated_at"] = time.time()
                write_json(root / "state.json", state)
                time.sleep(5)
    finally:
        backend.stop_owned_children(active)
        for item in active.values():
            item["record"]["status"] = "interrupted"
            item["attempt"].update({
                "status": "interrupted", "exit_code": item["process"].returncode,
                "finished_at": time.time(), "seconds": time.monotonic() - item["start"],
            })
        write_json(root / "state.json", state)
