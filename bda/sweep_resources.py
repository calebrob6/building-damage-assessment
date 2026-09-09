# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Conservative admission control for jobs sharing a Linux cgroup v2."""

from __future__ import annotations

from pathlib import Path

GIB = 1024**3


class MemoryPressureError(RuntimeError):
    """The cgroup no longer has enough safe working-set headroom."""


def memory_snapshot(root=Path("/sys/fs/cgroup")) -> dict:
    root = Path(root)
    maximum = (root / "memory.max").read_text().strip()
    if maximum == "max":
        raise RuntimeError("A finite cgroup v2 memory.max is required for memory admission")
    limit = int(maximum)
    if limit <= 0:
        raise ValueError("Invalid cgroup memory limit")
    high = (root / "memory.high").read_text().strip()
    if high != "max":
        limit = min(limit, int(high))
    current = int((root / "memory.current").read_text())
    stats = {key: int(value) for key, value in
             (line.split() for line in (root / "memory.stat").read_text().splitlines())}
    events = {key: int(value) for key, value in
              (line.split() for line in (root / "memory.events").read_text().splitlines())}
    # Keep active, dirty and writeback pages in the budget; never drop shared caches.
    reclaimable = max(0, stats["inactive_file"] - stats["file_dirty"] - stats["file_writeback"])
    return {
        "limit_bytes": limit,
        "current_bytes": current,
        "working_set_bytes": max(0, current - reclaimable),
        "reclaimable_file_bytes": reclaimable,
        "anonymous_bytes": stats["anon"],
        "shmem_bytes": stats["shmem"],
        "oom_kills": events["oom_kill"],
        "oom_group_kills": events.get("oom_group_kill", 0),
    }


def cpu_quota(root=Path("/sys/fs/cgroup")) -> float | None:
    quota, period = (Path(root) / "cpu.max").read_text().split()
    return None if quota == "max" else int(quota) / int(period)


def process_tree_pss(pid: int, proc=Path("/proc")) -> int:
    """Count shared mappings proportionally, rather than summing forked-worker RSS."""
    pending, seen, total = [int(pid)], set(), 0
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        directory = Path(proc) / str(current)
        try:
            children = (directory / "task" / str(current) / "children").read_text()
            pending.extend(int(child) for child in children.split())
            lines = (directory / "smaps_rollup").read_text().splitlines()
        except (FileNotFoundError, ProcessLookupError):
            continue
        values = [int(line.split()[1]) * 1024 for line in lines if line.startswith("Pss:")]
        if len(values) != 1:
            raise ValueError(f"Missing or ambiguous PSS for process {current}")
        total += values[0]
    return total


def admission(snapshot: dict, active_pss: list[int], policy: dict) -> dict:
    reserve = policy["reserve_bytes"]
    per_job = policy["per_job_bytes"]
    if reserve <= 0 or per_job <= 0:
        raise ValueError("Memory reserve and per-job budgets must be positive")
    pending_growth = sum(max(0, per_job - measured) for measured in active_pss)
    projected = snapshot["working_set_bytes"] + pending_growth + per_job
    return {
        **snapshot,
        "active_job_pss_bytes": active_pss,
        "reserved_growth_bytes": pending_growth,
        "new_job_budget_bytes": per_job,
        "safety_reserve_bytes": reserve,
        "projected_working_set_bytes": projected,
        "allowed": projected + reserve <= snapshot["limit_bytes"],
    }


def check_pressure(snapshot: dict, policy: dict, baseline_events: dict) -> None:
    if (snapshot["oom_kills"] > baseline_events["oom_kills"]
            or snapshot["oom_group_kills"] > baseline_events["oom_group_kills"]):
        raise MemoryPressureError("The cgroup reported a new OOM kill; stopping owned jobs")
    if snapshot["working_set_bytes"] + policy["reserve_bytes"] > snapshot["limit_bytes"]:
        raise MemoryPressureError("Cgroup working-set safety reserve exhausted; stopping owned jobs")
