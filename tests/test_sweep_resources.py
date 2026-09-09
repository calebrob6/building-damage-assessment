"""Cgroup accounting and memory admission tests without touching live limits."""

from pathlib import Path
import tempfile
import unittest

from bda.sweep_resources import (
    GIB, MemoryPressureError, admission, check_pressure, cpu_quota,
    memory_snapshot, process_tree_pss,
)


class ResourceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        values = {
            "memory.max": str(96 * GIB),
            "memory.high": "max",
            "memory.current": str(95 * GIB),
            "memory.stat": f"inactive_file {65 * GIB}\nfile_dirty 0\nfile_writeback 0\nanon {13 * GIB}\nshmem {3 * GIB}\n",
            "memory.events": "oom_kill 3\noom_group_kill 0\n",
            "cpu.max": "800000 100000",
        }
        for name, content in values.items():
            (self.root / name).write_text(content)
        self.policy = {"reserve_bytes": 16 * GIB, "per_job_bytes": 8 * GIB}

    def test_cache_is_not_confused_with_nonreclaimable_memory(self):
        snapshot = memory_snapshot(self.root)
        self.assertEqual(snapshot["working_set_bytes"], 30 * GIB)
        self.assertEqual(snapshot["current_bytes"], 95 * GIB)
        self.assertEqual(cpu_quota(self.root), 8)
        decision = admission(snapshot, [3 * GIB] * 4, self.policy)
        self.assertTrue(decision["allowed"])
        self.assertEqual(decision["projected_working_set_bytes"], 58 * GIB)

    def test_startup_reservations_prevent_oversubscribing(self):
        snapshot = memory_snapshot(self.root)
        self.assertFalse(admission(snapshot, [0] * 6, self.policy)["allowed"])
        snapshot["working_set_bytes"] = 39 * GIB
        self.assertTrue(admission(snapshot, [3 * GIB] * 6, self.policy)["allowed"])
        self.assertFalse(admission(snapshot, [3 * GIB] * 7, self.policy)["allowed"])

    def test_dirty_pages_and_memory_high_are_conservative(self):
        (self.root / "memory.high").write_text(str(80 * GIB))
        (self.root / "memory.stat").write_text(
            f"inactive_file {65 * GIB}\nfile_dirty {2 * GIB}\nfile_writeback {GIB}\n"
            f"anon {13 * GIB}\nshmem {3 * GIB}\n"
        )
        snapshot = memory_snapshot(self.root)
        self.assertEqual(snapshot["limit_bytes"], 80 * GIB)
        self.assertEqual(snapshot["working_set_bytes"], 33 * GIB)

    def test_oom_and_reserve_guards(self):
        snapshot = memory_snapshot(self.root)
        baseline = dict(snapshot)
        check_pressure(snapshot, self.policy, baseline)
        snapshot["oom_kills"] += 1
        with self.assertRaises(MemoryPressureError):
            check_pressure(snapshot, self.policy, baseline)
        snapshot = dict(baseline, working_set_bytes=81 * GIB)
        with self.assertRaises(MemoryPressureError):
            check_pressure(snapshot, self.policy, baseline)

    def test_missing_finite_limit_is_not_replaced_by_host_ram(self):
        (self.root / "memory.max").write_text("max")
        with self.assertRaisesRegex(RuntimeError, "finite cgroup"):
            memory_snapshot(self.root)

    def test_process_tree_uses_pss_and_handles_exited_workers(self):
        for pid, children, pss in ((1, "2 3", 200), (2, "", 100)):
            directory = self.root / str(pid)
            (directory / "task" / str(pid)).mkdir(parents=True)
            (directory / "task" / str(pid) / "children").write_text(children)
            (directory / "smaps_rollup").write_text(f"Rss: 99999 kB\nPss: {pss} kB\n")
        self.assertEqual(process_tree_pss(1, self.root), 300 * 1024)


if __name__ == "__main__":
    unittest.main()
