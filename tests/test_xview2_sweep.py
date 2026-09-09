"""Offline sweep safeguards: no CUDA initialization or subprocess training."""

from contextlib import ExitStack
import importlib
from pathlib import Path
import shutil
import unittest
from unittest.mock import Mock, patch
import uuid

from bda.xview2 import digest, read_json, write_json

sweep = importlib.import_module("scripts.sweep_xview2_dinov3_upernet")
GPU4_UUID = "GPU-four"


class SweepTests(unittest.TestCase):
    def setUp(self):
        printer = patch("builtins.print")
        printer.start()
        self.addCleanup(printer.stop)
        self.root = (Path("outputs") / (".sweep-tests-" + uuid.uuid4().hex)).resolve()
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root)
        self.gpu_map = {4: GPU4_UUID, 5: "GPU-five", 6: "GPU-six", 7: "GPU-seven"}
        real_lock = sweep.ControllerLock
        locks = patch.object(
            sweep, "ControllerLock",
            side_effect=lambda root, filename=".controller.lock": real_lock(
                self.root if Path(root) == sweep.REPO / "outputs" else root, filename,
            ),
        )
        locks.start()
        self.addCleanup(locks.stop)

    def rows(self):
        return [{**trial, "status": "complete", "epochs_completed": 15,
                 "damaged_f1": 0.5, "best_val_loss": 0.2, "test_damaged_f1": 1 - trial["lr"]}
                for trial in sweep.enumerate_trials()]

    def state(self):
        return {"status": "training", "trials": {
            t["id"]: {**t, "status": "pending", "attempts": []} for t in sweep.enumerate_trials()
        }, "evaluations": {}}

    def test_exact_twelve_recipes(self):
        trials = sweep.enumerate_trials()
        self.assertEqual(len(trials), 12)
        self.assertEqual(len({t["id"] for t in trials}), 12)
        for grouping in ("any", "major", "destroyed"):
            self.assertEqual([t["lr"] for t in trials if t["grouping"] == grouping],
                             [1e-5, 3e-5, 1e-4, 3e-4])
        self.assertTrue(all(t["max_epochs"] == 15 and t["num_workers"] == 4
                            and t["cpu_threads"] == 2 for t in trials))

    def test_gpu_isolation_and_no_fallback(self):
        for gpu, identifier in self.gpu_map.items():
            env = sweep.child_environment(identifier, self.gpu_map)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], identifier)
            self.assertEqual(env["OMP_NUM_THREADS"], "2")
            command, _ = sweep.trial_command(self.root, "data", sweep.enumerate_trials()[0], identifier)
            self.assertEqual(command[command.index("--gpu") + 1], "0")
            self.assertEqual(command[command.index("--require-gpu-uuid") + 1], identifier)
            self.assertEqual(command[command.index("--evaluation-split") + 1], "val")
        with self.assertRaises(ValueError):
            sweep.child_environment("GPU-zero", self.gpu_map)

    def test_expanded_gpu_selection_still_excludes_gpu_one(self):
        expanded = {**self.gpu_map, 0: "GPU-zero", 2: "GPU-two", 3: "GPU-three"}
        for identifier in expanded.values():
            self.assertEqual(sweep.child_environment(identifier, expanded)["CUDA_VISIBLE_DEVICES"], identifier)
        with self.assertRaises(ValueError):
            sweep.child_environment("GPU-one", {**expanded, 1: "GPU-one"})
        with self.assertRaises(ValueError):
            sweep.query_gpus([1, 4])
        with self.assertRaises(ValueError):
            sweep.query_gpus([4, 4])

    def test_resource_migration_preserves_training_protocol(self):
        previous = {
            "schema_version": 1, "trials": sweep.enumerate_trials(),
            "max_concurrent": 4, "gpu_map": {"4": "four"}, "manifest_hash": "same",
            "code_hashes": {"sweep_xview2_dinov3_upernet.py": "old",
                            "train_eval_xview2_dinov3_upernet.py": "training"},
        }
        current = {
            **previous, "schema_version": 2, "max_concurrent": 7,
            "gpu_map": {"4": "four", "0": "zero"}, "gpu_order": [4, 0],
            "memory_policy": {"reserve_bytes": 16, "per_job_bytes": 8},
            "code_hashes": {"sweep_xview2_dinov3_upernet.py": "new",
                            "sweep_resources.py": "resources",
                            "train_eval_xview2_dinov3_upernet.py": "training"},
        }
        sweep.validate_resource_change(previous, current)
        with self.assertRaisesRegex(ValueError, "dataset"):
            sweep.validate_resource_change(previous, dict(current, manifest_hash="changed"))
        with self.assertRaisesRegex(ValueError, "training/data/model"):
            sweep.validate_resource_change(previous, {
                **current, "code_hashes": {**current["code_hashes"],
                                         "train_eval_xview2_dinov3_upernet.py": "changed"},
            })

    def test_query_requires_expected_physical_gpus(self):
        output = "\n".join(f"{i}, {identifier}, 4, 0" for i, identifier in self.gpu_map.items())
        with patch.object(sweep.subprocess, "run", return_value=Mock(stdout=output)):
            found = sweep.query_gpus()
        self.assertEqual({i: g["uuid"] for i, g in found.items()}, self.gpu_map)
        with patch.object(sweep.subprocess, "run", return_value=Mock(stdout="0, GPU-zero, 4, 0")):
            with self.assertRaises(RuntimeError):
                sweep.query_gpus()

    def test_occupied_gpu_rejected(self):
        current = {i: {"uuid": identifier, "memory_mib": 4, "utilization": 0}
                   for i, identifier in self.gpu_map.items()}
        with patch.object(sweep, "query_gpus", return_value=current), \
                patch.object(sweep.subprocess, "run", return_value=Mock(stdout=f"{GPU4_UUID}, 42")):
            self.assertFalse(sweep.available_gpu(GPU4_UUID, self.gpu_map))
            self.assertTrue(sweep.available_gpu("GPU-five", self.gpu_map))
        current[5]["memory_mib"] = 4096
        with patch.object(sweep, "query_gpus", return_value=current), \
                patch.object(sweep.subprocess, "run", return_value=Mock(stdout="")):
            self.assertFalse(sweep.available_gpu("GPU-five", self.gpu_map))

    def test_winners_use_validation_not_test_and_deterministic_ties(self):
        rows = self.rows()
        rows[2]["damaged_f1"] = 0.6
        rows[2]["test_damaged_f1"] = 0
        winners = sweep.select_winners(list(reversed(rows)))
        self.assertEqual(winners["any"]["lr"], 1e-4)
        self.assertEqual(winners["major"]["lr"], 1e-5)
        rows[5]["best_val_loss"] = 0.1
        self.assertEqual(sweep.select_winners(rows)["major"]["lr"], 3e-5)

    def test_incomplete_failed_duplicate_nonfinite_never_select(self):
        with self.assertRaises(ValueError):
            sweep.select_winners(self.rows()[:-1])
        for field, value in (("status", "failed"), ("epochs_completed", 14),
                             ("damaged_f1", float("nan")), ("damaged_f1", None)):
            rows = self.rows()
            rows[-1][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                sweep.select_winners(rows)
        rows = self.rows()
        rows[-1] = rows[0]
        with self.assertRaises(ValueError):
            sweep.select_winners(rows)

    def test_unique_roots_explicit_resume_and_lock(self):
        first = sweep.choose_root(self.root)
        second = sweep.choose_root(self.root)
        self.assertNotEqual(first, second)
        with self.assertRaises(FileExistsError):
            sweep.choose_root(self.root, first)
        self.assertEqual(sweep.choose_root(self.root, first, resume=True), first)
        with self.assertRaises(ValueError):
            sweep.choose_root(self.root, resume=True)
        with sweep.ControllerLock(first):
            with self.assertRaisesRegex(RuntimeError, "controller"):
                with sweep.ControllerLock(first):
                    pass

    def test_live_child_resume_guard_accounts_for_pid_reuse(self):
        state = {"trials": {"one": {"attempts": [{"pid": 123, "process_start_ticks": "456"}]}}}
        with patch.object(sweep, "process_start_ticks", return_value="456"):
            with self.assertRaisesRegex(RuntimeError, "still alive"):
                sweep.reject_live_children(state)
        with patch.object(sweep, "process_start_ticks", return_value="789"):
            sweep.reject_live_children(state)

    def test_failure_bounded_to_four_owned_children_and_all_trials_recorded(self):
        state = self.state()
        processes = []
        def spawn(*args, **kwargs):
            process = Mock(pid=1000 + len(processes), returncode=1)
            process.poll.return_value = 1
            processes.append(process)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], list(self.gpu_map.values())[len(processes) - 1])
            self.assertTrue(kwargs["start_new_session"])
            return process
        with ExitStack() as stack:
            stack.enter_context(patch.object(sweep, "available_gpu", return_value=True))
            stack.enter_context(patch.object(sweep.subprocess, "Popen", side_effect=spawn))
            stack.enter_context(patch.object(sweep, "process_start_ticks", return_value="42"))
            cleanup = stack.enter_context(patch.object(
                sweep, "stop_owned_children",
                side_effect=lambda active: [item["log"].close() for item in active.values()],
            ))
            with self.assertRaisesRegex(RuntimeError, "failed"):
                sweep.execute_jobs(self.root, state, "trials", sweep.enumerate_trials(),
                                   self.gpu_map, "data", 4, Mock(side_effect=FileNotFoundError),
                                   gpu_wait_seconds=0)
        self.assertEqual(len(processes), 4)
        self.assertEqual(len(cleanup.call_args.args[0]), 3)
        saved = read_json(self.root / "state.json")
        self.assertEqual(len(saved["trials"]), 12)
        self.assertEqual(sum(r["status"] == "failed" for r in saved["trials"].values()), 1)
        self.assertEqual(sum(r["status"] == "pending" for r in saved["trials"].values()), 8)

    def test_all_twelve_jobs_finish_in_bounded_waves(self):
        state = self.state()
        finished = set()
        launched = []
        def validate(job):
            if job["id"] not in finished:
                raise FileNotFoundError(job["id"])
        def spawn(command, **kwargs):
            grouping = command[command.index("--grouping") + 1]
            lr = float(command[command.index("--lr") + 1])
            identifier = f"{grouping}/lr_{lr:.0e}"
            finished.add(identifier)
            launched.append((identifier, kwargs["env"]["CUDA_VISIBLE_DEVICES"]))
            process = Mock(pid=1000 + len(launched), returncode=0)
            process.poll.return_value = 0
            return process
        with patch.object(sweep, "available_gpu", return_value=True), \
                patch.object(sweep.subprocess, "Popen", side_effect=spawn), \
                patch.object(sweep, "process_start_ticks", return_value="42"), \
                patch.object(sweep.time, "sleep"):
            sweep.execute_jobs(self.root, state, "trials", sweep.enumerate_trials(),
                               self.gpu_map, "data", 4, validate)
        self.assertEqual(len(launched), 12)
        self.assertEqual([gpu for _, gpu in launched], list(self.gpu_map.values()) * 3)
        self.assertTrue(all(r["status"] == "complete" for r in state["trials"].values()))
        self.assertTrue(all(len(r["attempts"]) == 1 for r in state["trials"].values()))

    def test_unavailable_gpus_leave_all_pending_without_spawning(self):
        state = self.state()
        with patch.object(sweep, "available_gpu", return_value=False), \
                patch.object(sweep.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "No allowed GPU"):
                sweep.execute_jobs(self.root, state, "trials", sweep.enumerate_trials(),
                                   self.gpu_map, "data", 4, Mock(side_effect=FileNotFoundError),
                                   gpu_wait_seconds=0)
        spawn.assert_not_called()
        self.assertTrue(all(r["status"] == "pending" for r in state["trials"].values()))

    def test_memory_admission_refuses_free_gpus_without_headroom(self):
        state = self.state()
        snapshot = {
            "limit_bytes": 96 * sweep.GIB, "current_bytes": 95 * sweep.GIB,
            "working_set_bytes": 75 * sweep.GIB, "oom_kills": 3, "oom_group_kills": 0,
        }
        policy = {"per_job_bytes": 8 * sweep.GIB, "reserve_bytes": 16 * sweep.GIB}
        with patch.object(sweep, "available_gpu", return_value=True), \
                patch.object(sweep, "memory_snapshot", return_value=snapshot), \
                patch.object(sweep.subprocess, "Popen") as spawn:
            with self.assertRaisesRegex(RuntimeError, "Cgroup memory admission"):
                sweep.execute_jobs(
                    self.root, state, "trials", sweep.enumerate_trials(),
                    self.gpu_map, "data", 4, Mock(side_effect=FileNotFoundError),
                    gpu_wait_seconds=0, memory_policy=policy,
                )
        spawn.assert_not_called()
        self.assertFalse(state["memory_admission"]["allowed"])
        self.assertTrue(all(r["status"] == "pending" for r in state["trials"].values()))

    def test_completed_jobs_reused_only_after_validation(self):
        state = self.state()
        for row in state["trials"].values():
            row["status"] = "complete"
        with patch.object(sweep.subprocess, "Popen") as spawn:
            validate = Mock()
            sweep.execute_jobs(self.root, state, "trials", sweep.enumerate_trials(),
                               self.gpu_map, "data", 4, validate)
        self.assertEqual(validate.call_count, 12)
        spawn.assert_not_called()
        with self.assertRaisesRegex(ValueError, "mismatch"):
            sweep.execute_jobs(self.root, state, "trials", sweep.enumerate_trials(),
                               self.gpu_map, "data", 4, Mock(side_effect=ValueError("mismatch")))

    def test_eval_command_only_uses_explicit_winner_test(self):
        trial = sweep.enumerate_trials()[0]
        command, output = sweep.trial_command(self.root, "data", trial, GPU4_UUID,
                                              checkpoint="/selected/best.ckpt")
        self.assertIn("--eval-only", command)
        self.assertEqual(command[command.index("--checkpoint") + 1], "/selected/best.ckpt")
        self.assertEqual(command[command.index("--evaluation-split") + 1], "test")
        self.assertEqual(output, self.root / "winners/any/evaluation")

    def test_invalid_eval_never_reused(self):
        path = self.root / "test_metrics.json"
        result = {"split": "test", "n_test_images": 2, "grouping": "any", "checkpoint": "best"}
        write_json(path, result)
        with self.assertRaisesRegex(ValueError, "Invalid"):
            sweep.validate_test(path, {"grouping": "any", "checkpoint": "best"}, "hash")

    def test_complete_trial_validates_configuration_manifest_steps_and_checkpoint(self):
        trial = sweep.enumerate_trials()[0]
        directory = self.root / trial["id"]
        checkpoints = directory / "checkpoints"
        checkpoints.mkdir(parents=True)
        best = checkpoints / "best.ckpt"
        best.touch()
        (checkpoints / "last.ckpt").touch()
        config = {**trial, "manifest_hash": "hash", "evaluation_split": "val"}
        state = {
            "status": "complete", "epochs_completed": 15, "global_step": 33330,
            "counts": {"train": 8889, "val": 279, "test": 933},
            "config_hash": digest(config), "selected_checkpoint": str(best), "selected_epoch": 12,
            "best_val_loss": 0.2, "attempts": [{"fit_seconds": 100}], "total_seconds": 120,
        }
        metrics = {"split": "val", "n_val_images": 279, "manifest_hash": "hash",
                   "grouping": "any", "checkpoint": str(best), "damaged_f1": 0.5}
        write_json(directory / "config.json", config)
        write_json(directory / "training_state.json", state)
        write_json(directory / "val_metrics.json", metrics)
        row = sweep.validate_trial(self.root, trial, "hash")
        self.assertEqual(row["global_step"], 33330)
        self.assertEqual(row["checkpoint"], str(best))
        with self.assertRaisesRegex(ValueError, "manifest"):
            sweep.validate_trial(self.root, trial, "changed")
        state["global_step"] = 33329
        write_json(directory / "training_state.json", state)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            sweep.validate_trial(self.root, trial, "hash")

    def test_failure_summary_accounts_for_every_trial_without_winners(self):
        state = self.state()
        state["status"] = "failed"
        state["trials"][sweep.enumerate_trials()[0]["id"]]["status"] = "failed"
        sweep.write_summary(self.root, state, "hash", {})
        result = read_json(self.root / "summary.json")
        self.assertEqual(len(result["trials"]), 12)
        self.assertEqual(result["winners"], {})
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len((self.root / "validation_sweep.csv").read_text().splitlines()), 13)

    def test_cleanup_signals_only_own_live_child_groups(self):
        live = Mock(pid=1001)
        live.poll.return_value = None
        live.wait.side_effect = [sweep.subprocess.TimeoutExpired("owned", 30), 0]
        finished = Mock(pid=1002)
        finished.poll.return_value = 0
        active = {"one": {"process": live, "log": Mock()},
                  "two": {"process": finished, "log": Mock()}}
        with patch.object(sweep.os, "killpg") as signal_group:
            sweep.stop_owned_children(active)
        self.assertEqual([call.args for call in signal_group.call_args_list],
                         [(1001, sweep.signal.SIGTERM), (1001, sweep.signal.SIGKILL)])
        for item in active.values():
            item["log"].close.assert_called_once()

    def test_prepare_only_resume_and_configuration_mismatch(self):
        manifest = {
            "include_tier3": True, "seed": 0, "val_fraction": 0.1,
            "counts": {"train": {"total": 8889, "by_source": {"train": 2520, "tier3": 6369}},
                       "val": {"total": 279}, "test": {"total": 933}},
        }
        manifest["manifest_hash"] = digest(manifest)
        weights = {"manifest_hash": manifest["manifest_hash"], "groupings": {
            grouping: {"weights": [0.1, 1.0, 1.9], "pixel_counts": [100, 10, 1]}
            for grouping in ("any", "major", "destroyed")
        }}
        baselines = self.root / "baselines"
        for grouping in ("any", "major", "destroyed"):
            write_json(baselines / f"xview2_dinov3_upernet_{grouping}/test_metrics.json",
                       {"grouping": grouping, "n_test_images": 933})
        run = self.root / "run"
        args = ["sweep", "--run-dir", str(run), "--baseline-root", str(baselines), "--prepare-only"]
        with ExitStack() as stack:
            stack.enter_context(patch.object(sweep, "build_manifest", return_value=manifest))
            stack.enter_context(patch.object(sweep, "validate_manifest", return_value={
                "status": "validated", "manifest_hash": manifest["manifest_hash"],
                "image_size": [1024, 1024], "target_codes": [0, 1, 2, 3, 4],
                "images_validated": 10101,
            }))
            stack.enter_context(patch.object(sweep, "shared_class_weights", return_value=weights))
            all_gpus = {**self.gpu_map, 0: "GPU-zero", 2: "GPU-two", 3: "GPU-three"}
            stack.enter_context(patch.object(
                sweep, "query_gpus",
                side_effect=lambda selected=sweep.DEFAULT_GPUS: {
                    i: {"uuid": all_gpus[i]} for i in selected
                },
            ))
            execute = stack.enter_context(patch.object(sweep, "execute_jobs"))
            stack.enter_context(patch("builtins.print"))
            with patch.object(sweep.sys, "argv", args):
                sweep.main()
            self.assertEqual(read_json(run / "state.json")["status"], "prepared")
            self.assertEqual(len(read_json(run / "state.json")["trials"]), 12)
            with patch.object(sweep.sys, "argv", args + ["--resume"]):
                sweep.main()
            self.assertEqual(len(read_json(run / "state.json")["controller_attempts"]), 2)
            before_migration = read_json(run / "state.json")
            before_migration["error"] = "planned scheduler handoff"
            before_migration["status"] = "interrupted"
            write_json(run / "state.json", before_migration)
            migration_args = args + [
                "--resume", "--reconfigure-resources", "--reuse-preflight",
                "--gpus", "4", "5", "6", "7", "0", "2", "3", "--max-concurrent", "7",
            ]
            with patch.object(sweep.sys, "argv", migration_args):
                sweep.main()
            expanded = read_json(run / "config.json")
            self.assertEqual(expanded["gpu_order"], [4, 5, 6, 7, 0, 2, 3])
            self.assertEqual(expanded["max_concurrent"], 7)
            self.assertEqual(len(expanded["gpu_map"]), 7)
            self.assertNotIn("error", read_json(run / "state.json"))
            self.assertEqual(read_json(run / "state.json")["controller_attempts"][-2]["error"],
                             "planned scheduler handoff")
            self.assertTrue((run / "config_history" / f"{before_migration['config_hash']}.json").exists())
            # Recover an interruption between the config and state atomic writes.
            write_json(run / "state.json", before_migration)
            with patch.object(sweep.sys, "argv", migration_args):
                sweep.main()
            self.assertEqual(read_json(run / "state.json")["config_hash"], digest(expanded))
            # Resume inherits resource settings instead of reverting to four GPUs.
            with patch.object(sweep.sys, "argv", args + ["--resume"]):
                sweep.main()
            self.assertEqual(read_json(run / "config.json"), expanded)
            config = read_json(run / "config.json")
            config["max_concurrent"] = 1
            write_json(run / "config.json", config)
            with patch.object(sweep.sys, "argv", args + ["--resume"]):
                with self.assertRaisesRegex(ValueError, "mismatch"):
                    sweep.main()
            execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
