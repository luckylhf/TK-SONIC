import csv
from pathlib import Path

import torch
from loguru import logger
from transformers import TrainerCallback

# Compact the file when it exceeds this many rows (header excluded)
_COMPACT_THRESHOLD = 5000


class ImResampleCallback(TrainerCallback):
    """Callback to resample motion during training. Supports multigpu."""

    def __init__(
        self,
        motion_resample_frequency,
        skip_resample_frequency=None,
        blacklist_min_failures: int = 100,
    ):
        super().__init__()
        self.motion_resample_frequency = motion_resample_frequency
        self.skip_resample_frequency = skip_resample_frequency
        self.blacklist_min_failures = blacklist_min_failures
        self._failure_log_path: Path | None = None
        self._blacklist_path: Path | None = None
        self._blacklist: set[str] = set()
        self._curriculum_cb = None
        self._blacklist_updated_this_step = False

    def on_train_begin(self, args, state, control, **kwargs):
        output_dir = Path(args.output_dir) if hasattr(args, "output_dir") else Path(".")
        self._failure_log_path = output_dir.parent / "motion_failure_stats.csv"
        self._blacklist_path = output_dir.parent / "motion_blacklist.txt"
        # Load existing blacklist from previous runs
        if self._blacklist_path.exists():
            with open(self._blacklist_path) as f:
                self._blacklist = {line.strip() for line in f if line.strip() and not line.startswith("#")}
            logger.info(f"[Resample] Loaded {len(self._blacklist)} blacklisted motions from {self._blacklist_path}")
        # Find CurriculumCallback to read current stage patterns
        self._curriculum_cb = None
        from gear_sonic.trl.callbacks.curriculum_callback import CurriculumCallback  # noqa: PLC0415
        cb_handler = kwargs.get("callback_handler")
        if cb_handler is not None:
            search_list = cb_handler.callbacks
        elif hasattr(kwargs.get("model"), "callback_handler"):
            search_list = kwargs["model"].callback_handler.callbacks
        else:
            search_list = []
        for cb in search_list:
            if isinstance(cb, CurriculumCallback):
                self._curriculum_cb = cb
                cb._resample_cb = self  # back-reference for blacklist in _apply_stage
                break
        if self._curriculum_cb is None:
            logger.warning("[Resample] CurriculumCallback not found — blacklist will use no stage filter")

        # On startup: scan failure stats CSV and auto-blacklist motions that already exceed threshold
        self._apply_blacklist_from_stats()

    def _apply_blacklist_from_stats(self) -> None:
        """On startup: scan failure stats CSV and auto-blacklist motions exceeding threshold."""
        if self._failure_log_path is None or not self._failure_log_path.exists():
            return
        import re as _re  # noqa: PLC0415
        # Aggregate total_failures per motion from all rows in the CSV
        cumulative: dict[str, float] = {}
        with open(self._failure_log_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("motion_name", "")
                if not name:
                    continue
                if name not in cumulative:
                    cumulative[name] = 0.0
                cumulative[name] += float(row.get("batch_failures", 0))

        newly_blacklisted = []
        for name, total_fails in cumulative.items():
            if name not in self._blacklist and total_fails >= self.blacklist_min_failures:
                self._blacklist.add(name)
                newly_blacklisted.append(name)

        if newly_blacklisted:
            if self._blacklist_path is not None:
                with open(self._blacklist_path, "w") as f:
                    f.write(f"# Motion blacklist — total_failures >= {self.blacklist_min_failures}\n")
                    for name in sorted(self._blacklist):
                        f.write(f"{name}\n")
            logger.info(f"[Resample] Startup blacklist: added {len(newly_blacklisted)} motions "
                        f"(total: {len(self._blacklist)}) from failure stats")

    def _apply_blacklist_to_motion_lib(self, env) -> None:
        """Apply current blacklist to motion_lib via apply_filter."""
        import re as _re  # noqa: PLC0415
        try:
            motion_lib = env.motion_command.motion_lib
        except AttributeError:
            try:
                motion_lib = env._motion_lib  # noqa: SLF001
            except AttributeError:
                return
        if motion_lib is None or not hasattr(motion_lib, "apply_filter"):
            return
        filter_patterns = self._curriculum_cb.current_filter_patterns if self._curriculum_cb else []
        base_exclude = self._curriculum_cb.current_exclude_patterns if self._curriculum_cb else []
        blacklist_patterns = [_re.escape(name) for name in self._blacklist]
        exclude = blacklist_patterns + base_exclude
        n = motion_lib.apply_filter(filter_patterns, exclude_patterns=exclude)
        logger.info(f"[Resample] Blacklist applied: {n} motions active ({len(self._blacklist)} excluded)")

    def on_step_end(self, args, state, control, **kwargs):
        self.env = kwargs.get("env")
        self.accelerator = kwargs.get("accelerator")
        self.device = self.accelerator.device

        # On first step: apply blacklist to motion_lib (env not available in on_train_begin)
        if state.global_step == 0 and self._blacklist:
            self._apply_blacklist_to_motion_lib(self.env)

        should_resample = (state.global_step + 1) % self.motion_resample_frequency == 0
        should_skip = (
            self.skip_resample_frequency is not None
            and (state.global_step + 1) % self.skip_resample_frequency == 0
        )

        if should_resample and not should_skip:
            self._dump_failure_stats(state.global_step + 1)

            # Sync blacklist update across all processes so apply_filter + resample happen together.
            # Main process detects new blacklisted motions; non-main process has empty cumulative.
            # Without sync, main calls resample_motion() but non-main doesn't → reset_all() deadlock.
            if self.accelerator is not None and self.accelerator.num_processes > 1:
                flag = torch.tensor(
                    [1 if self._blacklist_updated_this_step else 0], device=self.device
                )
                torch.distributed.broadcast(flag, src=0)
                if flag.item() and not self._blacklist_updated_this_step:
                    # Non-main process: reload blacklist from file (main just wrote it) then apply
                    if self._blacklist_path is not None and self._blacklist_path.exists():
                        with open(self._blacklist_path) as f:
                            self._blacklist = {
                                line.strip() for line in f
                                if line.strip() and not line.startswith("#")
                            }
                    self._apply_blacklist_to_motion_lib(self.env)

            self.env.resample_motion()
            self._blacklist_updated_this_step = False

    def _get_motion_lib(self):
        env = self.env
        try:
            return env.motion_command.motion_lib
        except AttributeError:
            pass
        try:
            return env._motion_lib  # noqa: SLF001
        except AttributeError:
            return None

    @staticmethod
    def _load_cumulative(path: Path) -> tuple[dict[str, dict], int]:
        """Read CSV and return (cumulative totals per motion, row count)."""
        cumulative: dict[str, dict] = {}
        row_count = 0
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row_count += 1
                name = row["motion_name"]
                if name not in cumulative:
                    cumulative[name] = {"total_failures": 0.0, "total_episodes": 0.0, "last_step": 0}
                cumulative[name]["total_failures"] += float(row.get("batch_failures", 0))
                cumulative[name]["total_episodes"] += float(row.get("batch_episodes", 0))
                cumulative[name]["last_step"] = max(
                    cumulative[name]["last_step"], int(row.get("step", 0))
                )
        return cumulative, row_count

    @staticmethod
    def _write_compacted(path: Path, cumulative: dict[str, dict]) -> None:
        """Overwrite file with one merged row per motion, sorted by total failure rate."""
        rows = []
        for name, c in cumulative.items():
            eps = c["total_episodes"]
            fr = c["total_failures"] / eps if eps > 0 else 0.0
            rows.append((name, fr, c["total_failures"], eps, c["last_step"]))
        rows.sort(key=lambda x: x[1], reverse=True)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "motion_name", "batch_failure_rate", "batch_failures",
                             "batch_episodes", "total_failure_rate", "total_failures", "total_episodes"])
            for name, fr, fails, eps, last_step in rows:
                writer.writerow([last_step, name, f"{fr:.4f}", f"{fails:.0f}",
                                 f"{eps:.0f}", f"{fr:.4f}", f"{fails:.0f}", f"{eps:.0f}"])

    def _dump_failure_stats(self, step: int, top_n: int = 100, min_failures: int = 5) -> None:
        """Append per-motion failure stats; compact file when it grows too large."""
        if self._failure_log_path is None:
            return
        motion_lib = self._get_motion_lib()
        if motion_lib is None or not hasattr(motion_lib, "adp_samp_num_failures"):
            return

        num_failures = motion_lib.adp_samp_num_failures.cpu().numpy()
        num_episodes = motion_lib.adp_samp_num_episodes.cpu().numpy()
        bins = motion_lib.adp_samp_bins.cpu().numpy()
        keys = getattr(motion_lib, "_motion_data_keys", None)
        if keys is None:
            keys = getattr(motion_lib, "curr_motion_keys", None)
        if keys is None:
            return

        # Aggregate per motion across bins
        motion_stats: dict[int, dict] = {}
        for bin_idx, (motion_id, _, _) in enumerate(bins):
            mid = int(motion_id)
            fails = float(num_failures[bin_idx])
            eps = float(num_episodes[bin_idx])
            if mid not in motion_stats:
                motion_stats[mid] = {"failures": fails, "episodes": eps}
            else:
                motion_stats[mid]["failures"] += fails
                motion_stats[mid]["episodes"] += eps

        # Load existing rows and check if compaction is needed (main process only)
        cumulative: dict[str, dict] = {}
        row_count = 0
        is_main = not hasattr(self, "accelerator") or self.accelerator is None or self.accelerator.is_main_process
        if is_main and self._failure_log_path.exists():
            cumulative, row_count = self._load_cumulative(self._failure_log_path)
            if row_count >= _COMPACT_THRESHOLD:
                self._write_compacted(self._failure_log_path, cumulative)
                row_count = len(cumulative)
                logger.info(f"[Resample] Compacted failure stats to {row_count} rows")

        # Build batch rows (only motions with enough failures)
        batch_rows = []
        for mid, stats in motion_stats.items():
            name = keys[mid] if mid < len(keys) else f"motion_{mid}"
            if stats["failures"] < min_failures:
                continue
            if name not in cumulative:
                cumulative[name] = {"total_failures": 0.0, "total_episodes": 0.0, "last_step": 0}
            cumulative[name]["total_failures"] += stats["failures"]
            cumulative[name]["total_episodes"] += stats["episodes"]
            cumulative[name]["last_step"] = step
            fr = stats["failures"] / stats["episodes"] if stats["episodes"] > 0 else 0.0
            total_fr = (cumulative[name]["total_failures"] / cumulative[name]["total_episodes"]
                        if cumulative[name]["total_episodes"] > 0 else 0.0)
            batch_rows.append((name, fr, stats["failures"], stats["episodes"],
                               total_fr, cumulative[name]["total_failures"], cumulative[name]["total_episodes"]))

        if not batch_rows:
            return

        batch_rows.sort(key=lambda x: x[1], reverse=True)

        # Only main process writes files to avoid race conditions in multi-GPU training
        is_main = not hasattr(self, "accelerator") or self.accelerator is None or self.accelerator.is_main_process

        if is_main:
            write_header = not self._failure_log_path.exists()
            with open(self._failure_log_path, "a", newline="") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["step", "motion_name", "batch_failure_rate", "batch_failures",
                                     "batch_episodes", "total_failure_rate", "total_failures", "total_episodes"])
                for name, fr, fails, eps, total_fr, total_fails, total_eps in batch_rows[:top_n]:
                    writer.writerow([step, name, f"{fr:.4f}", f"{fails:.0f}", f"{eps:.0f}",
                                     f"{total_fr:.4f}", f"{total_fails:.0f}", f"{total_eps:.0f}"])

            logger.info(f"[Resample] Failure stats: {len(batch_rows)} motions appended (file ~{row_count + len(batch_rows)} rows)")

        # --- Blacklist: add motions whose cumulative failures exceed threshold ---
        newly_blacklisted = []
        for name, c in cumulative.items():
            if name not in self._blacklist and c["total_failures"] >= self.blacklist_min_failures:
                self._blacklist.add(name)
                newly_blacklisted.append(name)

        if newly_blacklisted and is_main:
            # Persist blacklist to file
            with open(self._blacklist_path, "w") as f:
                f.write(f"# Motion blacklist — total_failures >= {self.blacklist_min_failures}\n")
                for name in sorted(self._blacklist):
                    f.write(f"{name}\n")
            logger.info(f"[Resample] Blacklisted {len(newly_blacklisted)} new motions "
                        f"(total: {len(self._blacklist)}) → {self._blacklist_path}")

        if newly_blacklisted:
            # Re-apply current filter with updated blacklist exclusions
            self._apply_blacklist_to_motion_lib(self.env)
            self._blacklist_updated_this_step = True  # signal to skip resample_motion

