"""Curriculum learning callback: auto-upgrades motion set when ep_len threshold is met."""

from __future__ import annotations

import re
from collections import deque
from pathlib import Path

import numpy as np
import yaml
from loguru import logger
from transformers import TrainerCallback


class CurriculumCallback(TrainerCallback):
    """Automatically upgrades the active motion set as training improves.

    Reads a YAML curriculum config with a list of stages.  Each stage defines:
    - ``filter_patterns``: regex patterns for motion keys (empty = all motions)
    - ``upgrade_ep_len``: ep_len sliding-window mean threshold to trigger upgrade
    - ``upgrade_hold_steps``: how many consecutive steps must satisfy the threshold

    The callback checks ep_len every step and calls ``motion_lib.apply_filter()``
    when the upgrade condition is met.  The last stage has no upgrade condition.

    Config is loaded from ``curriculum_cfg_path`` at construction time.
    The callback is a no-op if the path is None or the file does not exist.
    """

    def __init__(self, curriculum_cfg_path: str | None = None):
        super().__init__()
        self.stages: list[dict] = []
        self.current_stage: int = 0
        self._hold_counter: int = 0
        self._ep_len_window: deque = deque(maxlen=50)
        self._initialized: bool = False
        # Expose current stage patterns for other callbacks (e.g. ImResampleCallback blacklist)
        self.current_filter_patterns: list[str] = []
        self.current_exclude_patterns: list[str] = []
        self._resample_cb = None  # set by ImResampleCallback after init

        if curriculum_cfg_path is None:
            return
        cfg_path = Path(curriculum_cfg_path)
        if not cfg_path.exists():
            logger.warning(f"[Curriculum] Config not found: {cfg_path}, curriculum disabled.")
            return
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        self.stages = cfg.get("stages", [])
        if self.stages:
            logger.info(
                f"[Curriculum] Loaded {len(self.stages)} stages from {cfg_path}: "
                + ", ".join(s["name"] for s in self.stages)
            )

    def _get_motion_lib(self, env):
        """Walk env wrapper chain to find motion_lib."""
        try:
            return env.motion_command.motion_lib
        except AttributeError:
            pass
        try:
            return env._motion_lib  # noqa: SLF001
        except AttributeError:
            return None

    def _apply_stage(self, stage_idx: int, env) -> None:
        motion_lib = self._get_motion_lib(env)
        logger.info(f"[Curriculum] _apply_stage({stage_idx}): motion_lib={type(motion_lib).__name__ if motion_lib else None}")
        if motion_lib is None:
            logger.warning("[Curriculum] Could not find motion_lib, skipping stage apply.")
            return
        stage = self.stages[stage_idx]
        patterns = stage.get("filter_patterns", [])
        exclude_patterns = stage.get("exclude_patterns", None) or []
        self.current_filter_patterns = patterns
        self.current_exclude_patterns = exclude_patterns
        # Merge blacklist into exclude patterns if ImResampleCallback is available
        full_exclude = list(exclude_patterns)
        if self._resample_cb is not None and self._resample_cb._blacklist:
            import re as _re  # noqa: PLC0415
            full_exclude += [_re.escape(name) for name in self._resample_cb._blacklist]
        n = motion_lib.apply_filter(patterns, exclude_patterns=full_exclude)
        logger.info(
            f"[Curriculum] ▶ Stage {stage_idx}: '{stage['name']}' — {n} motions active"
            + (f" ({len(self._resample_cb._blacklist)} blacklisted)" if self._resample_cb and self._resample_cb._blacklist else "")
        )
        # Trigger resample so new motion set takes effect immediately
        if hasattr(env, "resample_motion"):
            env.resample_motion()

    def on_train_begin(self, args, state, control, **kwargs):
        """Apply stage 0 at training start."""
        if not self.stages:
            logger.info("[Curriculum] No stages configured, skipping.")
            return
        env = kwargs.get("env")
        logger.info(f"[Curriculum] on_train_begin: env={type(env).__name__ if env else None}, stages={len(self.stages)}")
        if env is None:
            logger.warning("[Curriculum] env is None in on_train_begin, cannot apply stage 0.")
            return
        self._apply_stage(0, env)
        self._initialized = True

    def on_step_end(self, args, state, control, **kwargs):
        if not self.stages or not self._initialized:
            return

        if self.current_stage >= len(self.stages) - 1:
            return  # Already at final stage

        stage = self.stages[self.current_stage]
        upgrade_ep_len = stage.get("upgrade_ep_len")
        upgrade_timeout_ratio = stage.get("upgrade_timeout_ratio")  # 超时完成比例阈值
        upgrade_hold = stage.get("upgrade_hold_steps", 50)
        if upgrade_ep_len is None and upgrade_timeout_ratio is None:
            return

        env = kwargs.get("env")

        # Read ep_len from state.lenbuffer
        lenbuffer = getattr(state, "lenbuffer", None)
        if not lenbuffer:
            for entry in reversed(state.log_history or []):
                if "train/objective/length" in entry:
                    self._ep_len_window.append(entry["train/objective/length"])
                    break
        else:
            if len(lenbuffer) > 0:
                self._ep_len_window.append(float(np.mean(list(lenbuffer)[-20:])))

        if not self._ep_len_window:
            return

        current_mean = float(np.mean(self._ep_len_window))

        # Check upgrade condition: ep_len OR timeout_ratio
        condition_met = False
        if upgrade_ep_len is not None and current_mean >= upgrade_ep_len:
            condition_met = True
        if upgrade_timeout_ratio is not None and env is not None:
            for entry in reversed(state.log_history or []):
                timeout_val = entry.get("Env/Episode_Termination/time_out")
                total_keys = [k for k in entry if k.startswith("Env/Episode_Termination/") and isinstance(entry[k], (int, float))]
                if timeout_val is not None and total_keys:
                    total = sum(entry[k] for k in total_keys)
                    if total > 0:
                        ratio = timeout_val / total
                        if ratio >= upgrade_timeout_ratio:
                            condition_met = True
                break

        # Extra guard: only for timeout_ratio mode — don't upgrade if ep_len is actively declining
        # (not needed for ep_len mode since ep_len itself is the condition)
        if condition_met and upgrade_timeout_ratio is not None and len(self._ep_len_window) >= 20:
            recent = float(np.mean(list(self._ep_len_window)[-10:]))
            older = float(np.mean(list(self._ep_len_window)[-20:-10]))
            if recent < older * 0.90:  # declining more than 10%
                condition_met = False

        if condition_met:
            self._hold_counter += 1
        else:
            # For ep_len mode: only reset if significantly below threshold (allow small fluctuations)
            if upgrade_ep_len is not None and current_mean >= upgrade_ep_len * 0.90:
                pass  # keep counter, small dip is ok
            else:
                self._hold_counter = 0

        if self._hold_counter >= upgrade_hold:
            self._hold_counter = 0
            self._ep_len_window.clear()
            self.current_stage += 1
            env = kwargs.get("env")
            if env is not None:
                stage = self.stages[self.current_stage - 1]
                cond_str = (
                    f"timeout_ratio≥{stage.get('upgrade_timeout_ratio')}"
                    if stage.get("upgrade_timeout_ratio") is not None
                    else f"ep_len={current_mean:.1f}≥{stage.get('upgrade_ep_len')}"
                )
                logger.info(
                    f"[Curriculum] {cond_str} for {upgrade_hold} steps → upgrading to stage {self.current_stage}"
                )
                self._apply_stage(self.current_stage, env)
