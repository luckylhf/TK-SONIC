#!/usr/bin/env python3
"""Training log analyzer for SONIC whole-body controller.

Usage:
    python analyze_training_log.py                    # latest run under logs_rl/<task>
    python analyze_training_log.py logs_rl/TRL_Tienkung2Pro_Track/RUN  # specific run dir or event file
    python analyze_training_log.py --task TRL_G1_Track    # different task
    python analyze_training_log.py --watch 30            # live refresh every N seconds
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
EVENT_GLOB = "events.out.tfevents.*"


# ---------------------------------------------------------------------------
# Tag groups — ordered for display
# ---------------------------------------------------------------------------

GROUPS: list[tuple[str, list[str]]] = [
    ("training", [
        "train/objective/length",
        "train/objective/rewards",
        "train/Policy/mean_noise_std",
        "train/objective/entropy",
        "train/curriculum/stage",
    ]),
    ("tracking", [
        "train/Env/Metrics/motion/error_anchor_pos",
        "train/Env/Metrics/motion/error_anchor_rot",
        "train/Env/Metrics/motion/error_anchor_lin_vel",
        "train/Env/Metrics/motion/error_anchor_ang_vel",
        "train/Env/Metrics/motion/error_body_pos",
        "train/Env/Metrics/motion/error_body_rot",
        "train/Env/Metrics/motion/error_body_lin_vel",
        "train/Env/Metrics/motion/error_body_ang_vel",
        "train/Env/Metrics/motion/error_joint_pos",
        "train/Env/Metrics/motion/error_joint_vel",
    ]),
    ("tracking_rewards", [
        "train/Env/Episode_Reward/tracking_vr_5point_local",
        "train/Env/Episode_Reward/tracking_anchor_pos",
        "train/Env/Episode_Reward/tracking_anchor_ori",
        "train/Env/Episode_Reward/tracking_relative_body_pos",
        "train/Env/Episode_Reward/tracking_relative_body_ori",
        "train/Env/Episode_Reward/tracking_body_linvel",
        "train/Env/Episode_Reward/tracking_body_angvel",
        "train/Env/Episode_Reward/tracking_joint_pos_waist",
        "train/Env/Episode_Reward/tracking_joint_pos_hip",
        "train/Env/Episode_Reward/tracking_joint_pos_hip_yaw",
        "train/Env/Episode_Reward/tracking_joint_pos_knee",
        "train/Env/Episode_Reward/tracking_joint_pos_ankle",
        "train/Env/Episode_Reward/tracking_joint_pos_shoulder",
        "train/Env/Episode_Reward/tracking_joint_pos_elbow",
        "train/Env/Episode_Reward/tracking_joint_vel_hip",
        "train/Env/Episode_Reward/tracking_joint_vel_hip_yaw",
        "train/Env/Episode_Reward/tracking_joint_vel_knee",
        "train/Env/Episode_Reward/tracking_joint_vel_ankle",
        "train/Env/Episode_Reward/tracking_joint_vel_shoulder",
        "train/Env/Episode_Reward/tracking_joint_vel_elbow",
        "train/Env/Episode_Reward/tracking_joint_vel_wrist",
        "train/Env/Episode_Reward/tracking_body_pos_wrist",
        "train/Env/Episode_Reward/tracking_body_pos_foot",
        "train/Env/Episode_Reward/anchor_planar_drift",
        "train/Env/Episode_Reward/anchor_planar_drift_ungated",
        "train/Env/Episode_Reward/anchor_static_planar_vel",
        "train/Env/Episode_Reward/anti_shake_ang_vel",
        "train/Env/Episode_Reward/feet_acc",
    ]),
    ("termination", [
        "train/Env/Episode_Termination/time_out",
        "train/Env/Episode_Termination/anchor_pos",
        "train/Env/Episode_Termination/anchor_ori_full",
        "train/Env/Episode_Termination/ee_body_pos",
        "train/Env/Episode_Termination/foot_pos_xyz",
        "train/Env/Episode_Termination/foot_pos_local",
    ]),
    ("losses", [
        "train/loss/policy_avg",
        "train/loss/value_avg",
        "train/loss/entropy_avg",
        "train/loss/weighted_ppo_loss_avg",
        "train/loss/total_aux_loss_avg",
        "train/loss/total_aux_loss_unscaled_avg",
        "train/loss/aux_g1_recon_avg",
        "train/loss/aux_g1_smpl_latent_avg",
        "train/loss/aux_g1_teleop_latent_avg",
        "train/loss/aux_teleop_smpl_latent_avg",
        "train/loss/aux_reencoded_smpl_g1_latent_avg",
    ]),
    ("policy", [
        "train/policy/approxkl_avg",
        "train/policy/clipfrac_avg",
        "train/val/advantage_mean",
        "train/val/advantage_std",
        "train/val/clipfrac_avg",
        "train/val/ratio",
        "train/val/ratio_var",
    ]),
    ("adp_samp", [
        "train/Env/adp_samp/num_episodes_mean",
        "train/Env/adp_samp/num_failures_mean",
        "train/Env/adp_samp/failure_rate_mean",
        "train/Env/adp_samp/effective_num_bins",
        "train/Env/adp_samp/prob_max_over_uniform",
        "train/Env/adp_samp/episodes_max_over_mean",
    ]),
    ("perf", [
        "train/fps",
        "train/eps",
        "train/collection_time",
        "train/learn_time",
        "train/tot_time",
        "train/lr",
    ]),
]

# Short display names (strip group prefix for readability)
SHORT: dict[str, str] = {
    "train/objective/length":                         "回合长度",      # ep_len
    "train/objective/rewards":                        "总奖励",        # reward
    "train/objective/entropy":                        "策略熵",        # entropy
    "train/Policy/mean_noise_std":                    "动作噪声",      # noise_std
    "train/curriculum/stage":                         "课程阶段",      # 0=站立 1=行走 2=跑步 3=全量
    # --- 跟踪误差（越低越好）---
    "train/Env/Metrics/motion/error_anchor_pos":      "根位置误差",    # anc_pos 米
    "train/Env/Metrics/motion/error_anchor_rot":      "根朝向误差",    # anc_rot 弧度
    "train/Env/Metrics/motion/error_anchor_lin_vel":  "根线速误差",    # anc_linv m/s
    "train/Env/Metrics/motion/error_anchor_ang_vel":  "根角速误差",    # anc_angv rad/s
    "train/Env/Metrics/motion/error_body_pos":        "体位置误差",    # body_pos 米
    "train/Env/Metrics/motion/error_body_rot":        "体朝向误差",    # body_rot 弧度
    "train/Env/Metrics/motion/error_body_lin_vel":    "体线速误差",    # body_linv m/s
    "train/Env/Metrics/motion/error_body_ang_vel":    "体角速误差",    # body_angv rad/s
    "train/Env/Metrics/motion/error_joint_pos":       "关节角误差",    # jnt_pos 弧度
    "train/Env/Metrics/motion/error_joint_vel":       "关节速误差",    # jnt_vel rad/s
    # --- 跟踪奖励分项 ---
    "train/Env/Episode_Reward/tracking_vr_5point_local": "5点跟踪",   # vr_5pt 腰+双手+双脚
    "train/Env/Episode_Reward/tracking_anchor_pos":      "锚点位置",   # anchor pos
    "train/Env/Episode_Reward/tracking_anchor_ori":      "锚点朝向",   # anchor ori
    "train/Env/Episode_Reward/tracking_relative_body_pos": "体位置",   # body pos
    "train/Env/Episode_Reward/tracking_relative_body_ori": "体朝向",   # body ori
    "train/Env/Episode_Reward/tracking_body_linvel":     "体线速",     # body linvel
    "train/Env/Episode_Reward/tracking_body_angvel":     "体角速",     # body angvel
    "train/Env/Episode_Reward/tracking_joint_pos_waist": "腰关节位",   # waist pos
    "train/Env/Episode_Reward/tracking_joint_pos_hip":   "髋滚俯位",   # hip roll/pitch pos
    "train/Env/Episode_Reward/tracking_joint_pos_hip_yaw": "髋偏位",   # hip yaw pos
    "train/Env/Episode_Reward/tracking_joint_pos_knee":  "膝关节位",   # knee pos
    "train/Env/Episode_Reward/tracking_joint_pos_ankle": "踝关节位",   # ankle pos
    "train/Env/Episode_Reward/tracking_joint_pos_shoulder": "肩关节位", # shoulder pos
    "train/Env/Episode_Reward/tracking_joint_pos_elbow": "肘关节位",   # elbow pos
    "train/Env/Episode_Reward/tracking_joint_vel_hip":   "髋滚俯速",   # hip roll/pitch vel
    "train/Env/Episode_Reward/tracking_joint_vel_hip_yaw": "髋偏速",   # hip yaw vel
    "train/Env/Episode_Reward/tracking_joint_vel_knee":  "膝关节速",   # knee vel
    "train/Env/Episode_Reward/tracking_joint_vel_ankle": "踝关节速",   # ankle vel
    "train/Env/Episode_Reward/tracking_joint_vel_shoulder": "肩关节速", # shoulder vel
    "train/Env/Episode_Reward/tracking_joint_vel_elbow": "肘关节速",   # elbow vel
    "train/Env/Episode_Reward/tracking_joint_vel_wrist": "腕关节速",   # wrist vel
    "train/Env/Episode_Reward/tracking_body_pos_wrist":  "腕末端位",   # wrist body pos
    "train/Env/Episode_Reward/tracking_body_pos_foot":   "脚末端位",   # foot body pos
    "train/Env/Episode_Reward/anchor_planar_drift":      "锚点漂移",   # gated planar drift penalty
    "train/Env/Episode_Reward/anchor_planar_drift_ungated": "漂移无门",  # ungated planar drift penalty
    "train/Env/Episode_Reward/anchor_static_planar_vel": "锚点平速",   # static planar vel penalty
    "train/Env/Episode_Reward/anti_shake_ang_vel":    "抗抖动",        # shake 末端角速度惩罚
    "train/Env/Episode_Reward/feet_acc":              "脚踝冲击",      # feet_acc 加速度惩罚
    # --- 终止原因（每步终止的env数量）---
    "train/Env/Episode_Termination/time_out":         "超时完成",      # t_timeout 正常完成
    "train/Env/Episode_Termination/anchor_pos":       "根位置漂",      # t_ancpos 位置漂移过大
    "train/Env/Episode_Termination/anchor_ori_full":  "根朝向偏",      # t_ancrot 朝向偏差过大
    "train/Env/Episode_Termination/ee_body_pos":      "末端超限",      # t_ee_pos 末端位置误差
    "train/Env/Episode_Termination/foot_pos_xyz":     "脚部超限",      # t_foot 世界坐标（旧）
    "train/Env/Episode_Termination/foot_pos_local":   "脚局部超限",    # t_foot 根相对坐标（新）
    # --- 损失函数 ---
    "train/loss/policy_avg":                          "策略损失",      # pol
    "train/loss/value_avg":                           "价值损失",      # val
    "train/loss/entropy_avg":                         "熵损失",        # ent
    "train/loss/weighted_ppo_loss_avg":               "PPO损失",       # ppo
    "train/loss/total_aux_loss_avg":                  "辅助损失",      # aux 含系数
    "train/loss/total_aux_loss_unscaled_avg":         "辅助原始",      # aux_raw 不含系数
    "train/loss/aux_g1_recon_avg":                    "G1重建",        # g1_rec token→动作→重建
    "train/loss/aux_g1_smpl_latent_avg":              "G1↔SMPL",      # g1→smpl 潜空间对齐
    "train/loss/aux_g1_teleop_latent_avg":            "G1↔遥操",      # g1→tele 潜空间对齐
    "train/loss/aux_teleop_smpl_latent_avg":          "遥操↔SMPL",    # tele→smpl 潜空间对齐
    "train/loss/aux_reencoded_smpl_g1_latent_avg":    "重编码G1",      # re→g1 一致性损失
    # --- PPO策略统计 ---
    "train/policy/approxkl_avg":                      "KL散度",        # kl >0.05需降学习率
    "train/policy/clipfrac_avg":                      "裁剪比例",      # clip% >30%更新过激
    "train/val/advantage_mean":                       "优势均值",      # adv_μ 应接近0
    "train/val/advantage_std":                        "优势标准差",    # adv_σ
    "train/val/clipfrac_avg":                         "值裁剪率",      # v_clip%
    "train/val/ratio":                                "值函数比",      # v_ratio 应接近1
    "train/val/ratio_var":                            "值比方差",      # v_ratio_v
    # --- 自适应采样统计 ---
    "train/Env/adp_samp/num_episodes_mean":           "均回合数",      # n_ep 每片段平均回合
    "train/Env/adp_samp/num_failures_mean":           "均失败数",      # n_fail 每片段平均失败
    "train/Env/adp_samp/failure_rate_mean":           "失败率",        # fail%
    "train/Env/adp_samp/effective_num_bins":          "有效片段",      # eff_bins
    "train/Env/adp_samp/prob_max_over_uniform":       "采样集中度",    # prob_peak 最高/均匀概率
    "train/Env/adp_samp/episodes_max_over_mean":      "回合不均",      # ep_imbal 最多/平均比
    # --- 性能指标 ---
    "train/fps":                                      "仿真FPS",       # fps 越高越好
    "train/eps":                                      "环境步速",      # eps
    "train/collection_time":                          "采集时间",      # coll_t 秒
    "train/learn_time":                               "更新时间",      # learn_t 秒
    "train/tot_time":                                 "累计时间",      # elapsed 秒
    "train/lr":                                       "学习率",        # lr
}

# Severity thresholds for penalty terms: (warn, critical)
SEVERITY: dict[str, tuple[float, float]] = {
    "train/objective/length":                        (30,  10),
    "train/objective/rewards":                       (0.0, -0.5),
    "train/Env/Episode_Termination/time_out":        (0.5,  0.2),
}

# Healthy direction: True = higher is better, False = lower is better
HIGHER_BETTER: dict[str, bool] = {
    "train/objective/length":               True,
    "train/objective/rewards":              True,
    "train/fps":                            True,
    "train/eps":                            True,
    "train/Env/adp_samp/effective_num_bins": True,
}

# Tags where value can be displayed as percentage
EP_LEN_MAX = 250

PCT_TAGS: dict[str, float] = {
    # For termination counts, use a reference value that makes sense
    # These are absolute counts, not percentages, so we use 1.0 as reference
    # to show them as-is (e.g., 13.75 terminations = 1375% of 1.0)
    # Better to remove these from PCT_TAGS and handle separately
}

# Theoretical maximum value for positive reward terms (weight * 1.0).
# Used to display current reward as a percentage of max in the %/! column.
REWARD_MAX: dict[str, float] = {
    "train/Env/Episode_Reward/tracking_vr_5point_local":    2.0,
    "train/Env/Episode_Reward/tracking_anchor_pos":         3.0,  # fullsplit weight
    "train/Env/Episode_Reward/tracking_anchor_ori":         2.0,
    "train/Env/Episode_Reward/tracking_relative_body_pos":  1.0,
    "train/Env/Episode_Reward/tracking_relative_body_ori":  2.0,
    "train/Env/Episode_Reward/tracking_body_linvel":        1.5,
    "train/Env/Episode_Reward/tracking_body_angvel":        0.3,
    "train/Env/Episode_Reward/tracking_joint_pos_waist":    1.1,
    "train/Env/Episode_Reward/tracking_joint_pos_hip":      0.65,
    "train/Env/Episode_Reward/tracking_joint_pos_hip_yaw":  1.1,
    "train/Env/Episode_Reward/tracking_joint_pos_knee":     0.65,
    "train/Env/Episode_Reward/tracking_joint_pos_ankle":    2.4,
    "train/Env/Episode_Reward/tracking_joint_pos_shoulder": 0.9,
    "train/Env/Episode_Reward/tracking_joint_pos_elbow":    0.9,
    "train/Env/Episode_Reward/tracking_joint_vel_hip":      0.35,
    "train/Env/Episode_Reward/tracking_joint_vel_hip_yaw":  0.35,
    "train/Env/Episode_Reward/tracking_joint_vel_knee":     0.35,
    "train/Env/Episode_Reward/tracking_joint_vel_ankle":    0.5,
    "train/Env/Episode_Reward/tracking_joint_vel_shoulder": 0.2,
    "train/Env/Episode_Reward/tracking_joint_vel_elbow":    0.18,
    "train/Env/Episode_Reward/tracking_joint_vel_wrist":    0.18,
    "train/Env/Episode_Reward/tracking_body_pos_wrist":     1.5,
    "train/Env/Episode_Reward/tracking_body_pos_foot":      2.0,
}

# Target thresholds for error metrics (good-enough values).
# %/! column shows current/target — below 100% means on-target, above means not yet.
ERROR_TARGET: dict[str, float] = {
    "train/Env/Metrics/motion/error_anchor_pos":     0.10,   # m
    "train/Env/Metrics/motion/error_anchor_rot":     0.10,   # rad
    "train/Env/Metrics/motion/error_anchor_lin_vel": 0.20,   # m/s
    "train/Env/Metrics/motion/error_anchor_ang_vel": 0.50,   # rad/s
    "train/Env/Metrics/motion/error_body_pos":       0.05,   # m
    "train/Env/Metrics/motion/error_body_rot":       0.15,   # rad
    "train/Env/Metrics/motion/error_body_lin_vel":   0.30,   # m/s
    "train/Env/Metrics/motion/error_body_ang_vel":   1.00,   # rad/s — core torso only after fix
    "train/Env/Metrics/motion/error_joint_pos":      0.10,   # rad
    "train/Env/Metrics/motion/error_joint_vel":      1.00,   # rad/s
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Series:
    tag: str
    steps: list[int]
    values: list[float]

    @property
    def last_step(self) -> int:
        return self.steps[-1]

    @property
    def first(self) -> float:
        return self.values[0]

    @property
    def last(self) -> float:
        return self.values[-1]

    @property
    def best(self) -> float:
        return max(self.values) if HIGHER_BETTER.get(self.tag, False) else min(self.values)

    def window_mean(self, n: int) -> float:
        return _mean(self.values[-n:])

    def prev_window_mean(self, n: int) -> float:
        end = max(0, len(self.values) - n)
        start = max(0, end - n)
        return _mean(self.values[start:end]) if end > start else self.window_mean(n)

    def trend(self, n: int) -> float:
        return self.window_mean(n) - self.prev_window_mean(n)


def _mean(values: Iterable[float]) -> float:
    lst = list(values)
    return sum(lst) / len(lst) if lst else 0.0


def _dw(s: str) -> int:
    """Display width: CJK chars count as 2, others as 1."""
    import unicodedata
    w = 0
    for c in s:
        ea = unicodedata.east_asian_width(c)
        w += 2 if ea in ("W", "F") else 1
    return w


def _ljust_dw(s: str, width: int) -> str:
    return s + " " * max(0, width - _dw(s))


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def latest_event_file(search_root: Path) -> Path:
    candidates = [p for p in search_root.rglob(EVENT_GLOB) if p.is_file() and p.stat().st_size > 10_000]
    if not candidates:
        # Fall back to any file if none are large enough
        candidates = [p for p in search_root.rglob(EVENT_GLOB) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No TensorBoard event files found under: {search_root}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_event_file(target: Path | None, task: str) -> Path:
    if target is None:
        return latest_event_file(WORKSPACE_ROOT / "logs_rl" / task)
    resolved = target.expanduser().resolve()
    if resolved.is_file():
        return resolved
    if resolved.is_dir():
        return latest_event_file(resolved)
    raise FileNotFoundError(f"Path does not exist: {target}")


def load_series(event_file: Path) -> dict[str, Series]:
    from tensorboard.backend.event_processing import event_accumulator as ea
    import math
    event_dir = event_file.parent
    acc = ea.EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    acc.Reload()
    result: dict[str, Series] = {}
    for tag in acc.Tags().get("scalars", []):
        events = acc.Scalars(tag)
        if events:
            steps = []
            values = []
            for e in events:
                v = float(e.value)
                if math.isfinite(v):
                    steps.append(int(e.step))
                    values.append(v)
            if values:
                result[tag] = Series(tag=tag, steps=steps, values=values)
    return result


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt(v: float, tag: str = "") -> str:
    if v != 0.0 and abs(v) < 1e-3:
        return f"{v:.1e}"
    if abs(v) >= 1000:
        return f"{v:.0f}"
    if abs(v) >= 100:
        return f"{v:.1f}"
    if abs(v) >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _severity(tag: str, value: float) -> str:
    thresholds = SEVERITY.get(tag)
    if thresholds is None:
        return ""
    warn, critical = thresholds
    if value <= critical:
        return "!!"
    if value <= warn:
        return "!"
    return ""


def _trend_arrow(delta: float, tag: str) -> str:
    if abs(delta) < 1e-6:
        return "~"
    up = delta > 0
    good = HIGHER_BETTER.get(tag, False)
    if up:
        return "↑" if good else "↑!"
    return "↓!" if good else "↓"


def _bar(v: float, lo: float, hi: float, width: int = 8) -> str:
    if hi <= lo:
        return " " * width
    frac = max(0.0, min(1.0, (v - lo) / (hi - lo)))
    filled = round(frac * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_header(event_file: Path, series_map: dict[str, Series], max_iterations: int = 100000):
    any_series = next(iter(series_map.values()), None)
    step = any_series.last_step if any_series else 0
    run_dir = event_file.parent.name
    pct = step / max_iterations * 100
    bar = _bar(step, 0, max_iterations, width=30)
    print(f"\n  SONIC  |  run: {run_dir}")
    # Show relative path from run directory
    try:
        rel_path = event_file.relative_to(event_file.parent.parent.parent)
    except ValueError:
        rel_path = event_file
    print(f"  file: {rel_path}")
    print(f"  step: {step} / {max_iterations}  [{bar}]  {pct:.1f}%")
    print(f"{'=' * 60}")


def _percentile_value(s: "Series", pct: float) -> float:
    idx = min(len(s.values) - 1, max(0, round((len(s.values) - 1) * pct)))
    return s.values[idx]


def _percentile_step(s: "Series", pct: float) -> int:
    idx = min(len(s.steps) - 1, max(0, round((len(s.steps) - 1) * pct)))
    return s.steps[idx]


def _slot_width(widths: list[int]) -> int:
    return sum(widths) + 3 * (len(widths) - 1)


def _auto_cols(tags: list[str], series_map: dict[str, Series], mini: bool = False) -> int:
    """Pick 2/3/4 columns based on terminal width."""
    import shutil
    term_w = shutil.get_terminal_size().columns - 2

    if mini:
        hdr = ["metric", "now", "%/!", "↑"]
        fixed_val_w = [0, 8, 8, 2]
    else:
        hdr = ["metric", "80%", "90%", "100%", "%/!", "↑"]
        fixed_val_w = [0, 8, 8, 8, 8, 2]

    # Metric name col width from data; all value cols use fixed widths
    metric_w = max(_dw(h) for h in hdr)
    for tag in tags:
        short = SHORT.get(tag, tag.split("/")[-1])
        metric_w = max(metric_w, _dw(short))

    sample_widths = [metric_w] + fixed_val_w[1:]
    slot_w = sum(sample_widths) + 3 * (len(sample_widths) - 1)
    for cols in [4, 3, 2]:
        total = cols * slot_w + (cols - 1) * 3
        if total <= term_w:
            return cols
    return 1


def print_flat_table(
    group_tags: list[tuple[str, list[str]]], series_map: dict[str, Series], window: int, cols: int = 0,
    mini: bool = False,
):
    """Print table with group separator rows between relevance groups."""
    all_tags = [t for _, tags in group_tags for t in tags]
    if not all_tags:
        return
    if cols == 0:
        cols = _auto_cols(all_tags, series_map, mini=mini)

    ref_tag = max((t for t in all_tags if t in series_map), key=lambda t: len(series_map[t].steps), default=None)
    if ref_tag is None:
        return
    ref = series_map[ref_tag]

    if mini:
        s100 = _percentile_step(ref, 1.0)
        hdr = ["metric", f"{s100}", "%/!", "↑"]
    else:
        dataIdx = [0.8, 0.9, 1.0]
        s80 = _percentile_step(ref, dataIdx[0])
        s90 = _percentile_step(ref, dataIdx[1])
        s100 = _percentile_step(ref, dataIdx[2])
        hdr = ["metric", f"{s80}", f"{s90}", f"{s100}", "%/!", "↑"]
    n_cols = len(hdr)

    flat_cells: list[list[str] | str] = []
    for group_name, tags in group_tags:
        group_rows: list[list[str]] = []
        for tag in tags:
            s = series_map.get(tag)
            if s is None:
                continue
            short = SHORT.get(tag, tag.split("/")[-1])
            cur = _percentile_value(s, 1.0)
            if tag in PCT_TAGS:
                pct_sev = f"{cur / PCT_TAGS[tag] * 100:.1f}%"
            elif tag in REWARD_MAX:
                pct_sev = f"{cur / REWARD_MAX[tag] * 100:.1f}%"
            elif tag in ERROR_TARGET:
                gap = ERROR_TARGET[tag] - cur
                pct_sev = f"{gap:+.3f}"
            elif tag == "train/objective/length":
                pct_sev = f"{cur / EP_LEN_MAX * 100:.1f}%"
            else:
                pct_sev = _severity(tag, cur)
            arrow = _trend_arrow(s.trend(window), tag)
            v100 = _fmt(cur, tag)
            if mini:
                group_rows.append([short, v100, pct_sev, arrow])
            else:
                v80 = _fmt(_percentile_value(s, 0.8), tag)
                v90 = _fmt(_percentile_value(s, 0.9), tag)
                group_rows.append([short, v80, v90, v100, pct_sev, arrow])
        if group_rows:
            if flat_cells:
                flat_cells.append(group_name)
            flat_cells.extend(group_rows)

    if not flat_cells:
        return

    # Global column widths — metric col from data, value/pct/arrow cols fixed to avoid jitter
    # fixed widths: val=8, pct=7, arrow=2
    if mini:
        fixed = [0, 8, 8, 2]
    else:
        fixed = [0, 8, 8, 8, 8, 2]
    col_widths = [max(_dw(hdr[j]), fixed[j]) for j in range(n_cols)]
    for cell in flat_cells:
        if isinstance(cell, str):
            continue
        col_widths[0] = max(col_widths[0], _dw(cell[0]))
    slot_w = [col_widths[:] for _ in range(cols)]

    SEP = "   "
    COL_JOIN = " | "
    SEP_JOIN = "-+-"
    left_cols = {0}

    def _fmt_row(row: list[str], widths: list[int]) -> str:
        parts = []
        for j, val in enumerate(row):
            if j in left_cols:
                parts.append(_ljust_dw(val, widths[j]))
            else:
                parts.append(val.rjust(widths[j]))
        return COL_JOIN.join(parts)

    def _sep_row(widths: list[int]) -> str:
        return SEP_JOIN.join("-" * widths[j] for j in range(n_cols))

    hdr_line = SEP.join(_fmt_row(hdr, slot_w[c]) for c in range(cols))
    sep_line = SEP.join(_sep_row(slot_w[c]) for c in range(cols))
    print(hdr_line)
    print(sep_line)

    row_buf: list[list[str]] = [[] for _ in range(cols)]
    col_cur = 0

    def _flush() -> None:
        nonlocal col_cur
        if any(row_buf[c] for c in range(cols)):
            for c in range(col_cur, cols):
                row_buf[c] = [""] * n_cols
            print(SEP.join(_fmt_row(row_buf[c], slot_w[c]) for c in range(cols)))
        for c in range(cols):
            row_buf[c] = []
        col_cur = 0

    def _print_sep(name: str) -> None:
        # each slot: sum of field widths + " | " between fields
        one_slot = sum(col_widths) + 3 * (n_cols - 1)
        # content width = slots + SEP between slots; subtract 2 for leading "  " and 2 for trailing "  "
        slot_total = one_slot * cols + 3 * (cols - 1) - 4
        label = f" {name} "
        print(f"  {label.center(slot_total, '─')}  ")

    for item in flat_cells:
        if isinstance(item, str):
            _flush()
            _print_sep(item)
        else:
            row_buf[col_cur] = item
            col_cur += 1
            if col_cur >= cols:
                _flush()

    _flush()


def print_health(series_map: dict[str, Series], window: int, event_file: Path | None = None):
    """One-line health summary with key indicators."""
    def get(tag):
        s = series_map.get(tag)
        return s.last if s else None

    ep_len = get("train/objective/length")
    reward = get("train/objective/rewards")
    noise = get("train/Policy/mean_noise_std")
    kl = get("train/policy/approxkl_avg")
    clip = get("train/policy/clipfrac_avg")
    aux = get("train/loss/total_aux_loss_avg")
    g1_rec = get("train/loss/aux_g1_recon_avg")
    anc_pos = get("train/Env/Metrics/motion/error_anchor_pos")
    anc_rot = get("train/Env/Metrics/motion/error_anchor_rot")
    fps = get("train/fps")
    coll_t = get("train/collection_time")
    learn_t = get("train/learn_time")

    parts = []
    if ep_len is not None:
        bar = _bar(ep_len, 0, EP_LEN_MAX)
        parts.append(f"回合长度={ep_len:.0f} [{bar}]")
    if reward is not None:
        parts.append(f"总奖励={reward:.2f}")
    if anc_pos is not None:
        parts.append(f"根位置误差={anc_pos:.3f}")
    if anc_rot is not None:
        parts.append(f"根朝向误差={anc_rot:.3f}")
    if g1_rec is not None:
        parts.append(f"G1重建={g1_rec:.4f}")
    if kl is not None:
        flag = " ⚠" if kl > 0.05 else ""
        parts.append(f"KL={kl:.4f}{flag}")
    if clip is not None:
        flag = " ⚠" if clip > 0.3 else ""
        parts.append(f"裁剪={clip:.3f}{flag}")
    if aux is not None:
        parts.append(f"辅助损失={aux:.4f}")
    if noise is not None:
        parts.append(f"动作噪声={noise:.3f}")
    if fps is not None:
        parts.append(f"FPS={fps:.0f}")
    if coll_t is not None and learn_t is not None:
        parts.append(f"采集+更新={coll_t:.1f}+{learn_t:.1f}s")

    print(f"\n  状态: {' | '.join(parts)}")

    # Trend assessment
    ep_s = series_map.get("train/objective/length")
    reward_s = series_map.get("train/objective/rewards")
    if ep_s and len(ep_s.values) >= window * 2:
        ep_trend = ep_s.trend(window)
        rew_trend = reward_s.trend(window) if reward_s else 0.0
        if ep_trend > 2:
            status = "上升中"
        elif ep_trend < -2:
            status = "下降中 ⚠"
        elif abs(ep_trend) < 1 and abs(rew_trend) < 0.02:
            status = "平台期"
        else:
            status = "稳定"
        print(f"  趋势:  回合长度变化={ep_trend:+.1f}  奖励变化={rew_trend:+.3f}  → {status}")

    # Aux loss diagnosis
    if g1_rec is not None:
        v = g1_rec
        if v < 0.05:
            aux_status = "已收敛"
        elif v < 0.2:
            aux_status = "学习中"
        else:
            aux_status = "偏高 !"
        print(f"  辅助:  G1重建={v:.4f}  → {aux_status}")

    # Blacklist count — file lives at all_modes/ level (4 levels up from event file)
    blacklist_path = event_file.parent.parent.parent.parent / "motion_blacklist.txt"
    if blacklist_path.exists():
        with open(blacklist_path) as f:
            count = sum(1 for line in f if line.strip() and not line.startswith("#"))
        print(f"  黑名单: {count} 个动作已排除")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_mini(event_file: Path, series_map: dict[str, Series]):
    """Mini mode: one compact line per group showing only the latest value."""
    any_series = next(iter(series_map.values()), None)
    step = any_series.last_step if any_series else 0
    run_dir = event_file.parent.name
    pct = step / 100000 * 100
    bar = _bar(step, 0, 100000, width=20)
    print(f"\n  SONIC  step={step} [{bar}] {pct:.1f}%  run={run_dir}")

    for group_name, tags in GROUPS:
        items = []
        for tag in tags:
            s = series_map.get(tag)
            if s is None:
                continue
            short = SHORT.get(tag, tag.split("/")[-1])
            cur = s.last
            val_str = _fmt(cur, tag)
            if tag in REWARD_MAX:
                extra = f"({cur / REWARD_MAX[tag] * 100:.0f}%)"
            elif tag in ERROR_TARGET:
                gap = ERROR_TARGET[tag] - cur
                extra = f"({gap:+.3f})"
            elif tag == "train/objective/length":
                extra = f"({cur / EP_LEN_MAX * 100:.0f}%)"
            else:
                extra = ""
            items.append(f"{short}={val_str}{extra}")
        if items:
            print(f"  [{group_name}] {' | '.join(items)}")


def run_once(event_file: Path, window: int, groups: list[str] | None, mini: bool = False):
    series_map = load_series(event_file)
    if not series_map:
        print("No scalar data found.")
        return

    print_header(event_file, series_map)
    print_health(series_map, window, event_file=event_file)
    print()

    # Build active groups with deduplicated tags
    seen: set[str] = set()
    group_tags: list[tuple[str, list[str]]] = []
    active_groups = GROUPS if not groups else [(n, t) for n, t in GROUPS if n in groups]
    for group_name, tags in active_groups:
        deduped: list[str] = []
        for tag in tags:
            if tag not in seen:
                seen.add(tag)
                deduped.append(tag)
        if deduped:
            group_tags.append((group_name, deduped))

    print_flat_table(group_tags, series_map, window, cols=0, mini=mini)


def main():
    parser = argparse.ArgumentParser(description="Analyze SONIC training logs.")
    parser.add_argument(
        "event_file", nargs="?", type=Path,
        help="Event file or run directory. Defaults to latest under logs_rl/<task>.",
    )
    parser.add_argument(
        "--task", default="TRL_Tienkung2Pro_Track",
        help="Task name for default log path.",
    )
    parser.add_argument(
        "--window", type=int, default=10,
        help="Trailing window size for trend (default: 10).",
    )
    parser.add_argument(
        "--groups", nargs="*",
        choices=[n for n, _ in GROUPS],
        help="Show only these groups. Default: all.",
    )
    parser.add_argument(
        "--watch", type=float, default=0.0, metavar="SECONDS",
        help="Live refresh interval in seconds (e.g. --watch 30). 0 = run once.",
    )
    parser.add_argument(
        "--mini", action="store_true",
        help="Compact one-line-per-group output showing only the latest value.",
    )
    args = parser.parse_args()

    event_file = resolve_event_file(args.event_file, args.task)

    if args.watch > 0:
        last_output = ""
        try:
            while True:
                try:
                    buf = io.StringIO()
                    with contextlib.redirect_stdout(buf):
                        run_once(event_file, args.window, args.groups, mini=args.mini)
                    last_output = buf.getvalue()
                except Exception:
                    pass
                sys.stdout.write("\033[H\033[J" + last_output)
                sys.stdout.write(f"\n  [refreshing every {args.watch:.0f}s — Ctrl+C to stop]\n")
                sys.stdout.flush()
                time.sleep(args.watch)
        except KeyboardInterrupt:
            pass
    else:
        run_once(event_file, args.window, args.groups, mini=args.mini)


if __name__ == "__main__":
    main()
