"""分析 eval 失败动作的共性。

用法:
    python gear_sonic/scripts/analyze_eval_failures.py output/eval_xxx/metrics_eval.json
    python gear_sonic/scripts/analyze_eval_failures.py output/eval_xxx/metrics_eval.json --top 20
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


# ── 动作名解析 ──────────────────────────────────────────────────────────────

def parse_motion_key(key: str) -> dict:
    """从动作名提取结构化信息。

    示例: walk_ff_loop_360_003__A052_M
          neutral_idle_loop_001__A086_M
          inj_right_leg_idle_loop_max_003__A078
    """
    # 去掉末尾的 _M 标记和 __Axxx 编号
    base = re.sub(r"__A\d+(_M)?$", "", key)

    # 提取动作类型（第一个词或前两个词）
    parts = base.split("_")

    # 受伤动作特殊处理
    if parts[0] == "inj":
        action_type = "_".join(parts[:4])  # inj_right_leg_idle
        body_part = parts[1] + "_" + parts[2]  # right_leg
    else:
        action_type = parts[0]
        body_part = None

    # 方向
    direction = None
    for p in parts:
        if p in ("ff", "180", "270", "360", "315", "225", "090", "045", "135"):
            direction = p
            break

    # 阶段（start/loop/stop）
    phase = None
    for p in parts:
        if p in ("start", "loop", "stop", "idle"):
            phase = p
            break

    # 速度修饰
    speed = None
    for p in parts:
        if p in ("slow", "fast", "big", "small", "max", "min"):
            speed = p
            break

    return {
        "key": key,
        "action_type": action_type,
        "direction": direction,
        "phase": phase,
        "speed": speed,
        "body_part": body_part,
        "base": base,
    }


def get_category(key: str) -> str:
    """粗粒度分类。"""
    k = key.lower()
    if k.startswith("walk_ff"):
        return "前向行走"
    if k.startswith("walk_sideway"):
        return "侧向行走"
    if k.startswith("slow_walk") or k.startswith("neutral_walk"):
        return "慢走/中性走"
    if k.startswith("arc_walk"):
        return "弧形行走"
    if k.startswith("change_idle"):
        return "切换站立"
    if k.startswith("neutral_idle") or k.startswith("idle_loop"):
        return "中性站立"
    if k.startswith("stand"):
        return "站立"
    if k.startswith("inj"):
        return "受伤动作"
    return "其他"


# ── 主分析 ──────────────────────────────────────────────────────────────────

def analyze(metrics_path: str, top: int = 20) -> None:
    path = Path(metrics_path)
    if not path.exists():
        print(f"文件不存在: {path}")
        return

    with open(path) as f:
        data = json.load(f)

    # 提取 failed_keys
    failed_keys = data.get("failed_keys", [])
    all_keys = data.get("eval/all_metrics_dict", {}).get("motion_keys", [])
    success_rate = data.get("eval/success/success_rate", None)

    if not failed_keys:
        print("没有失败动作，或 metrics_eval.json 中没有 failed_keys 字段。")
        return

    total = len(all_keys) if all_keys else "未知"
    print(f"\n{'='*60}")
    print(f"总动作数: {total}  |  失败数: {len(failed_keys)}  |  成功率: {success_rate:.3f}" if success_rate else f"总动作数: {total}  |  失败数: {len(failed_keys)}")
    print(f"{'='*60}\n")

    # ── 1. 按大类统计 ──
    cat_counter = Counter(get_category(k) for k in failed_keys)
    all_cat_counter = Counter(get_category(k) for k in all_keys) if all_keys else None

    print("【失败动作大类分布】")
    for cat, cnt in cat_counter.most_common():
        total_in_cat = all_cat_counter[cat] if all_cat_counter else "?"
        pct = cnt / len(failed_keys) * 100
        fail_rate = f"({cnt}/{total_in_cat} = {cnt/total_in_cat*100:.0f}%失败)" if isinstance(total_in_cat, int) else ""
        print(f"  {cat:<12} {cnt:>4}个  占失败{pct:.1f}%  {fail_rate}")

    # ── 2. 按动作类型统计 ──
    parsed = [parse_motion_key(k) for k in failed_keys]
    type_counter = Counter(p["action_type"] for p in parsed)

    print(f"\n【失败动作类型 Top {top}】")
    for atype, cnt in type_counter.most_common(top):
        pct = cnt / len(failed_keys) * 100
        print(f"  {atype:<35} {cnt:>4}个  {pct:.1f}%")

    # ── 3. 方向分析 ──
    dir_counter = Counter(p["direction"] for p in parsed if p["direction"])
    if dir_counter:
        print("\n【失败动作方向分布】")
        for d, cnt in dir_counter.most_common():
            print(f"  {d:<8} {cnt:>4}个")

    # ── 4. 阶段分析（start/loop/stop）──
    phase_counter = Counter(p["phase"] for p in parsed if p["phase"])
    if phase_counter:
        print("\n【失败动作阶段分布】")
        for ph, cnt in phase_counter.most_common():
            print(f"  {ph:<8} {cnt:>4}个")

    # ── 5. 失败率最高的大类 ──
    if all_cat_counter:
        print("\n【各大类失败率排名】")
        fail_rates = []
        for cat in all_cat_counter:
            total_c = all_cat_counter[cat]
            failed_c = cat_counter.get(cat, 0)
            fail_rates.append((cat, failed_c, total_c, failed_c / total_c))
        fail_rates.sort(key=lambda x: -x[3])
        for cat, fc, tc, fr in fail_rates:
            bar = "█" * int(fr * 20)
            print(f"  {cat:<12} {bar:<20} {fr*100:.1f}%  ({fc}/{tc})")

    # ── 6. 每个指标的失败动作均值 vs 成功动作均值 ──
    all_dict = data.get("eval/all_metrics_dict", {})
    failed_dict = data.get("eval/failed_metrics_dict", {})
    metric_keys = [k for k in all_dict if k not in ("motion_keys", "terminated", "progress", "sampling_prob")]

    if metric_keys and failed_dict:
        print("\n【失败动作 vs 全体 指标对比 (越小越好)】")
        print(f"  {'指标':<30} {'全体均值':>10} {'失败均值':>10} {'差异':>8}")
        print(f"  {'-'*60}")
        diffs = []
        for mk in metric_keys:
            all_vals = np.array(all_dict.get(mk, []))
            fail_vals = np.array(failed_dict.get(mk, []))
            if len(all_vals) == 0 or len(fail_vals) == 0:
                continue
            all_mean = float(np.mean(all_vals))
            fail_mean = float(np.mean(fail_vals))
            diffs.append((mk, all_mean, fail_mean, fail_mean - all_mean))
        diffs.sort(key=lambda x: -abs(x[3]))
        for mk, am, fm, diff in diffs[:15]:
            sign = "↑" if diff > 0 else "↓"
            print(f"  {mk:<30} {am:>10.3f} {fm:>10.3f} {sign}{abs(diff):>6.3f}")

    # ── 7. 输出失败动作列表到文件 ──
    out_file = path.parent / "failed_keys.txt"
    with open(out_file, "w") as f:
        f.write("\n".join(sorted(failed_keys)))
    print(f"\n失败动作列表已保存到: {out_file}")
    print(f"共 {len(failed_keys)} 个失败动作\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 eval 失败动作共性")
    parser.add_argument("metrics_json", help="metrics_eval.json 路径")
    parser.add_argument("--top", type=int, default=20, help="显示 Top N 动作类型")
    args = parser.parse_args()
    analyze(args.metrics_json, top=args.top)
