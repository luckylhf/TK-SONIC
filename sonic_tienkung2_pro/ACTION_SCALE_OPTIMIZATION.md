# Action Space 优化分析

## 问题

当前 TienKung2 Pro 的 action scale 是否合理？

## 分析结果

### 当前 Action Scale 值

```
TIENKUNG2_PRO_ACTION_SCALE 范围: 0.063 - 0.125
平均值: 0.088

分布:
  - 腿部 (hip/knee):     0.084 - 0.118
  - 脚部 (ankle):        0.069
  - 手臂 (shoulder):     0.075 - 0.083
  - 手臂 (elbow/wrist):  0.063 - 0.083
  - 躯干 (body_yaw):     0.125
```

### 计算公式

```
action_scale = 0.25 * effort_limit / stiffness
```

这意味着 policy 输出的 action 被缩放到 effort_limit 的 **25%**。

### 问题

1. **Action 范围太小**：0.063 - 0.125 相对较小
2. **表达能力受限**：Policy 只能使用 effort limit 的 1/4
3. **站立任务困难**：需要更大的动作范围来保持平衡
4. **训练效率低**：Policy 无法充分利用机器人的动作能力

## 解决方案

### 修改 Action Scale 系数

**从 0.25 增加到 0.5**

```python
# 修改前
TIENKUNG2_PRO_ACTION_SCALE[_n] = 0.25 * _effort[_n] / _stiff[_n]

# 修改后
TIENKUNG2_PRO_ACTION_SCALE[_n] = 0.5 * _effort[_n] / _stiff[_n]
```

### 效果

```
新的 Action Scale 范围: 0.125 - 0.250
平均值: 0.176 (增加 2 倍)

Policy 现在可以使用 effort limit 的 50% (之前是 25%)
```

### 优势

1. **更大的动作范围**：Policy 有更多的控制权
2. **更好的表达能力**：可以执行更复杂的平衡动作
3. **更快的训练**：更大的 action space 意味着更多的学习空间
4. **更稳定的站立**：可以做出更大的纠正动作

## 其他优化

同时进行的其他优化：

1. **放宽 Termination 阈值**
   - 增加 episode 长度
   - 让机器人有更多时间学习

2. **增加数据收集**
   - `num_steps_per_env`: 24 → 32
   - 每次迭代收集更多数据

3. **降低学习率**
   - `actor_learning_rate`: 2e-5 → 1e-5
   - `critic_learning_rate`: 1e-3 → 5e-4
   - 提高训练稳定性

## 建议

✅ **已实施**：
- Action scale 从 0.25 增加到 0.5
- Termination 阈值放宽
- 学习率降低
- `num_steps_per_env` 增加

📊 **下一步**：
1. 运行优化版本的训练
2. 监控 episode length 和 reward 的改进
3. 如果仍然不够，考虑进一步增加 action scale 到 0.75 或 1.0
4. 添加更多的 standing motion variations

## 参考

- 文件: `gear_sonic/envs/manager_env/robots/tienkung2_pro.py` (第 389-402 行)
- 配置: `gear_sonic/config/exp/manager/universal_token/all_modes/sonic_tienkung2_pro_standing.yaml`
- 文档: `gear_sonic/CLAUDE.md` (优化部分)
