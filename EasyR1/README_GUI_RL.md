# GUI 强化学习 (RL) 训练指南

本文档说明如何使用 `examples/` 目录下的 `gui_grpo.sh` 和 `gui_grpo_qwen3.sh` 进行 GUI 智能体的 GRPO（Group Relative Policy Optimization）强化学习训练。

## 概述

两个脚本均基于 EasyR1 框架，用于训练视觉语言模型（VLM）执行 GUI 操作任务，输出格式化的动作指令（如点击、输入、滑动等）。

| 脚本 | 适用模型 | 奖励函数 | 说明 |
|------|----------|----------|------|
| `gui_grpo.sh` | Qwen2.5-VL-3B-Instruct | `r1gui.py` | Qwen2.5-VL 系列，支持 `<think>` 标签 |
| `gui_grpo_qwen3.sh` | Qwen3-VL-8B-Instruct | `r1gui_qwen3vl.py` | Qwen3-VL 系列，支持 `<thinking>` 标签 |

## 快速开始

### 1. 环境准备

确保已安装 EasyR1 及相关依赖：

```bash
cd EasyR1
pip install -e .
```

### 2. 修改配置

在运行前，需修改脚本顶部的三个变量：

```bash
# 模型路径：基础模型或 SFT checkpoint
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct   # gui_grpo.sh
# 或
MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct     # gui_grpo_qwen3.sh

# 训练集：HuggingFace 格式为 dataset_name@split，本地路径如 path/to/train.parquet
TRAIN_FILES=your_dataset@train

# 验证集
VAL_FILES=your_dataset@test
```

### 3. 运行训练

```bash
# Qwen2.5-VL 训练
bash examples/gui_grpo.sh

# Qwen3-VL 训练
bash examples/gui_grpo_qwen3.sh
```

## 数据集格式

训练数据需包含以下字段，且与 `prompt_key=context` 和 `reward_type=sequential` 配合使用。

### 必需字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `context` | string | 提示内容，通常包含多模态信息（如图片占位符） |
| `images` | list | 与任务相关的截图或界面图像 |
| `ground_truth` | dict | 用于计算奖励的标注信息 |

### `ground_truth` 结构

```python
{
    "gt_action": str,        # 动作类型：click, longpress, type, write, scroll, openapp, swipe 等
    "gt_bbox": list,         # 归一化坐标 [x1, y1, x2, y2]，或 [x, y]，数值范围 0-1000
    "gt_point_2d": list,     # 或使用 point_2d 格式
    "gt_input_text": str,    # 输入文本（type/write 等动作需要）
    "image_width": int,      # 图像原始宽度
    "image_height": int,     # 图像原始高度
    "image_width_new": int,  # 预处理后的图像宽度（若与原始不同）
    "image_height_new": int, # 预处理后的图像高度
    "is_grounding": bool     # 是否为定位任务（True：点击/长按；False：需推理的任务）
}
```

### 动作类型映射

奖励函数支持的动作类型包括：

- `click` / `longpress`：需提供 `point_2d` 坐标
- `type` / `write` / `select` / `scroll` / `openapp` / `swipe` / `terminate` / `answer`：需提供 `value` 文本

## 模型输出格式

模型应输出如下结构，奖励函数会据此计算格式分与准确分：

```
<think>
你的逐步推理过程...
</think>
<answer>
{
  "action_type": "click",
  "action_description": "点击搜索按钮",
  "value": "None",
  "point_2d": [512, 256]
}
</answer>
```

- `point_2d`：屏幕坐标 `[x, y]`；若不适用，使用 `"none"` 或 `[-100, -100]`
- 对于 Qwen3-VL，使用 `<thinking>...</thinking>` 替代 `<think></think>`

## 奖励函数说明

奖励由三部分组成：

1. **format**：输出是否符合上述 JSON 结构（`action_type`, `action_description`, `value`, `point_2d`）
2. **accuracy**：预测动作与 `ground_truth` 是否一致
   - 点击/长按：坐标误差在阈值内得 1 分
   - 文本类动作：与 `gt_input_text` 的 F1 ≥ 0.5 得 1 分
3. **use_reasoning**：是否包含推理内容（`<think>` 或 `<thinking>`）

对于非 grounding 任务，若未使用推理，格式分会置为 0；对于 grounding 任务，若使用推理，格式分会置为 0。

最终 `overall` 分数：`(1 - format_weight) * accuracy + format_weight * format_score`，默认 `format_weight=0.1`。

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `trainer.total_epochs` | 2 | 训练轮数 |
| `trainer.n_gpus_per_node` | 8 | 每节点 GPU 数量 |
| `data.max_prompt_length` | 8092 | 最大 prompt 长度 |
| `data.max_response_length` | 1500 | 最大生成长度 |
| `data.max_pixels` | 2508800 | 图像最大像素数 |
| `worker.actor.global_batch_size` | 128 | 全局 batch size |
| `worker.rollout.n` | 8 | 每 prompt 采样数 |
| `worker.rollout.top_p` | 0.98 | 采样 top_p |
| `algorithm.adv_estimator` | grpo_weighted_positive_negative | 优势估计方式 |
| `algorithm.kl_coef` | 0.001 | KL 惩罚系数 |

### 显存不足时

可尝试：

- 降低 `worker.rollout.gpu_memory_utilization`（如 0.6）
- 降低 `worker.actor.micro_batch_size_per_device_for_update`
- 降低 `worker.actor.micro_batch_size_per_device_for_experience`
- 启用 `worker.actor.offload.offload_params=true`

## 多节点训练

1. 启动 Ray head 节点：

```bash
ray start --head --port=6379 --dashboard-host=0.0.0.0
```

2. 在各 worker 节点连接 head：

```bash
ray start --address=<head_node_ip>:6379
```

3. 在 head 节点执行训练脚本：

```bash
bash examples/gui_grpo.sh
```

## 合并 Checkpoint

训练完成后，可合并为 Hugging Face 格式：

```bash
python3 scripts/model_merger.py --local_dir checkpoints/easy_r1/<experiment_name>/global_step_<step>/actor
```

## 参考

- [EasyR1 主文档](../README.md)
- [GUI-R1 论文](https://arxiv.org/abs/2504.10458)
- [GRPO 算法介绍](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer)
