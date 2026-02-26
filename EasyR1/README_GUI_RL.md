# Reinforcement Learning for GUI-Libra

This document explains how to use the `gui_grpo.sh` and `gui_grpo_qwen3.sh` scripts under `examples/` for GRPO (Group Relative Policy Optimization) reinforcement learning training of GUI agents.

## Overview

Both scripts are built on the EasyR1 framework and train vision-language models (VLMs) to perform GUI tasks by outputting structured action instructions (click, type, scroll, etc.).

| Script | Model | Reward Function | Note |
|--------|-------|-----------------|------|
| `gui_grpo.sh` | Qwen2.5-VL-3B-Instruct | `r1gui.py` | Uses `<think>` tag |
| `gui_grpo_qwen3.sh` | Qwen3-VL-8B-Instruct | `r1gui_qwen3vl.py` | Uses `<thinking>` tag |

## Installation

```bash
cd EasyR1
bash set_up.sh
```

## Dataset

### Download

Clone the [GUI-Libra-81K-RL](https://huggingface.co/datasets/GUI-Libra/GUI-Libra-81K-RL) dataset from Hugging Face:

```bash
git lfs install
git clone https://huggingface.co/datasets/GUI-Libra/GUI-Libra-81K-RL
```

### Structure

The dataset contains three folders:

| Folder | Size | Description |
|--------|------|-------------|
| `downsampled/` | ~40K samples | Downsampled RL training data (**recommended**) |
| `shards/` | ~81K samples | Full SFT data that can also be used for RL training |
| `validation/` | — | Validation data for evaluating RL training progress |

We recommend using the `downsampled/` (40K) subset for RL training, as it provides better training efficiency while maintaining strong performance.

### Data Format

Each sample should contain the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `context` | string | Prompt content with multimodal information (e.g., image placeholders) |
| `images` | list | Screenshots or UI images associated with the task |
| `ground_truth` | dict | Annotation used for reward computation |

The `ground_truth` structure:

```json
{
    "gt_action": "click",
    "gt_bbox": [x1, y1, x2, y2],
    "gt_point_2d": [x, y],
    "gt_input_text": "search query",
    "image_width": 1920,
    "image_height": 1080,
    "image_width_new": 1920,
    "image_height_new": 1080,
    "is_grounding": true
}
```

- `gt_action`: Action type — `click`, `longpress`, `type`, `write`, `scroll`, `openapp`, `swipe`, `terminate`, `answer`, `select`
- `gt_bbox` / `gt_point_2d`: Normalized coordinates in range [0, 1000]
- `gt_input_text`: Required for text-based actions (`type`, `write`, etc.)
- `is_grounding`: `true` for grounding tasks (click/longpress); `false` for reasoning tasks

## Quick Start

### 1. Configure the Script

Edit the variables at the top of the training script:

```bash
# Model path: base model or SFT checkpoint
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

# Training set: path to parquet file(s)
TRAIN_FILES=path/to/GUI-Libra-81K-RL/data/downsampled/train.parquet

# Validation set
VAL_FILES=path/to/GUI-Libra-81K-RL/data/validation/test_androidcontrol_mind2web_dataset.parquet
```

### 2. Run Training

```bash
# Qwen2.5-VL
bash examples/gui_grpo.sh

# Qwen3-VL
bash examples/gui_grpo_qwen3.sh
```

## Model Output Format

The model should produce the following structured output:

```
<think>
Step-by-step reasoning about the GUI task...
</think>
<answer>
{
  "action_type": "click",
  "action_description": "Click the search button",
  "value": "None",
  "point_2d": [512, 256]
}
</answer>
```

- `point_2d`: Screen coordinates `[x, y]` in [0, 1000]; use `"none"` or `[-100, -100]` if not applicable
- For Qwen3-VL, use `<thinking>...</thinking>` instead of `<think>...</think>`

## Reward Function

The reward consists of three components:

1. **Format**: Whether the output follows the required JSON structure (`action_type`, `action_description`, `value`, `point_2d`)
2. **Accuracy**: Whether the predicted action matches the `ground_truth`
   - Click/longpress: coordinate error within threshold scores 1
   - Text actions: F1 score with `gt_input_text` >= 0.5 scores 1
3. **Reasoning usage**: Whether reasoning content (`<think>` or `<thinking>`) is present
   - For non-grounding tasks: format score is set to 0 if no reasoning is used
   - For grounding tasks: format score is set to 0 if reasoning is used

Overall score: `(1 - format_weight) * accuracy + format_weight * format_score` (default `format_weight=0.1`)

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trainer.total_epochs` | 2 | Training epochs |
| `trainer.n_gpus_per_node` | 8 | GPUs per node |
| `data.max_prompt_length` | 8092 | Max prompt length |
| `data.max_response_length` | 1500 | Max generation length |
| `data.max_pixels` | 2508800 | Max image pixels |
| `worker.actor.global_batch_size` | 128 | Global batch size |
| `worker.rollout.n` | 8 | Samples per prompt |
| `worker.rollout.top_p` | 0.98 | Sampling top_p |
| `algorithm.adv_estimator` | grpo_weighted_positive_negative | Advantage estimator |
| `algorithm.kl_coef` | 0.001 | KL penalty coefficient |

### Out of Memory

Try the following adjustments:

- Lower `worker.rollout.gpu_memory_utilization` (e.g., 0.6)
- Lower `worker.actor.micro_batch_size_per_device_for_update`
- Lower `worker.actor.micro_batch_size_per_device_for_experience`
- Enable `worker.actor.offload.offload_params=true`


## Checkpoint Merging

After training, merge checkpoints into Hugging Face format:

```bash
python3 scripts/model_merger.py \
    --local_dir checkpoints/easy_r1/<experiment_name>/global_step_<step>/actor
```

## References

- [EasyR1 Documentation](../README.md)
- [GUI-Libra Paper](https://arxiv.org/abs/2602.22190)
- [GRPO Algorithm](https://huggingface.co/docs/trl/v0.16.1/en/grpo_trainer)
