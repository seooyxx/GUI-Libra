#!/bin/bash
set -x

# ========== Please modify the following variables as needed ==========
# MODEL_PATH: path to your base model or SFT checkpoint
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

# TRAIN_FILES: training dataset (HuggingFace: dataset_name@split, or local path like path/to/train.parquet)
TRAIN_FILES=your_dataset@train

# VAL_FILES: validation dataset
VAL_FILES=your_dataset@test
# =====================================================================

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    trainer.total_epochs=2 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VAL_FILES} \
    data.format_prompt=./examples/format_prompt/guir1_r+g_notemplate.jinja \
    data.prompt_key=context \
    data.val_batch_size=512 \
    data.max_prompt_length=8092 \
    data.max_response_length=1500 \
    data.max_pixels=2508800  \
    data.rollout_batch_size=256 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=128  \
    worker.rollout.n=8 \
    worker.rollout.top_p=0.98 \
    worker.rollout.gpu_memory_utilization=0.8 \
    worker.actor.micro_batch_size_per_device_for_update=4  \
    worker.actor.micro_batch_size_per_device_for_experience=8  \
    worker.reward.reward_type=sequential \
    worker.reward.reward_function=./examples/reward_function/r1gui.py:compute_score \
    trainer.experiment_name=Qwen2.5VL_3B_fromcpt636_weightedSFT_ratio2_mixnoreasoning_grpo_ngsx1.5mean_0.5-1_topp0.98_rollout8_wkl0.001_cliph0.2_80ktrain_len1500 \
    trainer.n_gpus_per_node=8 \
    trainer.save_freq=100 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.2 \
    worker.actor.loss_avg_mode=token \
    worker.actor.model.freeze_vision_tower=False \
    algorithm.adv_estimator=grpo_weighted_positive_negative \
    algorithm.kl_coef=0.001 \
    algorithm.weighted_posneg_base=0.5 \
    algorithm.weighted_posneg_coef=1.5 \

    # algorithm.disable_kl=True \
  