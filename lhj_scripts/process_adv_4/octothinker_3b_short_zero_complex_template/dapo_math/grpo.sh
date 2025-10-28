#!/usr/bin/env bash
set -xeuo pipefail

source /root/miniconda3/etc/profile.d/conda.sh

conda activate /mnt/shared-storage-user/shared-storage-ailab-llmfudan/liaochenyang/envs/verl

cd /mnt/shared-storage-user/shared-storage-ailab-llmfudan/liaochenyang/verl-0.5.0

cp -r /mnt/shared-storage-user/liaochenyang/cfs /root

export PARTITION=${GROUP}
export CFSCTL=/root/cfs/bin/cfsctl
export CFG=/root/cfs/cfsd.cfg

handle_sigterm() {
    echo "Received SIGTERM signal. Cleaning up..."
    $CFSCTL -p $PARTITION -n $NODE_COUNT -X $MASTER_ADDR -s $CFG stop
    exit 0
}
# 为sigterm信号安装处理函数
trap 'handle_sigterm' SIGTERM
$CFSCTL -p $PARTITION -n $NODE_COUNT -X $MASTER_ADDR -s $CFG start  
[ $? -ne 0 ] && exit 1

# swanlab
project_name='verl_process_dapo_adv4'
# !!!! 实验名必须以 verl- 开头，不能有下划线
exp_name='verl-grpo-octothinker-complex-template'
# Paths
ckpts_dir="/nvme/liaochenyang/checkpoints_verl/process_adv_4/octothinker_3b_short_zero_complex_template/dapo_math/GRPO/${exp_name}"
prefix="/nvme/liaochenyang/checkpoints_verl/"
rel="${ckpts_dir#${prefix}}"        # 去掉前缀，得到相对路径
export WANDB_DIR="./wandb/${rel}"
mkdir -p "$WANDB_DIR" 
export WANDB_MODE="offline"
timeline_json_file="${ckpts_dir}/time_line_file.json"
tarin_rollout_dir="${ckpts_dir}/train_rollout"
val_data_dir="${ckpts_dir}/val_data"

python3 -m verl.trainer.main_ppo \
    data.train_files=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/liaochenyang/shared_datas/verl_data/dapo_math/dapo-math-17k-800token-octothinker.parquet \
    data.val_files=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/liaochenyang/shared_datas/verl_data/skyworkmath/test_500_octothinker.parquet \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/models/OctoThinker/3B/OctoThinker-3B-Short-Zero-complex-template \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=10 \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.rollout_data_dir=${tarin_rollout_dir} \
    trainer.validation_data_dir=${val_data_dir} \
    ray_init.timeline_json_file=${timeline_json_file} \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    trainer.val_before_train=False \
    custom_reward_function.path=/mnt/shared-storage-user/shared-storage-ailab-llmfudan/liaochenyang/verl-0.5.0/skyworkmath.py

$CFSCTL -p $PARTITION -n $NODE_COUNT -X $MASTER_ADDR -s $CFG stop