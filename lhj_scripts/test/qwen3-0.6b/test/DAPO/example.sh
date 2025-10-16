#!/usr/bin/env bash
set -xeuo pipefail

# Data
max_prompt_length=$((512 * 1))
max_response_length=$((1024 * 8))


# Algorithm DAPO
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
# Loss
use_kl_loss=False
kl_loss_coef=0.0
# Updata Actor
ppo_mini_batch_size=32
total_epochs=1


# DAPO Clip-Higher
clip_ratio_low=0.2
clip_ratio_high=0.28


# DAPO Dynamic Sampling
train_prompt_bsz=128
gen_prompt_bsz=$((train_prompt_bsz * 2))

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10


# DAPO Token-Level Loss
loss_agg_mode="token-mean"


# DAPO Overlong Reward Shaping
enable_overlong_buffer=True
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0


# Rollout
n_resp_per_prompt=16
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7


# Ray Cluster
nnodes=1
n_gpus_per_node=8


# GPU Memory Related Paremeter
sp_size=8
use_dynamic_bsz=True
actor_ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1



# WANDB
project_name='verl_dapo'
exp_name='DAPO-Qwen3-0.6B'
# Paths
ckpts_dir="/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/checkpoints_verl/test/qwen3-0.6b/test/DAPO/${exp_name}"
timeline_json_file="${ckpts_dir}/time_line_file.json"

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- 
python3 -m recipe.dapo.main_dapo \
    data.train_files=/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/data/gsm8k/train.parquet \
    data.val_files=/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/data/gsm8k/test.parquet \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_input_ids=False \
    data.return_raw_chat=False \
    data.return_full_prompt=False \
    data.shuffle=True \
    data.filter_overlong_prompts=False \
    data.trust_remote_code=True \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="/fs-computility/llm_fudan/shared/models/Qwen3/Qwen3-0.6B" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend="torch" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.engine_kwargs.sglang="flashinfer" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.val_before_train=True \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.resume_mode=auto \
    trainer.balance_batch=True \
    ray_init.timeline_json_file=${timeline_json_file}
