#!/usr/bin/env bash
set -xeuo pipefail

# Ray Cluster
nnodes=1
n_gpus_per_node=8


# Data
max_prompt_length=$((800))
max_response_length=$((1024 * 8))


# Rollout
n_resp_per_prompt=8
temperature=0.6
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Actor Update Batch
# ppo_mini_batch_size=$((nnodes * n_gpus_per_node * 4))
ppo_mini_batch_size=64
total_epochs=10


# DAPO Clip-Higher
clip_ratio_low=0.2
clip_ratio_high=0.28


# DAPO Dynamic Sampling
# train_prompt_bsz=$((32 * ppo_mini_batch_size / n_resp_per_prompt))
train_prompt_bsz=8
gen_prompt_bsz=2
# 每次 rollout batch 是 gen_prompt_bsz，直到凑够 train_prompt_bsz 的数据
# 如果不够，就继续 rollout，最多 rollout max_num_gen_batches 个 batch
enable_filter_groups=True
filter_groups_metric="seq_outcome_reward"
max_num_gen_batches=100


# DAPO Token-Level Loss
loss_agg_mode="token-mean"


# DAPO Overlong Reward Shaping
enable_rm=False
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 1)) # 该值一定要小于 max_response_length
overlong_penalty_factor=1.0


# Custom Reward Score
enable_llm_process_reward=False
enable_llm_process_critique=True
enable_process_reward_model_score=False
enable_think=True

# Updata Actor
actor_lr=1e-6
actor_lr_warmup_steps=0


# GPU Memory Related Paremeter
# 序列并行的值，必须能被 num_attention_heads 整除
sp_size=4
use_dynamic_bsz=True
actor_ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len_per_gpu=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1
use_remove_padding=True
enable_activation_offload=False

# WANDB
project_name='verl_process_dapo_adv2'
exp_name='dapo-8k-train_bsz8-ppo_bsz64-rollout_n8-critique-reflection-nostepadv-nosteptopn'
# Paths
ckpts_dir="/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/checkpoints_verl/process_adv_2/qwen2.5-7b-instruct/dapo_math/DAPO/${exp_name}"
timeline_json_file="${ckpts_dir}/time_line_file.json"
reward_model_path="/fs-computility/llm_fudan/shared/models/Qwen2.5-Math/Qwen2.5-Math-PRM-7B"
tarin_rollout_dir="${ckpts_dir}/train_rollout"
val_data_dir="${ckpts_dir}/val_data"

# ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
#     --working-dir "${WORKING_DIR}" \
#     -- 
python3 -m recipe.process_dapo.main_process_dapo \
    data.train_files="/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/data/dapo_math/dapo-math-17k-800token.parquet" \
    data.val_files="/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/data/skyworkmath/test_500_100.parquet" \
    data.prompt_key="prompt" \
    data.truncation="middle" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_input_ids=False \
    data.return_raw_chat=True \
    data.return_full_prompt=False \
    data.shuffle=True \
    data.filter_overlong_prompts=False \
    data.trust_remote_code=True \
    algorithm.adv_estimator="process_grpo" \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path="/fs-computility/llm_fudan/shared/models/Qwen2.5/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=${use_remove_padding} \
    actor_rollout_ref.model.enable_activation_offload=${enable_activation_offload} \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend="torch" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
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
    actor_rollout_ref.actor.optim.lr=${actor_lr} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=${actor_lr_warmup_steps} \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.process_grpo_adv.coef_bad_step_if_adv_pos=1.0 \
    actor_rollout_ref.actor.process_grpo_adv.coef_good_step_if_adv_neg=1.0 \
    actor_rollout_ref.actor.process_grpo_adv.top_n=0 \
    actor_rollout_ref.actor.process_grpo_adv.topn_mode="scale" \
    actor_rollout_ref.actor.process_grpo_adv.topn_scale=1.3 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.engine_kwargs.sglang="flashinfer" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_seqs=${gen_prompt_bsz} \
    actor_rollout_ref.rollout.do_sample=True \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    reward_model.enable=${enable_rm} \
    reward_model.reward_manager="process_dapo" \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    custom_reward_function.path="/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/verl/utils/reward_score/hybrid_score.py" \
    custom_reward_function.name="compute_score" \
    custom_reward_function.llm_process_reward.enable=${enable_llm_process_reward} \
    custom_reward_function.llm_process_reward.temperature=0.1 \
    custom_reward_function.llm_process_reward.max_tokens=512 \
    custom_reward_function.llm_process_reward.concurrency=32 \
    custom_reward_function.llm_process_reward.coefficient=0.5 \
    custom_reward_function.llm_process_reward.split_step_num=5 \
    custom_reward_function.llm_process_reward.enable_think=${enable_think}  \
    custom_reward_function.process_reward_model.enable=${enable_process_reward_model_score} \
    custom_reward_function.process_reward_model.temperature=0.1 \
    custom_reward_function.process_reward_model.max_tokens=512 \
    custom_reward_function.process_reward_model.concurrency=1 \
    custom_reward_function.process_reward_model.coefficient=0.5 \
    custom_reward_function.llm_process_critique.enable=${enable_llm_process_critique} \
    custom_reward_function.llm_process_critique.temperature=0.1 \
    custom_reward_function.llm_process_critique.max_tokens=8192 \
    custom_reward_function.llm_process_critique.concurrency=32 \
    custom_reward_function.llm_process_critique.coefficient=0.5 \
    custom_reward_function.llm_process_critique.split_step_num=1 \
    custom_reward_function.llm_process_critique.enable_think=${enable_think} \
    custom_reward_function.reflection.enable=${enable_filter_groups} \
    custom_reward_function.reflection.temperature=0.1 \
    custom_reward_function.reflection.max_tokens=8192 \
    custom_reward_function.reflection.concurrency=32 \
    custom_reward_function.llm_process_critique.enable_think=${enable_think} \
    trainer.logger='["console","swanlab"]' \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${exp_name} \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${nnodes} \
    trainer.val_before_train=False \
    trainer.test_freq=100 \
    trainer.save_freq=20 \
    trainer.total_epochs=${total_epochs} \
    trainer.default_local_dir=${ckpts_dir} \
    trainer.resume_mode=auto \
    trainer.balance_batch=True \
    trainer.rollout_data_dir=${tarin_rollout_dir} \
    trainer.validation_data_dir=${val_data_dir} \
    ray_init.timeline_json_file=${timeline_json_file}
