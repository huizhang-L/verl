# Tested successfully on the hiyouga/verl:ngc-th2.6.0-cu126-vllm0.8.4-flashinfer0.2.2-cxx11abi0 image.
# It outperforms the Qwen2 7B base model by two percentage points on the test set of GSM8K.

set -x
ray job submit --address="http://172.30.33.109:8265" \
    --runtime-env=verl/trainer/runtime_env.yaml \
    --no-wait \
    -- \
    python3 -m verl.trainer.main_ppo \
        data.train_files=/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/data/skyworkmath/train_500_100.parquet \
        data.val_files=/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/data/skyworkmath/test_500_100.parquet \
        data.train_batch_size=128 \
        data.max_prompt_length=512 \
        data.max_response_length=8192 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        actor_rollout_ref.model.path=/fs-computility/llm_fudan/shared/models/Qwen3/Qwen3-0.6B \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=64 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
        actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
        actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
        actor_rollout_ref.rollout.val_kwargs.n=1 \
        actor_rollout_ref.rollout.val_kwargs.do_sample=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        algorithm.adv_estimator=grpo \
        trainer.critic_warmup=0 \
        trainer.logger='["console", "wandb"]' \
        trainer.project_name=verl \
        trainer.experiment_name=test \
        trainer.default_local_dir=/fs-computility/llm_fudan/lvhuijie/my_git_repo/verl/checkpoints/test/qwen3-0.6b/test/GRPO/test \
        trainer.n_gpus_per_node=2 \
        trainer.nnodes=1 \
        trainer.save_freq=20 \
        trainer.test_freq=5 \
        trainer.total_epochs=3 $@