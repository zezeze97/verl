#!/bin/bash
set -xeuo pipefail

project_name='GRPO-GroundCUA'
exp_name='GRPO-Qwen3_vl-8B-GroundCUA-npu-8card'

# Single-node 8-card starter config for Qwen3-VL-8B on Ascend NPU.
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export WANDB_MODE=offline

NNODES=${NNODES:-1}
NPUS_PER_NODE=${NPUS_PER_NODE:-8}
gen_tp=${GEN_TP:-1}
sp_size=${SP_SIZE:-1}
actor_fsdp_size=${ACTOR_FSDP_SIZE:-${NPUS_PER_NODE}}
ENGINE=${1:-vllm}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REWARD_FN_PATH=${REWARD_FN_PATH:-"${SCRIPT_DIR}/reward_fn_point_in_box.py"}

WORK_ROOT=${WORK_ROOT:-/home/ma-user/work/preliminary_gui/z00967441}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${WORK_ROOT}/verl"}
#MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-VL-8B-Instruct"}
#MODEL_PATH=${WORK_ROOT}/model_ckpts/UI-Voyager
MODEL_PATH=${MODEL_PATH:-"${WORK_ROOT}/model_ckpts/Qwen3-VL-8B-Instruct"}

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
DATA_ROOT=${DATA_ROOT:-"${RAY_DATA_HOME}/data/GroundCUA"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/train.parquet"}
TEST_FILE=${TEST_FILE:-"${DATA_ROOT}/valid.parquet"}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${RAY_DATA_HOME}/tensorboard_dir/${project_name}/${exp_name}"}
max_ckpt_to_keep=${MAX_CKPT_TO_KEEP:-1}

max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
# Qwen3-VL advertises a very large native context window. Keep vLLM aligned with
# the training lengths instead of reserving KV cache for the full 262144 tokens.
max_model_len=${MAX_MODEL_LEN:-4096}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-2}
filter_overlong_prompts_workers=${FILTER_OVERLONG_PROMPTS_WORKERS:-1}

# Rollout correction parameters
rollout_is=${ROLLOUT_IS:-sequence}
rollout_is_threshold=${ROLLOUT_IS_THRESHOLD:-2.0}
rollout_is_batch_normalize=${ROLLOUT_IS_BATCH_NORMALIZE:-true}
rollout_rs=${ROLLOUT_RS:-token_k1}
rollout_rs_threshold=${ROLLOUT_RS_THRESHOLD:-0.6_1.6}

extra_rollout_args=()
# Older vLLM builds do not recognize --disable-mm-preprocessor-cache, so keep it opt-in.
if [[ "${DISABLE_MM_PREPROCESSOR_CACHE:-0}" == "1" ]]; then
    extra_rollout_args+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True")
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.train_batch_size=64 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.return_raw_chat=True \
    data.shuffle=False \
    data.dataloader_num_workers=${dataloader_num_workers} \
    data.filter_overlong_prompts_workers=${filter_overlong_prompts_workers} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model','optimizer','extra']" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${actor_fsdp_size} \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.max_model_len=${max_model_len} \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold} \
    reward.custom_reward_function.path="${REWARD_FN_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=${NPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.max_actor_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.max_critic_ckpt_to_keep=${max_ckpt_to_keep} \
    trainer.val_before_train=True \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    "${extra_rollout_args[@]}" \
    "$@"
