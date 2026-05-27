#!/bin/bash

set -e

echo "========================================"
echo "vLLM Server Launch"
echo "Model: Qwen2.5-32B-Instruct (FP8)"
echo "Purpose: Namo-Namah KG Triplet Extraction"
echo "========================================"

# ---------------- ENV ----------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2,3          # No space — space here breaks CUDA device parsing
export NCCL_DEBUG=WARN                   # INFO is very noisy in production; use WARN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# DO NOT set CUDA_LAUNCH_BLOCKING=1 in production — serializes all CUDA ops
# Uncomment only when debugging a CUDA error:
# export CUDA_LAUNCH_BLOCKING=1

echo -e "\nChecking CUDA availability..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

echo -e "\nGPU Configuration:"
echo "  CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"


# ---------------- VERSION CHECK ----------------
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, BF16 support: {torch.cuda.is_bf16_supported()}')"


# ---------------- RUN ----------------
HF_TOKEN=$(cat ./hf_token_namo_namah.txt) \
vllm serve Qwen/Qwen2.5-14B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 16384 \
  --dtype bfloat16 \
  --enable-chunked-prefill \
  --max-num-seqs 32 \
  --host 0.0.0.0 \
  --port 8001 \
  2>&1 | tee error.log