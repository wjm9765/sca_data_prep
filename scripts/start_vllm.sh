#!/bin/bash

MODEL_NAME=${1:-"Qwen/Qwen3-Omni-30B-A3B-Captioner"}
export HF_HOME="/workspace/models"
export MODEL_NAME="${HF_HOME}/Qwen3-Omni-30B-A3B-Captioner"

sudo apt update && sudo apt install -y git ffmpeg btop nvtop screen

git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
cd vllm
pip install -r requirements/build.txt
pip install -r requirements/cuda.txt
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
VLLM_USE_PRECOMPILED=1 pip install -e . -v --no-build-isolation
# If you meet an "Undefined symbol" error while using VLLM_USE_PRECOMPILED=1, please use "pip install -e . -v" to build from source.
# Install the Transformers
pip install git+https://github.com/huggingface/transformers
pip install accelerate
pip install qwen-omni-utils -U
pip install -U flash-attn --no-build-isolation

NCCL_P2P_DISABLE=1 vllm serve "${MODEL_NAME}" \
    --port 8000 \
    --host 0.0.0.0 \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --max-model-len 32768 \
    --compilation-config '{"level": 3, "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128], "max_capture_size": 16384}' \
    -tp 4 \
    --dtype bfloat16
