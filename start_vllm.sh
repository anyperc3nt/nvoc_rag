#!/bin/bash

# Упорядочиваем карты по шине PCI (как они стоят в слотах)
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Теперь 0 — это точно первая карта A6000
export CUDA_VISIBLE_DEVICES=0

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
# MODEL_NAME="Qwen/Qwen2.5-14B-Instruct"


python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --port 8000 \
    --host 0.0.0.0