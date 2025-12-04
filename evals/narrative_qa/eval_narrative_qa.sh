#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python eval_narrative_qa.py \
    --output-dir ./narrative_qa_1024_15 \
    --embedding-api-url http://localhost:30000/v1/embeddings \
    --embedding-model Alibaba-NLP/gte-Qwen2-7B-instruct \
    --chunk-size 1024 \
    --chunk-overlap 100 \
    --top-k 15 \
    --split test
