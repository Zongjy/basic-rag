#!/bin/bash

python eval_multihop_rag.py \
    --output-dir ./multihop_rag_output \
    --embedding-api-url http://localhost:30000/v1/embeddings \
    --embedding-model Alibaba-NLP/gte-Qwen2-7B-instruct \
    --chunk-size 1024 \
    --chunk-overlap 100 \
    --top-k 15
