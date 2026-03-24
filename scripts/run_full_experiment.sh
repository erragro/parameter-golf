#!/bin/bash
set -e

cd /workspace/parameter-golf

echo "=== Step 1: Install deps ==="
pip install huggingface_hub sentencepiece -q

echo "=== Step 2: Download FineWeb sp1024 (10 shards) ==="
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10

echo "=== Step 3: Retokenize with semantic tokenizer ==="
python3 scripts/retokenize_fineweb.py

echo "=== Step 4: Run A — Control (sp1024 tokenizer, SOTA config) ==="
mkdir -p logs
RUN_ID=experiment_control \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/sota_repro/train_gpt.py 2>&1 | tee logs/experiment_control.log

echo "=== Step 5: Run B — Semantic BPE tokenizer, SOTA config ==="
RUN_ID=experiment_semantic \
DATA_PATH=./data/datasets/fineweb10B_semantic1024/ \
TOKENIZER_PATH=./data/tokenizers/semantic_1024_bpe.model \
VOCAB_SIZE=1024 \
torchrun --standalone --nproc_per_node=8 \
    records/sota_repro/train_gpt.py 2>&1 | tee logs/experiment_semantic.log

echo "=== Step 6: Compare ==="
python3 scripts/compare_runs.py
