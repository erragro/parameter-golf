# Semantic BPE Tokenizer — Function Word Downsampling

**Author:** Surajit Chaudhuri (@erragro)
**Date:** 2026-03-24
**Track:** Non-record (model slightly exceeds 16MB; semantic experiment in progress)
**Baseline val_bpb:** 1.1619 (pre-quant, step 5478, 8xH100 SXM, 600s wallclock)

---

## Hypothesis

Standard BPE tokenization allocates significant vocabulary capacity to grammatical function words (*the*, *a*, *is*, *was*, *and*, etc.), which carry low semantic entropy. The hypothesis is:

> **Downsampling function words during BPE training forces the vocabulary toward content-bearing tokens, making BigramHash bigram features more informative — yielding lower bits-per-byte.**

This is orthogonal to all existing leaderboard approaches (QAT, BigramHash, SWA, architecture changes) and can be stacked on top of the current SOTA config.

---

## Method

### 1. Semantic Tokenizer Training

A custom SentencePiece BPE model (`semantic_1024_bpe.model`) is trained on FineWeb with function word downsampling:

```python
FUNCTION_WORDS = {'the','a','an','is','was','were','are','be','been','being',
    'has','have','had','do','does','did','will','would','could','should',
    'may','might','in','on','at','to','for','of','with','by',
    'and','or','but','so','yet','very','really','just','quite','rather'}
DOWNSAMPLE_RATE = 0.3  # keep 30% of function word occurrences
```

Training args: `model_type='bpe'`, `vocab_size=1024`, `character_coverage=0.9995`.

### 2. Controlled A/B Experiment

- **Run A (control):** SOTA config (Int6 QAT + BigramHash + SWA + 10L) with standard `sp1024` tokenizer
- **Run B (semantic):** Same SOTA config with `semantic_1024_bpe` tokenizer on re-tokenized FineWeb

Both runs use identical architecture, hyperparameters, and random seed. Only the tokenizer differs.

### 3. PyTorch 2.4 Compatibility Fix

The SOTA script used `enable_gqa` in `F.scaled_dot_product_attention()`, which was added in PyTorch 2.5. Fixed by manually expanding KV heads:

```python
if self.num_kv_heads != self.num_heads:
    k = k.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
    v = v.repeat_interleave(self.num_heads // self.num_kv_heads, dim=1)
y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=True)
```

---

## Results

| Run | val_bpb | Notes |
|-----|---------|-------|
| Baseline repro (sp1024, SOTA config) | **1.1619** | Pre-quant, step 5478, 600s on 8xH100 |
| Semantic BPE experiment | *in progress* | Full A/B results pending |

---

## Reproducing

```bash
git clone https://github.com/erragro/parameter-golf.git
cd parameter-golf
bash scripts/run_full_experiment.sh
```

This downloads data, builds the semantic dataset, runs both experiments, and prints the delta.

---

## Files

| File | Description |
|------|-------------|
| `train_gpt.py` | SOTA repro script with PyTorch 2.4 fix |
| `scripts/build_semantic_tokenizer.py` | Builds `semantic_1024_bpe.model` |
| `scripts/retokenize_fineweb.py` | Re-tokenizes FineWeb with semantic tokenizer |
| `scripts/run_full_experiment.sh` | One-command full experiment runner |
| `data/tokenizers/semantic_1024_bpe.model` | Pre-built semantic tokenizer |
