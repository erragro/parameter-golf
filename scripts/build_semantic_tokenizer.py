"""
Build a semantically-biased BPE tokenizer by downsampling function words during training.

The hypothesis: high-frequency function words (the, is, a, ...) consume vocabulary slots
but carry low semantic weight. By downsampling them at tokenizer-training time, the 1024-
slot vocabulary is biased toward content words. BigramHash combinations then become more
semantically rich.

Usage:
    python3 scripts/build_semantic_tokenizer.py [--n-texts N] [--downsample-rate R]

Output:
    ./data/tokenizers/semantic_1024_bpe.model
    ./data/tokenizers/semantic_1024_bpe.vocab
"""

import argparse
import glob
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
TOKENIZERS_DIR = REPO_ROOT / "data" / "tokenizers"

# Function words targeted for downsampling per the deployment spec
FUNCTION_WORDS = {
    # Articles
    "the", "a", "an",
    # Auxiliaries
    "is", "was", "were", "are", "be", "been", "being",
    "has", "have", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might",
    # High-frequency prepositions
    "in", "on", "at", "to", "for", "of", "with", "by",
    # Conjunctions
    "and", "or", "but", "so", "yet",
    # Filler
    "very", "really", "just", "quite", "rather",
}


def downsample_function_words(text: str, rate: float = 0.3) -> str:
    """Keep only `rate` fraction of function word occurrences."""
    words = text.split()
    result = []
    for word in words:
        clean = word.lower().strip('.,!?;:"\'`')
        if clean in FUNCTION_WORDS:
            if random.random() < rate:
                result.append(word)
        else:
            result.append(word)
    return " ".join(result)


def load_texts_from_bin_shards(data_dir: str, n_texts: int, sp_model_path: str) -> list[str]:
    """
    Read token IDs from existing .bin shards, decode back to text using the
    standard sp1024 tokenizer, and return raw text strings.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(sp_model_path)

    HEADER_INTS = 256
    HEADER_BYTES = HEADER_INTS * 4  # int32
    MAGIC = 20240520

    files = sorted(glob.glob(os.path.join(data_dir, "fineweb_train_*.bin")))
    if not files:
        raise FileNotFoundError(f"No training shards found in {data_dir}")

    texts = []
    tokens_buffer = []
    TOKENS_PER_DOC_APPROX = 512  # approximate; we split on boundary tokens

    for fpath in files:
        if len(texts) >= n_texts:
            break
        header = np.fromfile(fpath, dtype="<i4", count=HEADER_INTS)
        if int(header[0]) != MAGIC or int(header[1]) != 1:
            raise ValueError(f"Unexpected shard header in {fpath}")
        num_tokens = int(header[2])
        token_ids = np.fromfile(fpath, dtype="<u2", count=num_tokens, offset=HEADER_BYTES)

        # Decode chunks of ~512 tokens as pseudo-documents
        chunk_size = TOKENS_PER_DOC_APPROX
        for start in range(0, len(token_ids), chunk_size):
            if len(texts) >= n_texts:
                break
            chunk = token_ids[start : start + chunk_size].tolist()
            text = sp.decode(chunk)
            if text.strip():
                texts.append(text)

    print(f"Loaded {len(texts)} text chunks from {len(files)} shard(s)")
    return texts


def build_semantic_tokenizer(
    output_prefix: str,
    vocab_size: int = 1024,
    n_texts: int = 5_000_000,
    downsample_rate: float = 0.3,
    data_dir: str | None = None,
    sp_model_path: str | None = None,
) -> None:
    if data_dir is None:
        data_dir = str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024")
    if sp_model_path is None:
        sp_model_path = str(TOKENIZERS_DIR / "fineweb_1024_bpe.model")

    if not Path(sp_model_path).exists():
        raise FileNotFoundError(
            f"Base tokenizer not found at {sp_model_path}. "
            "Run: python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1"
        )

    print(f"Loading up to {n_texts:,} text chunks from {data_dir} ...")
    texts = load_texts_from_bin_shards(data_dir, n_texts, sp_model_path)

    print(f"Applying function-word downsampling (keep rate={downsample_rate}) ...")
    processed = [downsample_function_words(t, rate=downsample_rate) for t in texts]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
        for text in processed:
            if text.strip():
                f.write(text + "\n")
        temp_path = f.name

    print(f"Training SentencePiece BPE on {len(processed):,} lines -> {output_prefix} ...")
    spm.SentencePieceTrainer.train(
        input=temp_path,
        model_prefix=output_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=0.9995,
        max_sentencepiece_length=16,
        split_digits=True,
    )
    os.unlink(temp_path)
    print(f"Done. Saved: {output_prefix}.model  {output_prefix}.vocab")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build semantic BPE tokenizer")
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--n-texts", type=int, default=5_000_000, help="Max text chunks to use")
    parser.add_argument("--downsample-rate", type=float, default=0.3, help="Keep fraction for function words")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to fineweb10B_sp1024 shards")
    parser.add_argument("--sp-model", type=str, default=None, help="Path to base sp1024 .model file")
    parser.add_argument(
        "--output",
        type=str,
        default=str(TOKENIZERS_DIR / "semantic_1024_bpe"),
        help="Output prefix (without .model extension)",
    )
    args = parser.parse_args()

    TOKENIZERS_DIR.mkdir(parents=True, exist_ok=True)

    build_semantic_tokenizer(
        output_prefix=args.output,
        vocab_size=args.vocab_size,
        n_texts=args.n_texts,
        downsample_rate=args.downsample_rate,
        data_dir=args.data_dir,
        sp_model_path=args.sp_model,
    )


if __name__ == "__main__":
    main()
