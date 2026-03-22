"""
Re-tokenize the FineWeb dataset using a custom SentencePiece model and write
output in the same .bin shard format as the standard sp1024 pipeline.

This produces ./data/datasets/fineweb10B_semantic1024/ which can be used
as DATA_PATH for the controlled experiment (Run B).

The .bin format:
  - 256 x int32 little-endian header
    - header[0] = 20240520  (magic)
    - header[1] = 1         (version)
    - header[2] = num_tokens
  - num_tokens x uint16 little-endian token IDs

Usage:
    python3 scripts/retokenize_fineweb.py \
        --tokenizer ./data/tokenizers/semantic_1024_bpe.model \
        --source-dir ./data/datasets/fineweb10B_sp1024 \
        --output-dir ./data/datasets/fineweb10B_semantic1024 \
        --train-shards 10 \
        --base-tokenizer ./data/tokenizers/fineweb_1024_bpe.model
"""

import argparse
import glob
import os
import struct
from pathlib import Path

import numpy as np
import sentencepiece as spm

REPO_ROOT = Path(__file__).resolve().parent.parent
HEADER_INTS = 256
HEADER_BYTES = HEADER_INTS * 4
MAGIC = 20240520
VERSION = 1


def read_bin_shard(path: str) -> np.ndarray:
    """Read token IDs from a .bin shard file (uint16 LE)."""
    header = np.fromfile(path, dtype="<i4", count=HEADER_INTS)
    if int(header[0]) != MAGIC or int(header[1]) != VERSION:
        raise ValueError(f"Bad header in {path}")
    num_tokens = int(header[2])
    return np.fromfile(path, dtype="<u2", count=num_tokens, offset=HEADER_BYTES)


def write_bin_shard(path: str, tokens: np.ndarray) -> None:
    """Write token IDs as a .bin shard file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    header = np.zeros(HEADER_INTS, dtype="<i4")
    header[0] = MAGIC
    header[1] = VERSION
    header[2] = len(tokens)
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.astype("<u2").tobytes())


def retokenize_shard(
    src_path: str,
    dst_path: str,
    base_sp: spm.SentencePieceProcessor,
    new_sp: spm.SentencePieceProcessor,
    chunk_size: int = 512,
) -> int:
    """
    Read a shard tokenized with base_sp, decode to text, re-encode with new_sp,
    and write to dst_path. Returns number of output tokens.
    """
    if Path(dst_path).exists():
        print(f"  skip (exists): {dst_path}")
        return 0

    src_tokens = read_bin_shard(src_path)
    output_tokens = []

    # Process in chunks to reconstruct approximate document boundaries
    for start in range(0, len(src_tokens), chunk_size):
        chunk = src_tokens[start : start + chunk_size].tolist()
        text = base_sp.decode(chunk)
        if text.strip():
            new_ids = new_sp.encode(text, out_type=int)
            output_tokens.extend(new_ids)

    out_array = np.array(output_tokens, dtype=np.uint16)
    write_bin_shard(dst_path, out_array)
    return len(out_array)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-tokenize FineWeb shards with a custom tokenizer")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=str(REPO_ROOT / "data" / "tokenizers" / "semantic_1024_bpe.model"),
        help="Path to the new .model file",
    )
    parser.add_argument(
        "--base-tokenizer",
        type=str,
        default=str(REPO_ROOT / "data" / "tokenizers" / "fineweb_1024_bpe.model"),
        help="Path to the base sp1024 .model used to decode the source shards",
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "datasets" / "fineweb10B_sp1024"),
        help="Directory containing the source .bin shards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(REPO_ROOT / "data" / "datasets" / "fineweb10B_semantic1024"),
        help="Directory to write re-tokenized shards",
    )
    parser.add_argument("--train-shards", type=int, default=10, help="Number of train shards to process")
    args = parser.parse_args()

    for p in (args.tokenizer, args.base_tokenizer):
        if not Path(p).exists():
            raise FileNotFoundError(f"Not found: {p}")

    print(f"Loading base tokenizer: {args.base_tokenizer}")
    base_sp = spm.SentencePieceProcessor()
    base_sp.load(args.base_tokenizer)

    print(f"Loading new tokenizer: {args.tokenizer}")
    new_sp = spm.SentencePieceProcessor()
    new_sp.load(args.tokenizer)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Validation shards
    val_files = sorted(glob.glob(os.path.join(args.source_dir, "fineweb_val_*.bin")))
    print(f"Re-tokenizing {len(val_files)} validation shard(s) ...")
    for src in val_files:
        shard_name = Path(src).name
        dst = os.path.join(args.output_dir, shard_name)
        n = retokenize_shard(src, dst, base_sp, new_sp)
        if n:
            print(f"  {shard_name}: {n:,} tokens")

    # Training shards
    train_files = sorted(glob.glob(os.path.join(args.source_dir, "fineweb_train_*.bin")))
    if args.train_shards > 0:
        train_files = train_files[: args.train_shards]
    print(f"Re-tokenizing {len(train_files)} training shard(s) ...")
    for src in train_files:
        shard_name = Path(src).name
        dst = os.path.join(args.output_dir, shard_name)
        n = retokenize_shard(src, dst, base_sp, new_sp)
        if n:
            print(f"  {shard_name}: {n:,} tokens")

    print(f"\nDone. Output written to: {args.output_dir}")
    out_files = list(Path(args.output_dir).glob("*.bin"))
    print(f"Total shards written: {len(out_files)}")


if __name__ == "__main__":
    main()
