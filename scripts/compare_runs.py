"""
Compare val_bpb between the control (standard sp1024) and semantic BPE runs.

Usage:
    python3 scripts/compare_runs.py [--control PATH] [--semantic PATH]

Defaults to:
    logs/experiment_control.log
    logs/experiment_semantic.log
"""

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_log(path: str) -> dict:
    with open(path) as f:
        content = f.read()
    bpb  = re.search(r"val_bpb[:\s]+([\d.]+)", content)
    loss = re.search(r"val_loss[:\s]+([\d.]+)", content)
    size = re.search(r"compressed.*?(\d+)\s*bytes", content)
    return {
        "val_bpb":  float(bpb.group(1))  if bpb  else None,
        "val_loss": float(loss.group(1)) if loss else None,
        "bytes":    int(size.group(1))   if size else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare control vs semantic BPE run logs")
    parser.add_argument(
        "--control",
        default=str(REPO_ROOT / "logs" / "experiment_control.log"),
        help="Path to control run log",
    )
    parser.add_argument(
        "--semantic",
        default=str(REPO_ROOT / "logs" / "experiment_semantic.log"),
        help="Path to semantic BPE run log",
    )
    args = parser.parse_args()

    for label, path in [("Control", args.control), ("Semantic", args.semantic)]:
        if not Path(path).exists():
            print(f"[{label}] Log not found: {path}")

    control  = parse_log(args.control)
    semantic = parse_log(args.semantic)

    print(f"Control:  {control['val_bpb']:.4f}  (val_loss={control['val_loss']}, bytes={control['bytes']})")
    print(f"Semantic: {semantic['val_bpb']:.4f}  (val_loss={semantic['val_loss']}, bytes={semantic['bytes']})")

    if control["val_bpb"] is not None and semantic["val_bpb"] is not None:
        delta = control["val_bpb"] - semantic["val_bpb"]
        print(f"Delta:    {delta:+.4f}  (positive = semantic is better)")

        SIGNIFICANCE_THRESHOLD = 0.005
        if delta > SIGNIFICANCE_THRESHOLD:
            print(f"\n✓ Signal found (delta > {SIGNIFICANCE_THRESHOLD}). Run 3x for p<0.01, then submit PR.")
        elif 0 < delta <= SIGNIFICANCE_THRESHOLD:
            print(f"\n~ Marginal result. Try higher downsample rate (0.1). Run 5x.")
        elif abs(delta) <= 0.002:
            print(f"\n→ Neutral. BigramHash may already handle it. Try structured compression markers.")
        else:
            print(f"\n✗ Grammar carries signal. Pivot to syntax-aware tokenizer.")


if __name__ == "__main__":
    main()
