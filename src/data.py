"""Download and load the MedHallu dataset.

The published parquet files total ~11 MB, so we fetch them straight over HTTP
instead of pulling in the `datasets` library. Cached locally after first run.
"""
import argparse
import io

import pandas as pd
import requests

from config import CONFIGS, DATA_DIR, HF_REPO
from config import (COL_CATEGORY, COL_DIFFICULTY, COL_HALLU, COL_KNOWLEDGE,
                    COL_QUESTION, COL_TRUTH)

PARQUET_INDEX = f"https://huggingface.co/api/datasets/{HF_REPO}/parquet"


def download(config: str, force: bool = False) -> pd.DataFrame:
    local = DATA_DIR / f"{config}.parquet"
    if local.exists() and not force:
        return pd.read_parquet(local)

    index = requests.get(PARQUET_INDEX, timeout=60).json()
    if config not in index:
        raise SystemExit(f"config {config!r} not in {list(index)}")

    frames = []
    for url in index[config]["train"]:
        resp = requests.get(url, timeout=300)
        resp.raise_for_status()
        frames.append(pd.read_parquet(io.BytesIO(resp.content)))

    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(local, index=False)
    return df


def load(config: str = "pqa_labeled") -> pd.DataFrame:
    return download(config)


def to_detection_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Explode each row into two labelled detection examples.

    The benchmark is binary: the model sees one answer at a time and decides
    whether it is hallucinated. Each source row therefore yields a positive
    (hallucinated, label 1) and a negative (ground truth, label 0) example.
    """
    def block(answer_col, label):
        out = pd.DataFrame({
            "question": df[COL_QUESTION],
            "knowledge": df[COL_KNOWLEDGE].map(_flatten_knowledge),
            "answer": df[answer_col],
            "label": label,
            "difficulty": df[COL_DIFFICULTY],
            "category": df[COL_CATEGORY],
            "source_row": df.index,
        })
        return out

    pairs = pd.concat([block(COL_HALLU, 1), block(COL_TRUTH, 0)], ignore_index=True)
    return pairs.sort_values(["source_row", "label"]).reset_index(drop=True)


def _flatten_knowledge(value) -> str:
    """`Knowledge` is a list of context sentences on the Hub; join to one string."""
    if isinstance(value, str):
        return value
    try:
        return " ".join(str(v) for v in value)
    except TypeError:
        return str(value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch and summarise MedHallu.")
    ap.add_argument("--config", default="pqa_labeled", choices=CONFIGS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    df = download(args.config, force=args.force)
    print(f"{args.config}: {len(df)} rows -> {DATA_DIR / (args.config + '.parquet')}")
    print("\ncolumns:", list(df.columns))
    print("\ndifficulty:\n", df[COL_DIFFICULTY].value_counts())
    print("\ncategory:\n", df[COL_CATEGORY].value_counts())

    ex = df.iloc[0]
    print("\n--- example row ---")
    print("Q :", str(ex[COL_QUESTION])[:200])
    print("GT:", str(ex[COL_TRUTH])[:200])
    print("H :", str(ex[COL_HALLU])[:200])
    print("  ", ex[COL_DIFFICULTY], "|", ex[COL_CATEGORY])
