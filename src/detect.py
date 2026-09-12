"""Run the MedHallu detection benchmark.

Reproduces the shape of the paper's headline results on a laptop:

  Table 2  overall / easy / medium / hard F1, by knowledge condition
  Table 4  effect of offering the model a "not sure" option

Three knowledge conditions, not the paper's two:

  none     the judge sees only the question and answer
  oracle   the judge gets the exact PubMed context the question came from.
           This is the paper's "with knowledge" setting -- perfect retrieval.
  rag      the judge gets the top-k passages an actual retriever fetched.
           See rag.py; this is the realistic middle case the paper skips.

The paper measures none (~0.53) and oracle (~0.78) and leaves the gap between
them unexplored. Running all three tells you how much of that +0.25 survives
imperfect retrieval.

Examples
--------
  python src/detect.py --backend constant --limit 300
  python src/detect.py --backend lexical --limit 300 --knowledge-mode none,oracle
  python src/detect.py --backend ollama:qwen2.5:1.5b-instruct --limit 100 \
      --knowledge-mode none,oracle,rag --retriever tfidf --k 3
"""
import argparse
import json
import time

import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

import data
from backends import get_backend
from config import CONFIGS, RESULTS_DIR

MODES = ("none", "oracle", "rag")


def score(preds, labels, allow_not_sure):
    """Precision / recall / F1 for the positive (hallucinated) class.

    With "not sure" enabled, abstentions are dropped before scoring and reported
    separately as `response_pct` -- the paper's Table 4 does the same, which is
    why precision can rise while coverage falls.
    """
    frame = pd.DataFrame({"pred": preds, "label": labels})
    total = len(frame)
    if allow_not_sure:
        frame = frame[frame["pred"] != 2]
    if frame.empty:
        return {"n": total, "response_pct": 0.0, "precision": float("nan"),
                "recall": float("nan"), "f1": float("nan")}
    return {
        "n": total,
        "response_pct": round(100 * len(frame) / total, 1),
        "precision": round(precision_score(frame["label"], frame["pred"], zero_division=0), 3),
        "recall": round(recall_score(frame["label"], frame["pred"], zero_division=0), 3),
        "f1": round(f1_score(frame["label"], frame["pred"], zero_division=0), 3),
    }


def evaluate(backend, pairs, knowledge_col, allow_not_sure, mode_label):
    preds = []
    desc = f"{backend.name} | {mode_label} | not_sure={allow_not_sure}"
    knowledge = pairs[knowledge_col] if knowledge_col else None

    for i, row in enumerate(tqdm(pairs.itertuples(index=False), total=len(pairs), desc=desc)):
        preds.append(backend.judge(
            question=row.question,
            answer=row.answer,
            knowledge=knowledge.iat[i] if knowledge is not None else None,
            allow_not_sure=allow_not_sure,
        ))

    out = pairs.copy()
    out["pred"] = preds

    # Difficulty labels a source row, and both of that row's examples carry it.
    # Slicing on the label keeps each difficulty balanced 1:1 the same way the
    # full set is -- otherwise precision within a slice is not comparable to
    # overall precision, or to the paper's per-difficulty columns.
    report = {"overall": score(out["pred"], out["label"], allow_not_sure)}
    for level in ("easy", "medium", "hard"):
        slice_ = out[out["difficulty"] == level]
        report[level] = score(slice_["pred"], slice_["label"], allow_not_sure)
    return report, out


def print_report(title, report):
    print(f"\n{title}")
    header = f"  {'slice':<9}{'n':>6}{'P':>8}{'R':>8}{'F1':>8}{'resp%':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in ("overall", "easy", "medium", "hard"):
        m = report[key]
        print(f"  {key:<9}{m['n']:>6}{m['precision']:>8}{m['recall']:>8}"
              f"{m['f1']:>8}{m['response_pct']:>8}")


def attach_rag(df, pairs, args):
    """Add a `knowledge_rag` column: the top-k passages a retriever fetched."""
    import rag  # imported lazily so non-RAG runs need no retrieval deps

    corpus = rag.build_corpus(df)
    n_gold = len(corpus)
    if args.corpus_extra:
        extra = rag.build_corpus(data.load(args.corpus_extra))
        corpus = corpus + extra
        print(f"rag corpus: {n_gold} gold + {len(extra)} distractors = {len(corpus)}")
    else:
        print(f"rag corpus: {len(corpus)} documents "
              "(no distractors -- retrieval will be near-perfect; "
              "use --corpus-extra for a realistic test)")

    retriever = rag.get_retriever(args.retriever, corpus)
    passages, ranked = rag.retrieve_knowledge(df, retriever, args.k, corpus)
    print(f"rag retriever: {retriever.name}   k={args.k}   "
          f"recall@{args.k}={rag.recall_at_k(ranked, args.k):.3f}")

    # pairs carry source_row, so the same retrieval serves both of a row's
    # examples without retrieving twice.
    by_row = dict(enumerate(passages))
    pairs["knowledge_rag"] = pairs["source_row"].map(by_row)
    return rag.recall_at_k(ranked, args.k)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", default="lexical",
                    help="constant | lexical | ollama:<model> | anthropic[:model] | openai[:model]")
    ap.add_argument("--config", default="pqa_labeled", choices=CONFIGS)
    ap.add_argument("--limit", type=int, default=200,
                    help="source rows to use; each yields 2 judgements")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--knowledge-mode", default="none,oracle",
                    help="comma list of: none, oracle, rag")
    ap.add_argument("--not-sure", action="store_true", help="offer the abstain option")
    ap.add_argument("--retriever", default="tfidf", help="rag only: tfidf | dense[:model]")
    ap.add_argument("--k", type=int, default=3, help="rag only: passages to retrieve")
    ap.add_argument("--corpus-extra", default=None,
                    help="rag only: another config to add as distractors, e.g. pqa_artificial")
    args = ap.parse_args()

    modes = [m.strip() for m in args.knowledge_mode.split(",") if m.strip()]
    bad = [m for m in modes if m not in MODES]
    if bad:
        raise SystemExit(f"unknown knowledge mode(s) {bad}; choose from {list(MODES)}")

    df = data.load(args.config)
    if args.limit and args.limit < len(df):
        df = df.sample(n=args.limit, random_state=args.seed)
    df = df.reset_index(drop=True)
    pairs = data.to_detection_pairs(df)

    backend = get_backend(args.backend)
    print(f"backend : {backend.name}")
    print(f"data    : {args.config}, {len(df)} rows -> {len(pairs)} judgements per pass")

    recall = None
    if "rag" in modes:
        recall = attach_rag(df, pairs, args)

    column = {"none": None, "oracle": "knowledge", "rag": "knowledge_rag"}
    results = {}
    for mode in modes:
        started = time.time()
        report, raw = evaluate(backend, pairs, column[mode], args.not_sure, mode)
        elapsed = time.time() - started
        results[mode] = report
        print_report(f"{mode}  ({elapsed:.1f}s, {elapsed / max(len(pairs), 1):.2f}s/judgement)",
                     report)
        raw.to_csv(RESULTS_DIR / f"raw_{_slug(backend.name)}_{mode}.csv", index=False)

    if len(modes) > 1:
        print(f"\n  {'condition':<10}{'F1':>8}{'vs none':>10}")
        print("  " + "-" * 28)
        base = results.get("none", {}).get("overall", {}).get("f1")
        for mode in modes:
            f1 = results[mode]["overall"]["f1"]
            delta = f"{f1 - base:+.3f}" if base is not None and mode != "none" else ""
            print(f"  {mode:<10}{f1:>8}{delta:>10}")
        if {"oracle", "rag"} <= set(modes) and base is not None:
            kept = results["rag"]["overall"]["f1"] - base
            total = results["oracle"]["overall"]["f1"] - base
            # The ratio only means anything when oracle knowledge actually
            # helped. If it hurt, the denominator is negative and the
            # percentage is worse than useless -- it looks like a result.
            if total > 0.01:
                print(f"\n  retrieval captured {100 * kept / total:.0f}% of the "
                      "oracle-knowledge gain")
            else:
                print("\n  oracle knowledge did not help this backend "
                      f"({total:+.3f} F1), so the retrieval ratio is undefined.")
                print("  Expected for the non-LLM baselines -- they cannot read "
                      "context. Use a real judge.")
        print("\n  paper: +0.251 F1 from oracle knowledge, averaged over general LLMs")

    summary = RESULTS_DIR / f"summary_{_slug(backend.name)}.json"
    payload = {"backend": backend.name, "config": args.config, "rows": len(df),
               "not_sure": args.not_sure, "modes": modes, "results": results}
    if recall is not None:
        payload["rag"] = {"retriever": args.retriever, "k": args.k,
                          "corpus_extra": args.corpus_extra, "recall_at_k": recall}
    summary.write_text(json.dumps(payload, indent=2))
    print(f"\nsaved -> {summary}")


def _slug(text):
    return "".join(c if c.isalnum() else "-" for c in text).strip("-")


if __name__ == "__main__":
    main()
