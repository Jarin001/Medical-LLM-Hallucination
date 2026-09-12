"""Retrieval layer, so detection can use *retrieved* knowledge instead of oracle.

Why this is worth doing
-----------------------
MedHallu's "with knowledge" setting hands the judge the exact PubMed context the
question was written from. That is **oracle retrieval** -- perfect recall, zero
noise. It is also the paper's largest single effect: about +0.25 F1 averaged
over general models.

So the paper measures two points:

    no knowledge          F1 ~ 0.53     (paper avg, general LLMs)
    oracle knowledge      F1 ~ 0.78

and leaves the interesting one unmeasured:

    RETRIEVED knowledge   F1 = ?        <- this module

A real deployment never has oracle context. It retrieves, imperfectly. Where the
retrieved number lands in that 0.25-wide gap tells you how much of the paper's
headline gain survives contact with a real pipeline. That is a genuine question
the paper does not answer.

The setup here also lets you separate the two failure modes, which the oracle
setting conflates:

    retrieval failure   the right passage was never fetched
    reasoning failure   it was fetched and the model still got it wrong

Corpus
------
One document per source row, built from that row's `Knowledge` field. So for
question i, document i is by construction the correct one -- which makes
recall@k directly measurable.

Be honest about what that means: this is a **closed corpus** where the answer is
guaranteed present. It is the optimistic case. A realistic setup retrieves from
all of PubMed, where the right passage may be absent entirely. Treat the numbers
here as an upper bound on what retrieval contributes; see `--corpus-extra` to
dilute the corpus with distractors and get closer to reality.

Usage
-----
  python src/rag.py --retriever tfidf --k 3
  python src/rag.py --retriever dense --k 3        # needs sentence-transformers
  python src/rag.py --retriever dense --k 5 --config pqa_artificial
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

import data
from config import CACHE_DIR, COL_KNOWLEDGE, COL_QUESTION, RESULTS_DIR


# --------------------------------------------------------------------------
# Corpus
# --------------------------------------------------------------------------
def build_corpus(df: pd.DataFrame) -> list[str]:
    """One document per row: that row's knowledge passages joined.

    Document i belongs to question i. That alignment is what makes recall@k
    computable without any extra annotation.
    """
    return [data._flatten_knowledge(v) for v in df[COL_KNOWLEDGE]]


# --------------------------------------------------------------------------
# Retrievers
# --------------------------------------------------------------------------
class TfidfRetriever:
    """Sparse lexical retrieval. No torch, no downloads, runs in a second.

    Weak on paraphrase -- a question and its source passage often share few
    exact words -- which is precisely why the dense retriever below is worth
    the extra 350 MB. Keep this as the cheap baseline.
    """

    name = "tfidf"

    def __init__(self, corpus: list[str]):
        self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english",
                                          sublinear_tf=True)
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def search(self, queries: list[str], k: int) -> np.ndarray:
        q = self.vectorizer.transform(queries)
        scores = (q @ self.doc_matrix.T).toarray()
        return np.argsort(-scores, axis=1)[:, :k]


class DenseRetriever:
    """Sentence-embedding retrieval with cosine similarity.

    The corpus is small enough (10k docs) that a plain matrix multiply beats
    setting up FAISS. Embeddings are cached to disk keyed by model and corpus
    size, since encoding 10k passages on CPU takes a couple of minutes.
    """

    def __init__(self, corpus: list[str],
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # optional dep
        self.name = f"dense:{model_name.split('/')[-1]}"
        self.model = SentenceTransformer(model_name, device="cpu")

        cache = CACHE_DIR / f"corpus_{model_name.replace('/', '_')}_{len(corpus)}.pkl"
        if cache.exists():
            self.doc_embeddings = pickle.loads(cache.read_bytes())
        else:
            self.doc_embeddings = self.model.encode(
                corpus, normalize_embeddings=True, batch_size=32,
                show_progress_bar=True)
            cache.write_bytes(pickle.dumps(self.doc_embeddings))

    def search(self, queries: list[str], k: int) -> np.ndarray:
        q = self.model.encode(queries, normalize_embeddings=True, batch_size=32,
                              show_progress_bar=True)
        scores = q @ self.doc_embeddings.T
        return np.argsort(-scores, axis=1)[:, :k]


def get_retriever(spec: str, corpus: list[str]):
    if spec == "tfidf":
        return TfidfRetriever(corpus)
    if spec.startswith("dense"):
        _, _, model = spec.partition(":")
        return DenseRetriever(corpus, model or "sentence-transformers/all-MiniLM-L6-v2")
    raise SystemExit(f"unknown retriever {spec!r} (use 'tfidf' or 'dense[:model]')")


# --------------------------------------------------------------------------
# Evaluation and use
# --------------------------------------------------------------------------
def recall_at_k(ranked: np.ndarray, k: int, gold: np.ndarray | None = None) -> float:
    """Fraction of questions whose own document appears in the top k.

    `gold[i]` is the corpus position of question i's own passage. It defaults to
    arange, which is right only when the queries ARE the corpus. Once you
    evaluate a sample of questions against a full corpus -- which you should,
    since shrinking the corpus to the sample makes retrieval trivially easy --
    the positions differ and must be passed in.
    """
    gold = (np.arange(len(ranked)) if gold is None else np.asarray(gold))[:, None]
    return float((ranked[:, :k] == gold).any(axis=1).mean())


def mrr(ranked: np.ndarray, gold: np.ndarray | None = None) -> float:
    """Mean reciprocal rank of the correct document."""
    gold = (np.arange(len(ranked)) if gold is None else np.asarray(gold))[:, None]
    hits = ranked == gold
    total = 0.0
    for row in range(len(ranked)):
        pos = np.flatnonzero(hits[row])
        if len(pos):
            total += 1.0 / (pos[0] + 1)
    return total / len(ranked)


def retrieve_knowledge(df: pd.DataFrame, retriever, k: int,
                       corpus: list[str] | None = None) -> tuple[list[str], np.ndarray]:
    """Return the concatenated top-k passages per question, plus the rankings.

    This is what gets handed to the judge in place of the oracle context. The
    top-k are joined in rank order, so a retriever that puts the right passage
    first is rewarded by the model reading it first.
    """
    corpus = corpus if corpus is not None else build_corpus(df)
    ranked = retriever.search(list(df[COL_QUESTION].astype(str)), k)
    passages = ["\n\n".join(corpus[j] for j in row) for row in ranked]
    return passages, ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pqa_labeled")
    ap.add_argument("--retriever", default="tfidf", help="tfidf | dense[:model]")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--corpus-extra", default=None,
                    help="add another config's passages as distractors, e.g. pqa_artificial")
    args = ap.parse_args()

    df = data.load(args.config)
    if args.limit:
        df = df.head(args.limit)
    df = df.reset_index(drop=True)

    corpus = build_corpus(df)
    n_gold = len(corpus)
    if args.corpus_extra:
        extra = build_corpus(data.load(args.corpus_extra))
        corpus = corpus + extra
        print(f"corpus: {n_gold} gold + {len(extra)} distractors = {len(corpus)}")
    else:
        print(f"corpus: {len(corpus)} documents (no distractors)")

    retriever = get_retriever(args.retriever, corpus)
    ranked = retriever.search(list(df[COL_QUESTION].astype(str)), max(args.k, 10))

    print(f"\nretriever: {retriever.name}   queries: {len(df)}")
    print(f"\n  {'metric':<12}{'value':>8}")
    print("  " + "-" * 20)
    for kk in (1, 3, 5, 10):
        print(f"  recall@{kk:<5}{recall_at_k(ranked, kk):>8.3f}")
    print(f"  {'MRR':<12}{mrr(ranked):>8.3f}")

    out = RESULTS_DIR / f"retrieval_{args.retriever.replace(':', '-')}_{args.config}.csv"
    pd.DataFrame({
        "question": df[COL_QUESTION],
        "gold_doc": np.arange(len(df)),
        "top1": ranked[:, 0],
        "hit@1": ranked[:, 0] == np.arange(len(df)),
        "hit@k": (ranked[:, :args.k] == np.arange(len(df))[:, None]).any(axis=1),
    }).to_csv(out, index=False)
    print(f"\nsaved -> {out}")

    print("\nNext: feed these to the judge instead of the oracle context --")
    print(f"  python src/detect.py --backend <b> --knowledge-mode rag "
          f"--retriever {args.retriever} --k {args.k}")


if __name__ == "__main__":
    main()
