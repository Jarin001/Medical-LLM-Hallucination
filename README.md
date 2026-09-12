# MedHallu on a laptop

A runnable, GPU-free version of **MedHallu: A Comprehensive Benchmark for
Detecting Medical Hallucinations in Large Language Models**
([arXiv:2502.14302](https://arxiv.org/abs/2502.14302),
[official repo](https://github.com/MedHallu/MedHallu)).

## Can the original be run on this laptop?

**The official repo: no.** Not a configuration problem — a hardware one.

| | Paper / official repo | This laptop |
|---|---|---|
| GPU | 4× NVIDIA RTX A6000, 48 GB VRAM each (192 GB total) | Intel Iris Xe integrated, no CUDA |
| Inference engine | vLLM 0.6.3 with `tensor_parallel_size=4` | vLLM needs a CUDA GPU; will not install usefully |
| Generator model | Qwen2.5-14B (~28 GB in fp16) | 15.7 GB system RAM total |
| Judge models | 14 models, 7B–14B each | — |
| Dataset generation | 26.5 GPU-hours | — |
| Free disk on C: | — | 3.8 GB |

Both entry points in the official repo (`Dataset Generation/generation.py`,
`Detection/detection_vllm_notsurecase.py`) import `vllm` at module level and
load 7B+ models, so neither will start here.

**What is genuinely reachable:** the authors published the finished dataset, so
the *benchmark* half of the paper does not need their hardware at all.

- The 10,000-row dataset is public on the Hub as
  [`UTAustin-AIHealth/MedHallu`](https://huggingface.co/datasets/UTAustin-AIHealth/MedHallu),
  11 MB of parquet, no token required. **No GPU, no regeneration.**
- Detection can be scored against any judge you can reach — a small quantised
  model on CPU, or an API model standing in for the paper's GPT-4o row.
- The semantic analysis (paper §5.3) is pure numerical work on released text.
  Runs on CPU in seconds.

Only *dataset generation* is truly out of reach, and `generate_demo.py` walks
through that pipeline step by step so the mechanism is still legible.

## Setup

Already done if you are reading this in place; the venv lives at `.venv`.
From scratch:

```bash
python -m venv .venv && .venv/Scripts/python.exe -m pip install pandas pyarrow scikit-learn scipy requests tqdm
```

Everything lives on `E:` on purpose — `config.py` also repoints `HF_HOME`, since
the default model cache under `%USERPROFILE%` would land on a nearly-full `C:`.

For the embedding measure (installed and verified here — torch 2.13.0+cpu,
sentence-transformers 6.0.0). CPU-only wheels, no CUDA:

```bash
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/python.exe -m pip install sentence-transformers
```

Optional, for a real local LLM judge — install [Ollama](https://ollama.com), then:

```bash
ollama pull qwen2.5:1.5b-instruct
```

## The four scripts

### 1. `data.py` — get the dataset

```bash
.venv/Scripts/python.exe src/data.py --config pqa_labeled
```

Fetches and caches the parquet, prints the distributions. Reproduces the paper's
Figure 3: `hard` 408 > `medium` 318 > `easy` 274 on the labeled split, and
question-misinterpretation dominating the categories (752/1000).

`to_detection_pairs()` is the important function: each source row becomes **two**
labelled examples — the hallucinated answer (label 1) and the ground truth
(label 0). That is what makes the benchmark balanced, and it matters for
reading the scores below.

### 2. `detect.py` — the benchmark (paper Tables 2 and 4)

```bash
.venv/Scripts/python.exe src/detect.py --backend constant --limit 300
```

```bash
.venv/Scripts/python.exe src/detect.py --backend ollama:qwen2.5:1.5b-instruct --limit 100 --both
```

Reports precision / recall / F1 overall and split by difficulty, with and
without the supporting knowledge. Backends: `constant`, `lexical`,
`ollama:<model>`, `anthropic[:model]`, `openai[:model]`.

**Run `constant` first.** It always answers "hallucinated" and therefore scores
**F1 = 0.667 at precision 0.500** on a balanced set, with no intelligence
whatsoever. Compare that to the paper's Table 2, without knowledge:

| Model (paper, no knowledge) | Overall F1 |
|---|---|
| *always-say-hallucinated baseline* | *0.667* |
| GPT-4o | 0.737 |
| Qwen2.5-14B-Instruct | 0.619 |
| Llama-3.1-8B-Instruct | 0.522 |
| Gemma-2-9b-Instruct | 0.515 |
| OpenBioLLM-Llama3-8B | 0.484 |
| Llama3-Med42-8B | 0.416 |

Only GPT-4o clears the trivial baseline. **Every other model in the table scores
below it**, several far below. That does not make the paper wrong — its finding
is precisely that these models are bad at this task — but it does mean F1 alone
flatters them, and it is why the paper also reports precision. Keep the constant
row in any table you build; it is the number that makes the others interpretable.

`--not-sure` adds the abstain option (Table 4). Watch precision rise while
`resp%` falls — the model is buying accuracy with coverage, which in a clinical
setting is often the right trade.

### 3. `semantics.py` — the semantic claim (paper §5.3, Table 3)

```bash
.venv/Scripts/python.exe src/semantics.py --config pqa_labeled
```

Paper §5.3 reports that **harder-to-detect hallucinations are semantically
closer to the ground truth**. Testing it directly needs 50 candidate
generations per question — GPU work. But difficulty *is* the discriminator vote
count, so the claim is testable on the released rows: are `hard` hallucinations
more similar to ground truth than `easy` ones?

On all 10,000 released rows, **the effect runs the other way** — consistently,
across all three measures, on both splits:

| split | measure | easy | medium | hard | p (two-sided) |
|---|---|---|---|---|---|
| pqa_labeled (1k) | rouge1 | 0.335 | 0.332 | **0.298** | 1.5e-03 |
| pqa_labeled | tfidf cosine | 0.400 | 0.379 | **0.332** | 5.2e-06 |
| pqa_labeled | embedding cosine | 0.743 | 0.725 | **0.692** | 4.7e-05 |
| pqa_artificial (9k) | rouge1 | 0.347 | 0.344 | **0.311** | 2.2e-23 |
| pqa_artificial | tfidf cosine | 0.419 | 0.414 | **0.365** | 8.0e-28 |
| pqa_artificial | embedding cosine | 0.742 | 0.738 | **0.701** | 8.2e-24 |

The embedding row matters most: it closes off "this is only lexical overlap, not
semantics". The reversal survives in embedding space. And the magnitudes line up
with the paper — Table 3 reports cosine 0.715 (fooled) vs 0.696 (not fooled),
which sits squarely inside the 0.69–0.74 range measured here. Comparable
quantity, opposite direction with respect to the difficulty label.

This is **not** a refutation, and the script says so when it prints. Two real
differences:

1. The paper compares *clusters* over 50 candidates per question; this compares
   the single hallucination that was released.
2. The generation pipeline's fallback rule (§3, Algorithm 1 Phase 2) selects,
   among failed candidates, the one with **maximum cosine similarity to the
   ground truth** — and labels it `easy`. So `easy` rows are enriched with
   answers explicitly chosen for being close to the truth. That selection
   effect pushes exactly the way the table shows.

This is the most useful thing in the repo to sit with. The dataset's difficulty
label is not a free-standing property of the text — it is partly an artifact of
how the pipeline terminated. If you build on MedHallu, that is worth knowing
before you treat `difficulty` as a measure of subtlety.

### 4. `generate_demo.py` — the pipeline you cannot run (paper Figure 2)

```bash
.venv/Scripts/python.exe src/generate_demo.py --dry-run --n 3
```

Walks one question at a time through generate → quality vote → entailment check
→ difficulty label. `--dry-run` needs no model and replays released rows.
With `--backend ollama:...` it generates genuinely new hallucinated answers and
runs a real judge vote — the honest small-scale version.

## Suggested order

1. `data.py` — see the data, confirm Figure 3.
2. `generate_demo.py --dry-run` — understand where `difficulty` comes from.
3. `detect.py --backend constant` — establish the floor.
4. `detect.py --backend lexical --both` — a non-LLM attempt.
5. `detect.py --backend ollama:... --limit 100 --both` — a real judge; expect
   the knowledge gap the paper reports.
6. `semantics.py` — the analysis, and the reversal.

## What this is not

Not a reproduction. One judge over a few hundred samples is not 14 models over
10,000, confidence intervals are not computed, and the dataset is consumed
rather than rebuilt. It is a working scaffold for understanding the benchmark
and for scoring any judge you can actually reach.

Original paper and dataset are the authors' work, MIT licensed.
