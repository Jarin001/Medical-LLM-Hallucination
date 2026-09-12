# Step 2 — Add RAG and CoT (do STEP1_VALIDATE_BASELINE.md first)

Target machine: **i7-1165G7, 16 GB RAM, NVIDIA MX450 (2 GB), ~99 GB free.**

Base: the official MedHallu repo and dataset. Baselines: the paper's Table 2,
already published — you do not re-run them.

---

## Which MedHallu models fit

**RAM is not the constraint. Speed is.** At Q4 quantisation nearly every model
in the paper fits in 16 GB; what varies is how long 200 rows takes on a 4-core
CPU. The MX450's 2 GB can offload perhaps 10–15 layers of a 3B model — try it,
but do not plan around it.

Times below assume ~750-token prompts and are rough. Measure your own.

### Tier 1 — start here (~10 s/judgement)

| Model | Ollama tag | Paper F1 (none → oracle) | Δ |
|---|---|---|---|
| Qwen2.5-3B-Instruct | `qwen2.5:3b-instruct` | 0.606 → 0.676 | +0.070 |
| Llama-3.2-3B-Instruct | `llama3.2:3b` | 0.499 → 0.734 | +0.235 |
| Gemma-2-2b-Instruct | `gemma2:2b` | 0.553 → 0.715 | +0.162 |

### Tier 2 — the real experiment (~25 s/judgement)

| Model | Ollama tag | Paper F1 (none → oracle) | Δ |
|---|---|---|---|
| **Qwen2.5-7B-Instruct** | `qwen2.5:7b-instruct` | 0.553 → 0.839 | **+0.286** |
| Llama-3.1-8B-Instruct | `llama3.1:8b` | 0.522 → 0.797 | +0.275 |
| DeepSeek-R1-Distill-8B | `deepseek-r1:8b` | 0.514 → 0.812 | +0.298 |
| BioMistral-7B | GGUF import | 0.570 → 0.648 | +0.078 |

### Tier 3 — overnight only

`gemma2:9b` (0.515 → 0.838, **Δ+0.323**, the largest knowledge effect in the
paper) and `qwen2.5:14b` (0.619 → 0.852). The 14B at Q4 is ~9 GB — tight
alongside the OS.

### Recommended three

1. **Qwen2.5-7B-Instruct** — general, Δ+0.286
2. **Llama-3.1-8B-Instruct** — general, Δ+0.275
3. **BioMistral-7B** — medical fine-tune, Δ+0.078

**Pick by Δ, not by F1.** Δ is how much oracle knowledge helped that model in
the paper. RAG is imperfect knowledge, so a model with Δ+0.286 has room for RAG
to show something. **Qwen2.5-3B has Δ+0.070 — almost no headroom**, so it is a
poor choice for the real run even though it is the fastest. Use it to validate
the pipeline, then switch.

BioMistral is in for contrast: the paper's medical fine-tunes barely benefit
from knowledge, and one (OpenBioLLM) gets *worse*. If RAG helps it anyway,
that is a finding.

---

## Steps

### 1. Install Ollama and pull a model

Get Ollama from <https://ollama.com>, then:

```bash
ollama pull qwen2.5:3b-instruct
```

Ollama serves an OpenAI-compatible endpoint on `localhost:11434`, which is how
the patched MedHallu script reaches it. **No account, no key, no cost.**

### 2. Clone MedHallu and install

```bash
git clone https://github.com/MedHallu/MedHallu.git
```

```bash
pip install pandas scikit-learn openai tqdm requests pyarrow
```

**No torch, no vLLM, no datasets.** The patch removes those requirements.

### 3. Patch the detection script

```bash
python patch_original.py /path/to/MedHallu
```

Seven changes; it backs up to `.py.orig` and has `--revert`. See the file header
for what each one does and why.

### 4. Build the data file

The MedHallu script cannot read MedHallu's own published dataset — the column
names differ. This converts it:

```bash
python src/make_original_csv.py --config pqa_labeled
```

### RAG needs an EXTERNAL knowledge source

RAG must search a corpus the benchmark knows nothing about. Searching
MedHallu's own `Knowledge` fields is not RAG — the correct passage would be
present by construction, and you would only be measuring ranking.

We use **MedRAG/textbooks**: 125,847 snippets from 18 medical textbooks, ~101 MB,
ungated. Downloaded and cached automatically on first use:

```bash
python src/rag.py --corpus textbooks --k 3 --limit 5
```

This is a genuinely hard test. MedHallu questions come from PubMed research
abstracts; medical textbooks may not contain the specific finding at all. That
is the realistic case, and it is why RAG will not reach oracle.

### 5. Configure

Edit the `CONFIGURATION` block at the top of the patched script:

```python
DF_PATH = "medhallu.csv"
CSV_PATH = "results.csv"
LIMIT = 200                        # not 10,000 -- that is days on a CPU
# RAG corpus: MedRAG/textbooks, downloaded automatically
```

and set the models list:

```python
models = [{'type': 'openai', 'model_name': 'qwen2.5:3b-instruct'}]
```

`'openai'` is the type because Ollama speaks the OpenAI protocol. The model
still runs locally.

### 6. Smoke test

Set `LIMIT = 20` and run:

```bash
python Detection/detection_vllm_notsurecase.py
```

Should finish in a few minutes and write five rows — one per condition. Check
the F1 values are numbers, not blanks. **Then** raise LIMIT.

### 7. Full run

```bash
python Detection/detection_vllm_notsurecase.py
```

Five conditions × 200 rows. ~1 hour on a 3B, ~3 hours on a 7B.

### 8. Compare against the paper

```bash
python compare_to_paper.py results.csv
```

Prints the paper's two published rows for that model, then yours beneath, with
the delta.

---

## How to read the comparison

| Your condition | Compare against | Question it answers |
|---|---|---|
| `baseline` | PAPER none | Did you reproduce their setup? Should be close. |
| `oracle` | PAPER oracle | Same check, with knowledge. |
| `cot` | PAPER none | Does reasoning alone beat the baseline? |
| `rag` | PAPER oracle | How much of perfect knowledge does retrieval recover? |
| `rag+cot` | PAPER oracle | Do they add up, or overlap? |

**Compare like with like.** RAG uses knowledge, so it belongs against the
paper's oracle row, not its no-knowledge row. Scoring RAG against the
no-knowledge baseline would make it look far better than it is.

`baseline` landing near PAPER none is your validation that everything works. If
it is wildly off, fix that before trusting any of the other rows.

---

## What to say about the limitations

State these before anyone asks:

- **Q4 quantisation, not fp16.** Different measurement from the paper's.
- **200 rows, not 10,000.** Differences under ~0.05 F1 are noise.
- **RAG retrieves from medical textbooks**, not PubMed. The specific finding a
  research abstract reports may not be in any textbook, so expect RAG to fall
  well short of oracle. That gap is the result, not a bug.
- **One judge model at a time**, not the paper's 14.

None of these invalidate the comparison. They bound what it can claim.
