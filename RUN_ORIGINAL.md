# Running the original MedHallu repo — a step-by-step guide

For the official repo at <https://github.com/MedHallu/MedHallu>, running the
**Detection** benchmark that produces the paper's Table 2.

Written for someone who has never used a rented GPU before. Every step tells you
what you're doing, why, and **how to tell it worked** before moving on.

---

## The core idea: build up, don't dive in

The full run is 13 models × 2 settings × 10,000 rows — roughly **10 hours** and
**190 GB of downloads**. If something is misconfigured, you find out hours in,
having paid for all of it.

So this guide is built as a ladder. Each rung is cheap and proves one thing:

| Phase | What it proves | Cost |
|---|---|---|
| **1** | The GPU exists and CUDA works | 2 min |
| **2** | torch + vLLM installed correctly | 20 min |
| **3** | You can download models | 5 min |
| **4** | Your data file is readable | 2 min |
| **5** | **The whole pipeline works end to end** | ~15 min, one small model, 20 rows |
| **6** | It scales | ~30 min, one model, 1,000 rows |
| **7** | Full run | ~10 hours |

**Do not skip Phase 5.** It is the single most valuable step in this document.
It exercises every part of the chain — data loading, model download, vLLM
startup, generation, parsing, metrics, CSV writing — for about 15 minutes and a
few pennies. If it passes, Phase 7 is very likely to just work.

---

# Phase 1 — Get a machine and confirm the GPU

## Where to get one

| Option | Notes |
|---|---|
| **Your lab's cluster** | Free. Ask your advisor two things: does the lab have a GPU cluster allocation, and who adds you to it (usually a sysadmin, not them). Northeastern's is called Discovery. |
| **RunPod** | Easiest rental. Pick a **PyTorch template** — CUDA and torch come preinstalled, which skips most of Phase 2. ~$0.50–$2.00/hr. Start here. |
| **Vast.ai** | Cheapest, slightly rougher edges. |
| **Lambda Labs** | Pricier, very clean. |
| **Colab Pro+** | Fine for a couple of models; sessions time out, so bad for the full run. |

**Full run costs roughly $20–40** if you rent. The Phase 5 smoke test is under a
dollar.

If you get cluster access, note that you can't run heavy jobs on the login node —
use `srun ... --pty bash` for an interactive GPU session, and `sbatch` for the
long run. Ask cluster support for their job template; every cluster differs.

## What size (applies to both routes)

**You do not need their 4 GPUs.** One card runs everything in their list. The
biggest model is Qwen2.5-14B, about 28 GB in fp16.

- **Good:** one A6000 (48 GB) or one A100 (40 or 80 GB)
- **Workable:** one RTX 4090 or L4 (24 GB) — fits everything except Qwen2.5-14B,
  which you'd skip or run in 8-bit
- **Enough for the smoke test:** anything with 16 GB+

**Disk: get at least 250 GB.** The 13 models total roughly **190 GB** of
downloads. This is the setting people forget, and it fails you at model 9 after
six hours — the worst possible time.

**If you're stuck with less disk**, run the models in two or three batches:
edit the `models` list down to 4–5 entries, run, then clear the cache and do the
next batch. Results append to your CSV, so batching costs you nothing:

```bash
rm -rf ~/.cache/huggingface/hub
```

Only run that between batches, never mid-run.

## Confirm it works

Open a terminal on the rented machine and run:

```bash
nvidia-smi
```

**Success looks like** a table showing your GPU name, total memory, and driver
version. Note the **CUDA Version** in the top-right — you need it in Phase 2.

**If this fails** (`command not found`), you don't have a GPU machine. Stop here
and fix that; nothing below will work.

---

# Phase 2 — Install the software

## Get the code

```bash
git clone https://github.com/MedHallu/MedHallu.git
```

```bash
cd MedHallu
```

## FIRST: make a dedicated environment. Do not skip this.

**Never `pip install vllm` into a conda `base` environment or any environment
you share with other work.** vLLM pins exact versions of torch, numpy,
transformers, protobuf and ~30 other packages, and pip will silently upgrade or
downgrade whatever is already there. On a shared lab machine this breaks other
people's projects.

```bash
conda create -n medhallu python=3.11 -y
```

```bash
conda activate medhallu
```

Your prompt must show `(medhallu)`, not `(base)`, before you install anything.
Check it:

```bash
python -c "import sys; print(sys.prefix)"
```

That path must contain `envs/medhallu`. If it says `anaconda3` with no `envs`,
you are still in base — stop and activate again.

No conda? Use a plain virtualenv instead:

```bash
python -m venv ~/medhallu-env && source ~/medhallu-env/bin/activate
```

### This warning is not theoretical

On 2026-09-11 we ran `pip install vllm` in `(base)` on a shared lab machine.
vLLM 0.29.0 replaced about 35 packages. The damage:

| Package | Was | Became | What it broke |
|---|---|---|---|
| **numpy** | 1.25.2 | 2.3.5 | scipy 1.11.1 (needs `numpy<1.28`), langchain, langchain-community |
| **transformers** | 5.9.0 | 5.17.0 | trl, pyiqa, sentence-transformers |
| **protobuf** | 5.29.6 | 6.33.6 | tensorboard, mediapipe, unsloth-zoo, grpcio-status, google-ai-generativelanguage |
| **torchvision** | 0.20.1+cu124 | 0.28.0 | nothing — 0.20.1 was already broken against torch 2.13.0; 0.28.0 is the correct pairing |
| **openai** | 1.109.1 | 3.13.0 | langchain-openai |
| **huggingface_hub** | 1.15.0 | 1.31.0 | text-generation |
| **pydantic** | 2.9.2 | 2.13.5 | anaconda-cloud-auth |

numpy is the one that hurts. scipy pins `numpy<1.28`, so scipy can stop
importing entirely, and a great deal sits on top of scipy.

That machine also had FreeSurfer pointed at ADNI neuroimaging data, plus
unsloth and pyiqa — other people's research, in the same base environment.

**Watch for the `ERROR: pip's dependency resolver does not currently take into
account...` block at the end of a pip run.** It is not cosmetic. It is the list
of what you just broke.

### RECOVERY — if you already ran `pip install vllm` in base

Nine steps, in order. Run the command, read the note under it, move on.

Steps 1–4 below were actually executed on our machine and are confirmed working.
The pinned versions come from the `Found existing installation:` lines in our
pip log — **if your log shows different versions, use yours, not ours.**

---

**Step 1 — remove vllm from base**

```bash
conda activate base
```

```bash
pip uninstall -y vllm flashinfer-python tilelang xgrammar compressed-tensors quack-kernels tokenspeed-mla tokenspeed-triton humming-kernels instanttensor nvidia-cutlass-dsl nvidia-cutlass-dsl-libs-base nvidia-cutlass-dsl-libs-core nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13 apache-tvm-ffi PyNvVideoCodec torchcodec depyf outlines_core lm-format-enforcer llguidance mistral_common openai-harmony model-hosting-container-standards prometheus-fastapi-instrumentator
```

---

**Step 2 — restore the packages vllm replaced**

```bash
pip install --no-deps "numpy==1.25.2" "transformers==5.9.0" "protobuf==5.29.6" "tokenizers==0.22.2" "safetensors==0.7.0" "huggingface_hub==1.15.0" "openai==1.109.1" "numba==0.57.1" "llvmlite==0.40.0"
```

```bash
pip install --no-deps "pydantic==2.9.2" "pydantic_core==2.23.4" "opencv-python-headless==4.11.0.86" "jsonschema==4.19.2" "fastapi==0.115.8" "starlette==0.45.3" "typer==0.25.1" "anyio==4.8.0" "click==8.4.0" "idna==3.11" "typing_extensions==4.15.0" "jiter==0.14.0" "rpds-py==0.10.6" "PyJWT==2.4.0" "prometheus-client==0.14.1" "truststore==0.8.0" "hf-xet==1.4.3" "cuda-pathfinder==1.5.1" "cuda-bindings==13.2.0" "googleapis-common-protos==1.68.0" "opentelemetry-api==1.30.0" "opentelemetry-sdk==1.30.0" "opentelemetry-proto==1.30.0" "opentelemetry-semantic-conventions==0.51b0" "opentelemetry-exporter-otlp-proto-common==1.30.0" "opentelemetry-exporter-otlp-proto-grpc==1.30.0"
```

> **Why `--no-deps` is required, not optional.** Without it this fails with
> `ResolutionImpossible`. A long-lived conda base env is usually already
> inconsistent by pip's metadata rules — ours had numba 0.57.1 next to numpy
> 1.25.2, while numba 0.57.1 declares `numpy<1.25`. It worked, but pip cannot
> *re-derive* that state. `--no-deps` means "install exactly these, touch
> nothing else," which is precisely the goal when restoring versions that were
> working minutes ago.

---

**Step 3 — restore torchvision to match your torch**

```bash
pip install --no-deps "torchvision==0.28.0"
```

```bash
python -c "import torch, torchvision, torchvision.ops; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| ops OK')"
```

> **Don't blindly restore the version from your log.** Ours had
> `torchvision 0.20.1+cu124` next to `torch 2.13.0` — a pairing that cannot
> work, since torchvision 0.20.x is built for torch ~2.5. It threw
> `RuntimeError: operator torchvision::nms does not exist` on import, and had
> been doing so long before anyone ran vLLM. (`xformers 0.0.29.post3` pinning
> `torch==2.6.0` in the same env is the other fingerprint of an old torch
> upgrade that left packages behind.) Restoring 0.20.1 faithfully restores a
> broken package.
>
> Pairing rule: torch 2.13 ↔ torchvision 0.28, torch 2.6 ↔ torchvision 0.21,
> torch 2.5 ↔ torchvision 0.20. Check your own `torch.__version__` and match.
>
> The second command matters: a version number doesn't prove it works, because
> the failure is in compiled op registration, not metadata.

---

**Step 4 — verify base is healthy**

```bash
python -c "import numpy, scipy, transformers; print('numpy', numpy.__version__); print('scipy', scipy.__version__); print('transformers', transformers.__version__)"
```

Expect `numpy 1.25.2`, `scipy 1.11.1`, `transformers 5.9.0`. **These three are
what break other people's work** — they must come back cleanly.

> Check these three on their own, not bundled with torch/torchvision. If
> torchvision throws, the traceback hides whether numpy and scipy are fine —
> which is exactly what happened to us the first time.
>
> If something still errors, check whether it was already broken *before* your
> install. Compare against the `Found existing installation:` line in your pip
> log: if that version could never have worked with the rest of the env, you
> found a pre-existing problem, not one you caused.

Base is now restored. The rest of these steps set up your own environment.

---

**Step 5 — create the separate environment**

```bash
conda create -n medhallu python=3.11 -y
```

```bash
conda activate medhallu
```

---

**Step 6 — confirm you actually left base**

```bash
python -c "import sys; print(sys.prefix)"
```

Output must contain `envs/medhallu`. If it doesn't, run `conda activate medhallu`
again. **Do not install anything until this passes.**

---

**Step 7 — install vllm, in the new environment**

```bash
pip install vllm
```

```bash
pip install transformers datasets pandas scikit-learn sentence-transformers openai
```

---

**Step 8 — confirm it works**

```bash
python -c "import torch; from vllm import LLM; print('torch', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count()); print('vllm OK')"
```

Need `CUDA: True` and `vllm OK`.

**Then jump to Phase 3.** Skip the "Install" section below — Step 7 already did it.

---

**Step 9 — tell whoever maintains the machine**

If it's shared, a job of theirs may have failed between your install and the
restore, and they will not connect it to you. One message now saves them days.

Worth mentioning to them separately: **torchvision and xformers were already
mismatched against torch 2.13.0 before any of this.** That's a real bug on the
box, independent of you, and they'll want to know.

## Install

**The repo has no `requirements.txt`.** You install by hand. Do it in this
order — it matters.

In your **dedicated environment** — confirm the prompt says `(medhallu)` one
more time — install vLLM first and let it choose its own torch version:

```bash
pip install vllm
```

This is the step most likely to break, because vLLM is fussy about matching
your CUDA version. Installing vLLM first and letting it pull the torch build it
wants avoids most conflicts. If you install torch first, pip will often
downgrade or replace it and you'll get cryptic errors later.

"Let vLLM pick its own versions" is safe **only** in an environment where
nothing else matters. That is exactly why the environment comes first.

Then the rest, which are all easy:

```bash
pip install transformers datasets pandas scikit-learn sentence-transformers openai
```

## Confirm it works

```bash
python -c "import torch; print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPUs:', torch.cuda.device_count())"
```

**Success looks like:**

```
torch 2.x.x+cu121
CUDA available: True
GPUs: 1
```

**`CUDA available: False` is the failure to watch for.** It means torch was
installed without GPU support. Fix it by reinstalling torch for your CUDA
version from the [PyTorch install page](https://pytorch.org/get-started/locally/),
matching the CUDA number you saw in `nvidia-smi`.

Then check vLLM itself:

```bash
python -c "from vllm import LLM, SamplingParams; print('vllm imports fine')"
```

**Do not continue until both of these pass.** Everything downstream assumes them.

---

# Phase 3 — Get access to the models

Several models are **gated**: you must accept a licence on the HuggingFace
website before you can download them.

```bash
huggingface-cli login
```

Paste a token from <https://huggingface.co/settings/tokens> (a "read" token is
enough).

Then visit each of these pages in a browser and click the agree button:

- `meta-llama/Llama-3.1-8B-Instruct`
- `meta-llama/Llama-3.2-3B-Instruct`
- `google/gemma-2-2b-it`
- `google/gemma-2-9b-it`

Approval is usually instant, but **do this before you start paying for GPU
time.** Occasionally Meta's form takes a while, and you don't want to discover
that with a meter running.

**Confirm it works** — this downloads a tiny file, not a model:

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().model_info('meta-llama/Llama-3.1-8B-Instruct').id)"
```

Success prints the model name. A 401 or 403 means the licence isn't accepted yet.

---

# Phase 4 — Prepare the data file

## Why this step exists

**Their detection script cannot read their own published dataset.** This is not
your mistake — it's a gap in the repo.

Their script reads a local CSV expecting these column names:

| Script expects | HuggingFace release provides |
|---|---|
| `question` | `Question` |
| `ground_truth` | `Ground Truth` |
| `least_similar_answer` | `Hallucinated Answer` |
| `knowledge` | `Knowledge` |
| `final_difficulty_level` | `Difficulty Level` |

It also calls `ast.literal_eval(df.loc[i, 'knowledge'])['contexts']`, so
`knowledge` must be a **stringified dict** with a `contexts` key — not the plain
list the release ships.

So you have to convert it first. The Detection README says to load from
HuggingFace, but the script would crash on it.

## Build both files now

Use the converter in this project. It needs no GPU, so run it on your laptop and
copy the results over — or just clone this project onto the GPU box.

**A 20-row file for the smoke test:**

```bash
python src/make_original_csv.py --config pqa_labeled --limit 20
```

**The full 1,000-row file:**

```bash
python src/make_original_csv.py --config pqa_labeled
```

For the complete 10,000 rows, also build `--config pqa_artificial` and
concatenate the two CSVs.

## Confirm it works

```bash
python -c "import pandas as pd, ast; df = pd.read_csv('data/original_format_pqa_labeled_first20.csv'); [ast.literal_eval(df.loc[i,'knowledge'])['contexts'] for i in range(len(df))]; print('rows:', len(df)); print('columns:', list(df.columns))"
```

**Success** prints 20 rows and the six lowercase column names. If
`ast.literal_eval` throws, the knowledge column is in the wrong format.

---

# Phase 5 — The smoke test (do not skip)

This is the important one. You will run the real script, unmodified in logic,
against **one small model and 20 rows**. Target: under 15 minutes.

## 5a — Fix the parser first

There's a bug in their scoring that you want fixed before you generate any
numbers. In `Detection/detection_vllm_notsurecase.py`, `calculate_metrics`
currently does:

```python
    for i in llm_answers:
        i_lower = i.lower()
        if any(x in i_lower for x in ['1', 'not', 'non']):
            llm_answers_int.append(1)
        elif any(x in i_lower for x in ['not sure', 'pass', 'skip', '2']):
            llm_answers_int.append(2)
        else:
            llm_answers_int.append(0)
```

`'not'` is tested in the first branch, so **`"not sure"` is scored as
"hallucinated"** and the `'not sure'` branch can never be reached. `'non'`
catches `"non-hallucinated"` and also scores it "hallucinated" — the opposite of
what the model said. And `"The answer is hallucinated"` matches none of the
three patterns, so it falls through to "factual". Inverted in both directions.

Replace that loop with this:

```python
    import re
    for i in llm_answers:
        text = (i or "").strip()
        m = re.search(r"[012]", text)        # the prompt asks for a bare digit
        if m:
            llm_answers_int.append(int(m.group()))
            continue
        low = text.lower()                    # fall back to words; order matters
        if any(x in low for x in ["not sure", "unsure", "pass", "skip"]):
            llm_answers_int.append(2)
        elif any(x in low for x in ["not hallucinat", "non-hallucinat",
                                     "not a hallucinat", "factual", "is correct"]):
            llm_answers_int.append(0)
        elif "hallucinat" in low:
            llm_answers_int.append(1)
        else:
            llm_answers_int.append(2)
```

Checked against 13 realistic replies: the original mis-scores 8 of them, this
version gets all 13 right.

## 5b — Point it at the small file

Open `Detection/detection_vllm_notsurecase.py`.

**Their README cites lines 304–305 and 143–144. Those numbers are stale.** The
real ones are below — I verified each against the current file.

**Lines 293–294** (currently `" "`):

```python
    df_path = "/full/path/to/original_format_pqa_labeled_first20.csv"
    csv_path = "/full/path/to/smoke_test_results.csv"
```

**Line 208** — set this to your actual GPU count:

```python
            tensor_parallel_size=1,    # was 4
```

Leaving it at 4 on a one-GPU machine is a guaranteed crash.

**Lines 296–311** — the `models` list. Comment out every line except one, and
use this one:

```python
    models = [
        {'type': 'hf', 'model_name': 'Qwen/Qwen2.5-3B-Instruct'},
    ]
```

Why this model: it's only ~6 GB, it's **not gated** so there's no licence wait,
and it's already in their list so you're not testing something off-script.

## 5c — Run it

```bash
cd Detection
```

```bash
python detection_vllm_notsurecase.py
```

## What success looks like

1. A few minutes of model download progress bars
2. vLLM startup logs, ending with something about a KV cache / profiling run
3. `Completed Qwen/Qwen2.5-3B-Instruct with knowledge = False`
4. The same line again with `knowledge = True`
5. A new `smoke_test_results.csv` containing **two rows** — one per setting

Open that CSV. You should see `precision`, `recall`, `f1`, and the
`easy_/medium_/hard_` columns, populated with numbers between 0 and 1.

**Don't judge the scores.** 20 rows is far too few to mean anything. You are
checking that the machinery runs, not what it says.

## If it fails

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA out of memory` | Model too big, or `tensor_parallel_size` wrong | Confirm line 208 matches your GPU count; lower `gpu_memory_utilization` on line 210 from 0.85 to 0.70 |
| Hangs forever at startup | `tensor_parallel_size=4` on fewer GPUs | Set line 208 to 1 |
| `KeyError: 'question'` | Wrong CSV | Redo Phase 4 |
| `KeyError: 'contexts'` | Knowledge column not a stringified dict | Redo Phase 4 with the converter |
| `401` / gated repo | Licence not accepted | Redo Phase 3 |
| `FileNotFoundError` | `df_path` wrong | Use an absolute path, not a relative one |
| All metrics are 0 or blank | Model output not parsing | Apply the 5a patch |

---

# Phase 6 — Scale up on one model

Same single model, now the full 1,000 rows. Change **only** `df_path`:

```python
    df_path = "/full/path/to/original_format_pqa_labeled.csv"
```

Run it again. Expect roughly 20–40 minutes.

**Now the numbers mean something.** Compare against the paper's Table 2 row for
Qwen2.5-3B-Instruct: **0.606** F1 without knowledge, **0.676** with.

If you land in that neighbourhood, your setup is validated and you can trust
Phase 7. If you're far off, something is wrong and it's much cheaper to find out
now.

Two reasons your number won't match exactly, both expected: their code shows the
judge one randomly chosen answer per row, so there's sampling noise; and you're
on 1,000 rows, not 10,000.

---

# Phase 7 — The full run

Restore the whole `models` list (lines 296–311), point `df_path` at your
10,000-row CSV, and start it.

```bash
python detection_vllm_notsurecase.py
```

Expect **8–12 hours.** Each model runs in its own subprocess and GPU memory is
freed between them, and results **append** to the CSV as it goes — so you can
watch progress, and a crash at model 9 doesn't lose models 1–8.

Run it under `tmux` or `screen` so a dropped SSH connection doesn't kill it:

```bash
tmux new -s medhallu
```

Then start the script inside that session. Detach with `Ctrl-B` then `D`, and
reattach later with `tmux attach -t medhallu`.

## One thing to know about the model list

The list holds **13** models, not the 14 in Table 2:

- `OpenMeditron/Meditron3-8B` is in the code but **not** in Table 2
- Both GPT-4o rows are **commented out** (line 310) because they need an API key

So the repo as shipped **cannot reproduce Table 2**. GPT-4o is the only model
that beats a trivial baseline, and it won't run unless you add the OpenAI entry
back and uncomment `openai.api_key` at line 18. Worth knowing before you claim
you reproduced the table.

---

# The other two scripts (optional)

## Similarity checking

```bash
python bidirectional_checking.py
```

Fill in `input_file` and `output_file` at **lines 120–121** first (both are
empty strings; the README's 143–144 is stale). Feed it the CSV from Phase 4 —
it expects `least_similar_answer` and `ground_truth`.

Note: it loads `roberta-large-mnli` (line 12) with threshold `0.5` (lines 53,
59), while the paper says `deberta-large-mnli` at `0.75`. **Their code and their
paper disagree.** Cite the code, not the paper.

## Topic breakdown

`Mesh.py` reads `./desc2025.xml` at **line 14** (README says 13). You download
that from the NIH separately, a few hundred MB. Skippable — it only produces a
subject-area breakdown.

---

# Dataset generation — you probably shouldn't

`Dataset Generation/generation.py` is the 26.5-hour, 4×A6000 job. The output is
already published, so there's rarely a reason to re-run it. It also needs an
OpenAI API key on top of the GPU, so it costs money as well as compute.

If you must:

- **lines 559–560** — `OUTPUT_FILE` and `CHECKPOINT_FILE` paths (both `" "`)
- **line 564** — your OpenAI API key (for TextGrad and one of the judges)
- **line 570** is hardcoded to `pqa_artificial` with `split="train[:9000]"`, so
  as shipped it only regenerates the 9,000 artificial rows, never the 1,000
  expert-annotated ones
- **`BATCH_SIZE` is set twice** — line 43 and again at line 558. Line 558 wins.

There's a checkpoint system, so it resumes after an interruption.

---

# Final checklist

Before you start Phase 7 and the clock:

- [ ] `nvidia-smi` shows your GPU
- [ ] **You are in a dedicated environment** — `sys.prefix` contains `envs/`, prompt is not `(base)`
- [ ] The last pip run ended with **no** `dependency resolver` error block
- [ ] `torch.cuda.is_available()` is `True`
- [ ] `from vllm import LLM` works
- [ ] `huggingface-cli login` done, all four gated licences accepted
- [ ] CSV built, verified with the `ast.literal_eval` check
- [ ] Parser patched (Phase 5a)
- [ ] `df_path` and `csv_path` are absolute paths
- [ ] `tensor_parallel_size` matches your GPU count
- [ ] **Phase 5 passed** — two rows in a results CSV
- [ ] **Phase 6 landed near 0.606 / 0.676** for Qwen2.5-3B
- [ ] ≥250 GB free disk
- [ ] Running inside `tmux` or `screen`

---

**Honesty note:** the line numbers, the column-name mismatch, the parser bug and
its fix, and the CSV converter are all verified against the actual files in the
repo. The install and run steps (Phases 2, 3, 5c, 7) come from reading their
code and standard practice — **I have not executed their pipeline**, because it
needs hardware I don't have. Treat the phases as a careful plan, not a
transcript.
