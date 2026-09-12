# MedHallu — what to say out loud

Glance at this before the meeting. Nothing here needs to be read aloud verbatim.

Two things your advisor will want: **what you changed**, and **what you found**.

---

## The 60-second version

> "MedHallu is a benchmark for whether language models can tell a real medical
> answer from a fabricated one. The authors took 10,000 question–answer pairs
> from PubMedQA, had an LLM write a convincing wrong answer for each, and
> labelled each fake easy / medium / hard by how many judge models it fooled.
> Then they scored 14 models on spotting them.
>
> Their code needs four A6000s and vLLM, so it won't run on my laptop. But they
> published the finished dataset on HuggingFace — public, 11 megabytes.
> Generating the dataset is the expensive half; evaluating on it isn't. So I
> rewrote the evaluation side to run locally with no GPU.
>
> It works and the data checks out against the paper. Along the way I found
> three things worth knowing before we build on this."

---

# PART 1 — What I changed from the original repo

## The short answer

> "I kept their dataset and their prompts. I replaced their entire model-serving
> and data-loading layer, because that's the part that assumes four GPUs. Their
> repo is four scripts; mine is six smaller modules with the GPU dependency
> removed and a pluggable judge instead."

## File by file

| Original repo | What I did | Why |
|---|---|---|
| `Detection/detection_vllm_notsurecase.py` | Rewrote as `detect.py` + `backends.py` | It hard-codes vLLM with `tensor_parallel_size=4` and fp16 7–14B models. Won't start without CUDA. |
| — (no equivalent) | Added `data.py` | Their script reads a local CSV from their own generation run. It cannot read the published dataset — see below. |
| — (no equivalent) | Added `config.py` | Redirects the model cache off `C:`, which has 3.8 GB free. |
| `Detection/bidirectional_checking.py` | Reworked as `semantics.py` | Theirs scores one pair at a time on GPU. Mine does three similarity measures over all 10,000 rows on CPU, plus a significance test. |
| `Dataset Generation/generation.py` | Shrank to `generate_demo.py` | The real thing is 26.5 GPU-hours. Mine walks a couple of rows through the same four steps so the pipeline is legible. |
| `Detection/Mesh.py` | **Not ported** | Needs a separate 300 MB MeSH XML download and only produces a topic breakdown. Skipped as not load-bearing. |
| `Prompts/*.txt` | **Kept, near-verbatim** | The detection prompt and the four hallucination-type definitions are the benchmark. Changing them would change what's measured. |

## Three changes that are methodological, not just plumbing

Be upfront about these — they're the ones worth defending.

**1. I score both answers per row; they score one at random.**
Their code does `random_val = random.randint(0, 1)` and shows the judge either
the true answer or the fake one, so each row yields one judgement. Mine uses
both, so each row yields two. Mine has no sampling noise and uses all the data;
theirs is balanced only in expectation. Different, and I'd argue better, but it
does mean my numbers aren't line-for-line comparable to theirs.

**2. I had to write my own data loader, because theirs can't read their own published dataset.**
Their script expects columns `question`, `ground_truth`, `least_similar_answer`,
`knowledge`, `final_difficulty_level`. The HuggingFace release uses
`Question`, `Ground Truth`, `Hallucinated Answer`, `Knowledge`,
`Difficulty Level`. It also does `ast.literal_eval(knowledge)['contexts']`,
expecting a stringified dict, where the release gives a plain list. So the
Detection README tells you to load from HuggingFace, but the script would crash
on it. Renaming is easy; it's just not done.

**3. Their NLI code and the paper disagree.**
The paper says `microsoft/deberta-large-mnli` at threshold 0.75.
`bidirectional_checking.py` uses `roberta-large-mnli` with a default threshold
of 0.5. Minor, but if we cite their entailment setup we should cite the code,
not the paper.

## What I did NOT do — say this before you're asked

> "This isn't a reproduction. One judge over a few hundred samples, not fourteen
> models over ten thousand. No confidence intervals. And I consume their dataset
> rather than regenerating it, so I've read the generation pipeline but not
> verified it end to end."

---

# PART 2 — Findings

## Finding 1 — a trivial baseline beats most of their models

> "The task is balanced — every row has one fake answer and one real one. So a
> classifier that answers 'hallucinated' every single time gets recall 1.0,
> precision 0.5, and an F1 of 0.667.
>
> Against their Table 2, without knowledge: only GPT-4o beats that, at 0.737.
> Qwen2.5-14B is 0.619, Llama-3.1-8B is 0.522, and the medical fine-tunes are
> worse — Med42-8B is 0.416. Thirteen of fourteen models score below a constant.
>
> That's consistent with their thesis that models are bad at this. But overall F1
> alone makes them look better than they are, and the paper never states the
> floor, so a reader can't tell that 0.52 means 'worse than nothing'. If we
> report on this benchmark, the constant baseline goes in the table."

*If pushed:* yes, this is a known property of F1 on balanced binary tasks. The
criticism is about reporting, not arithmetic.

## Finding 2 — one of their conclusions reverses on the released data

> "Section 5.3 says harder-to-detect hallucinations are semantically closer to
> the ground truth — that's their explanation for why hard ones are hard.
>
> The difficulty label is essentially the judge vote count, so I tested it
> directly: are hard fakes more similar to the true answer than easy fakes? Three
> measures — ROUGE-1, TF-IDF cosine, and sentence-embedding cosine — over all
> 10,000 rows.
>
> All three come out the opposite way. Easy fakes are *more* similar. On the 9k
> split the p-values run from about 1e-23 to 1e-28. The embedding numbers are
> 0.742 easy versus 0.701 hard, and their Table 3 reports 0.715 versus 0.696 —
> so I'm measuring a comparable quantity and getting the opposite sign.
>
> I don't think they're wrong, though. Two things differ. They cluster over 50
> candidate generations per question; I only have the single answer they
> released. And their fallback rule — Algorithm 1, Phase 2 — says that when
> generation fails five times, take the candidate with *maximum* cosine
> similarity to ground truth and label it easy. So the easy bucket is
> deliberately stocked with answers chosen for being close to the truth. That
> alone would produce what I see.
>
> The takeaway is that the difficulty label isn't a clean measure of subtlety —
> it's partly an artifact of how the generation loop terminated. I'd be careful
> using it as a difficulty axis in our own work."

**This is your strongest card.** Real observation, 10,000 rows, three measures,
and you have a mechanism that explains it without accusing anyone of sloppiness.

## Finding 3 — there's a parsing bug in their scoring code

> "In `calculate_metrics`, they turn the model's text reply into a 0/1/2 label
> with this:
>
>     if any(x in reply for x in ['1', 'not', 'non']):  -> 1
>     elif any(x in reply for x in ['not sure', 'pass', 'skip', '2']): -> 2
>
> Because `'not'` is tested in the first branch, the string 'not sure' matches
> there and is scored as 1, hallucinated. The `'not sure'` case in the second
> branch is unreachable for anything containing 'not'. And `'non'` catches
> 'non-hallucinated', which also becomes 1 — the exact opposite of what the model
> said. Meanwhile 'The answer is hallucinated' contains none of the three
> patterns, so it falls through to 0, factual. Inverted in both directions.
>
> I ran thirteen realistic replies through their exact logic and eight parse
> wrong. I also wrote a two-line fix that gets all thirteen right."

**Impact, stated honestly:** the prompt asks for a bare digit, so most replies
probably parse fine. But `max_tokens` is 512, so verbose replies are possible,
and DeepSeek-R1-Distill — which is in their model list — is a reasoning model
that thinks out loud. The "not sure" option in Table 4 also can essentially
never be recorded from a worded reply. **I can't quantify the real impact without
running their code on their hardware.** Say that; don't overclaim it.

---

## Likely questions

**"Can we use this dataset?"**
Yes. `UTAustin-AIHealth/MedHallu` on HuggingFace, MIT licensed, 10,000 rows,
11 MB, no access request. Two configs: `pqa_labeled` (1,000, expert-annotated
source) and `pqa_artificial` (9,000).

**"What would it cost to run the full thing properly?"**
The evaluation sweep is one decent GPU for a few hours — a single A100 or A6000
rental. Their generation run was 26.5 hours on 4× A6000, but we wouldn't repeat
that since the output is published.

**"Why couldn't you run their code?"**
Both entry points import vLLM at module level and load 7–14B models. vLLM needs
CUDA; my laptop has Intel integrated graphics. Not a config issue — the scripts
can't start.

**"Is the paper good?"**
The problem is real and the dataset is a genuine contribution — first one built
specifically for medical hallucination detection, sensible pipeline. The framing
oversells a bit: no baseline floor reported, and the difficulty label carries the
artifact above. Worth building on, with the caveats.

**"What's the actual task?"**
Binary classification on a single answer. The model sees a question and one
answer — not both side by side — and outputs hallucinated or factual. Two
settings, with and without the PubMed context. Adding context is their biggest
effect, about +0.25 F1 on general models.

**"What next?"**
Score a real model — Anthropic, OpenAI and Ollama backends are wired up, just
need a key or a local pull. Then redo the semantic test sliced by hallucination
*category* instead of difficulty, since category doesn't pass through the
fallback rule and should be cleaner.

---

## Numbers worth memorising

| | |
|---|---|
| Dataset | 10,000 rows; 1,000 labeled + 9,000 artificial |
| Split (labeled) | hard 408, medium 318, easy 274 |
| Their hardware | 4× RTX A6000; 26.5 GPU-hours to generate |
| Constant-baseline F1 | **0.667** |
| Best in paper | GPT-4o, 0.737 no-knowledge / 0.877 with |
| Worst | Med42-8B, 0.416 |
| Knowledge effect | about +0.25 F1, general models |
| Semantic reversal | easy 0.742 vs hard 0.701 embedding cosine, p ~ 1e-23 |
| Parse test | 8 of 13 realistic replies mis-scored |
