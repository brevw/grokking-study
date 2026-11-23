# Sequential Grokking Across Group Sizes: A Measurement Study

*Working plan for the group (Bogdan, Elvan, Lars, Ahmed). This is a plan, not paper text. Write the paper yourselves, as Tiberiu advised.*

---

## 0. TL;DR

We first train a small transformer until it **groks modular addition in Z_p**. We then **keep training the same network on modular addition in Z_q**, where `q ≠ p` is another prime or a composite number. Throughout both phases we **measure** three things:

1. **Speed:** does the second task grok faster or slower than it does from scratch?
2. **Retention:** how much of the first task does the network forget?
3. **Representation:** do the Fourier features learned for Z_p survive, get re-tuned to Z_q, or get replaced?

We frame this as a **continual learning** problem. The questions are whether earlier training gives positive transfer or causes primacy bias / loss of plasticity, and whether the old task is forgotten. Grokking is a convenient test case for these questions because generalization in it is sharp and easy to detect. The project is empirical and centred on measurement. Theory is limited to the minimum needed to design the metrics.

---

## 1. What changes compared to the proposal, and why

| Proposal | Feedback | New plan |
|---|---|---|
| Serial / parallel training on **different operations** (`+` then `×`) on the same Z_97 | Furuta et al. (2024, in `papers/`) already study this: pre-grokked embeddings/transformers transferred between `+`, `−`, `×`, and multi-task "co-grokking". Our novelty would be weak. | Keep the **operation fixed** (addition) and change the **group**: Z_p → Z_q. Tiberiu explicitly suggested this. |
| Discrete logarithm / `p ± 1` moduli | "Recommend against the discrete log"; "suggest another prime modulus rather than ±1; composite moduli also worth studying". | Drop the discrete log. Use target moduli **q that are further away**: other primes plus composites. |
| Motivation: "grokking of multiple operations is unexplored" | Contextualise as **continual learning**; there may be existing works. | Motivate with continual learning: catastrophic forgetting, loss of plasticity, primacy bias, warm-starting. Lyle et al. (2025, in `papers/`) is the closest bridge. |
| Only p = 97 | Fine for the project; to publish, show the result holds for other moduli. | Main grid uses p = 97. A smaller **robustness grid** uses a second source prime. |
| Language symbols vs group elements | He liked it. | Keep it and make it **central to the method**: tokens (symbols) and group elements are separate objects, and we explicitly control the map between them (Section 3.2). |
| Heavy mechanistic / theory parts | (we want less theory) | Replace them with **cheap, quantitative progress measures** that we log at every evaluation step (Section 5). |

**Why ±1 is a bad choice (worth one sentence in the paper):** if token `i` stands for element `i` in both groups, then `(a+b) mod p` and `(a+b) mod q` give the **same answer whenever `a+b < min(p,q)`**. For q = p+1 that covers about half of all pairs, so "transfer" would mostly mean "the old answers are still correct". We avoid this by choosing q further from p, and we **always report accuracy separately** on pairs where the old and new answers agree vs. disagree (Section 5.1).

---

## 2. Research questions and hypotheses

- **RQ1 (Forward transfer / plasticity).** Is the grokking time on Z_q after grokking Z_p shorter or longer than from a random initialization?
  - *H1a (positive transfer):* the "clock" machinery (embeddings on circles + trig-identity MLP) is reused, so grokking is faster.
  - *H1b (primacy bias):* the network is stuck in a low-norm, well-fitted solution for Z_p (low effective learning rate, Lyle et al.), so grokking is slower than from scratch.
  - We don't need to pick one. The measurement decides, and the answer may depend on q.
- **RQ2 (Does the source state matter?).** Is starting from a *grokked* network different from starting from a network that has only *memorized* Z_p?
- **RQ3 (What transfers: symbols or structure?).** Does any speed-up survive when we randomly permute which token stands for which element of Z_q? If it does, the network transfers *structure* (circular features). If it does not, it transfers *token identity*.
- **RQ4 (Prime vs composite).** For composite q (e.g. 91 = 7·13), does the network prefer Fourier frequencies that live in subgroups (`gcd(k, q) > 1`)? Is transfer from a prime p different from transfer into a composite q?
- **RQ5 (Backward transfer / forgetting).** How fast, and how completely, is Z_p forgotten while learning Z_q? Does forgetting happen before or after Z_q groks?

---

## 3. Setup

### 3.1 Task and data

- Operation: addition, `c = (a + b) mod n` for a modulus n (= p in phase A, = q in phase B).
- Dataset for modulus n: all `n²` pairs `(a, b)`. Use a **fixed random split** (per seed) into train fraction `α` and validation `1−α`.
- Main setting: `α = 0.3` (as in Nanda et al.; grokking is clear and reasonably fast). Secondary: `α = 0.5`.
- Both phases use the **same α**. Each phase has its own split.

### 3.2 Symbols vs. group elements (the key design choice)

Separate the **vocabulary** (symbols the network sees) from the **group** (what the symbols mean):

- **Vocabulary:** one shared token set `{0, 1, …, V−1}` plus special tokens, with `V ≥ max(all moduli used)`. For example, V = 128 covers every modulus up to 127. The architecture is **identical in every run**. Tokens a task doesn't use are never seen.
- **Equals / task tokens:** one special token **per modulus**, `=_n`. The input is the sequence `[a, b, =_n]` and the prediction is read at the last position.
  - *Why:* without it, Z_p and Z_q give **conflicting labels for the same input** `(a, b)`. Forgetting would then be forced by definition and couldn't be measured meaningfully. With `=_p` vs `=_q` the network *could* keep both, so forgetting becomes a real measurement. (In phase B, `=_q` is a fresh, untrained token.)
- **Symbol map** `σ_n : Z_n → tokens`:
  - **Identity map** (default): element `i` ↔ token `i`.
  - **Permuted map** (control for RQ3): in phase B, element `i` ↔ token `π(i)` for a random permutation `π`, drawn once per seed. The task is mathematically the same, but no token keeps its old "meaning".
- **Output:** logits over the V number tokens, **masked to the valid targets** of the current modulus (the first n tokens under the identity map, or `σ_n(Z_n)` under a permutation). This keeps the cross-entropy comparable across moduli.

### 3.3 Model and optimizer (fixed for every run)

Follow Nanda et al. (2023) to stay close to a well-understood baseline:

- 1-layer decoder-only transformer, `d_model = 128`, 4 heads (d_head = 32), MLP width 512, ReLU, learned positional embeddings, **no LayerNorm**, untied embed/unembed.
- AdamW, lr `1e-3`, weight decay `1.0`, betas (0.9, 0.98), full-batch gradient descent, short linear warm-up (10 steps).
- **At the start of phase B, reset the optimizer** (new AdamW state, same warm-up). This is the standard "warm-start" protocol, and it means the comparison with training from scratch differs only in the initial weights. Note this in the paper.
- Step budget: phase A up to 30k steps; phase B up to **50k steps**. Runs that don't grok within the budget are **censored**, not dropped (Section 6).

Optional (only if time allows): repeat the main comparison with a 2-layer MLP (Gromov 2023) to check the result isn't architecture-specific.

### 3.4 Moduli

- **Source (phase A):** p = 97 (main). Robustness: p = 59 or 113.
- **Targets (phase B), main grid for p = 97:**

| Type | q | Why |
|---|---|---|
| Prime, smaller | 59, 89 | removes tokens; 89 is fairly close |
| Prime, larger | 113 | adds unseen tokens 97…112 |
| Composite | 91 = 7·13 | Z_91 ≅ Z_7 × Z_13 (CRT), product of two primes |
| Composite | 96 = 2⁵·3 | many subgroups, close to 97 |
| Composite | 121 = 11² | prime power, adds new tokens |
| Control | 97 (same task) | sanity check: should grok immediately |

Keep `q ≠ p ± 1`. 96 is close to 97, so its agreement-set fraction will be high. Analyse it with that in mind, or replace it with 100 = 2²·5².

---

## 4. Experimental conditions

For each target q, each seed, train on Z_q under these conditions (the first three are required):

| ID | Name | Initial weights for phase B | Answers |
|---|---|---|---|
| **C0** | Scratch | random init | baseline grokking time `T_grok^scratch(q)` |
| **C1** | Warm, grokked | network trained on Z_p **until grokked + 2k steps of cleanup** | RQ1, RQ5 |
| **C2** | Warm, memorized | network trained on Z_p **until train acc ≥ 99% but val acc < 20%** | RQ2 |
| **C3** | Warm, grokked, permuted symbols | as C1, but phase B uses a random token permutation `π` | RQ3 |
| C4 | Warm, grokked, re-init embeddings | as C1, but the embedding (and unembedding) rows are reset to random | where transfer lives: embeddings vs. the rest |
| C5 (optional) | Joint training | train on Z_p ∪ Z_q from scratch (mix both datasets) | continual-learning upper reference (no ordering) |
| C6 (optional) | Warm + ELR re-warm / shrink-and-perturb | as C1, plus a standard plasticity fix (Lyle et al. 2025, Ash & Adams 2020) | do continual-learning fixes help if C1 is slow? |

Implementation notes:

- C1 and C2 checkpoints come from the **same phase-A run** (save both moments). That keeps the source network the same between conditions.
- Use the **same seed for phase-B data split / permutation** across C0–C4. Then, for a given seed, the only difference between conditions is the starting weights.

---

## 5. What we measure (the methodology core)

Log **every 50 steps** (evaluation on the full train and val sets is cheap at this size). Save **weights every 500 steps** for the post-hoc representation metrics.

### 5.1 Behavioural metrics

| Metric | Definition |
|---|---|
| Train / val accuracy and loss on **Z_q** | standard |
| Val accuracy on **Z_p** (with `=_p`) during phase B | forgetting curve (RQ5) |
| **Agreement split** | val acc on Z_q split into pairs where `(a+b) mod p == (a+b) mod q` (old answer still right) vs. not. *Only meaningful under the identity map.* |
| Accuracy on pairs involving **new tokens** (a or b ≥ p, when q > p) | do never-seen tokens get integrated? |
| Step-0 accuracy on Z_q | how much of Z_q the network solves "for free" before any phase-B training |

### 5.2 Derived timing metrics (the main numbers in the paper)

- `T_mem` = first step at which **train** acc ≥ 99%.
- `T_grok` = first step at which **val** acc ≥ 99% **and stays ≥ 99% for the next 1k steps** (to avoid counting spikes). Also report `T_95` (val ≥ 95%) as a less noisy alternative.
- `Δ = T_grok − T_mem`: the grokking delay itself.
- **Transfer ratio** `R = T_grok^scratch / T_grok^warm`, computed per seed pair. `R > 1` means the earlier training helped (positive transfer). `R < 1` means it hurt (primacy bias).
- **Forgetting:** `F = acc_p(start of B) − acc_p(end of B)`, and `T_forget`, the step at which Z_p val acc falls below 50%. Also note whether `T_forget` comes before or after `T_grok(q)`.

### 5.3 Representation metrics (cheap progress measures, computed from saved weights)

All of these are a few lines of NumPy on the embedding matrix `W_E` (rows = tokens).

1. **Fourier spectrum of the embeddings.** Take the rows for the tokens of the current group, ordered by group element `0…n−1` (i.e. undo `σ_n`). Compute the DFT over Z_n along the token axis, and the power per frequency `k = 1…⌊n/2⌋`.
   - **Key frequencies** `K_n`: the smallest set of frequencies that holds ≥ 90% of the non-constant power.
   - **Fourier sparsity** = power in the top-5 frequencies / total power. High sparsity = clean clock-like representation (Nanda et al.; Furuta et al.'s "Fourier frequency density" is similar).
2. **Old-frequency retention.** At the end of phase A, take the key frequencies `K_p`. During phase B, fit the embedding rows of tokens `0…min(p,q)−1` with the basis `{cos(2πka/p), sin(2πka/p) : k ∈ K_p}` by least squares and report **R²**. This shows how much of the old Z_p circle structure is still present.
3. **Frequency matching (new-vs-old).** For each new key frequency `k'/q`, find the nearest old one `k/p` in real-frequency terms. Compare the mean distance to the scratch runs. If warm runs land systematically closer than scratch runs, the network "re-tunes" its old circles instead of building new ones.
4. **Subgroup frequencies (composite q).** The fraction of `K_q` with `gcd(k, q) > 1`, for scratch vs. warm runs (RQ4).
5. **Representation drift.** Linear CKA between the embeddings (and MLP activations) at the start of phase B and at step t. Also the relative weight distance `‖θ_t − θ_A‖ / ‖θ_A‖`.
6. **Plasticity indicators** (from the continual-learning literature):
   - total parameter norm `‖θ‖₂` (weight decay drives it during grokking),
   - **effective learning rate** ≈ `‖Δθ_t‖ / ‖θ_t‖` per step (Lyle et al. 2025),
   - fraction of **dead MLP neurons** (never active over the full dataset).

   These are our candidates for *explaining* any primacy bias we see in RQ1.

Optional, only if everything else is done: the restricted/excluded loss of Nanda et al. for the key frequencies, or the clock vs. pizza distinction of Zhong et al.

---

## 6. Statistics and reporting

- **Seeds:** 5 seeds per (q, condition). The seed controls the init, the data split, and the permutation `π`.
- **Censoring:** runs that don't grok within the budget get `T_grok = budget` and are **marked as censored**. Report the fraction that grokked, and plot **survival curves** (fraction not yet grokked vs. step) per condition. These handle censoring correctly and are more informative than means.
- **Summary statistic:** median `T_grok` with interquartile range, and the median transfer ratio `R` with a bootstrap 95% CI (resample seeds).
- **Comparisons:** paired by seed (C1 vs C0 with the same phase-B split). A Wilcoxon signed-rank test, or just bootstrap CIs, is enough. Don't over-test, because the grid is small.
- **Log-scale x-axis** on all training curves. Grokking times vary by orders of magnitude.

---

## 7. Step-by-step work plan

### Step 1: Infrastructure and sanity checks (week 1)
1. Write `data.py`: builds datasets for any `n`, any symbol map `σ_n`, with `=_n` tokens and a seeded split.
2. Write `model.py`: the 1-layer transformer of Section 3.3 with a fixed vocabulary V = 128 + specials.
3. Write `train.py`: one function `train(init_weights_or_None, task_config, budget, log_every=50, ckpt_every=500)` that logs every metric of Section 5.1 to a CSV/JSON (or Weights & Biases).
4. **Reproduce standard grokking** on Z_97 (α = 0.3, wd = 1): train acc → 100% within ~1k steps, val acc jumps later. Measure wall-clock per step, then **recompute the compute budget from that number**.
5. Sanity: same-task control (p = q = 97, new `=` token). The warm network should reach 99% val almost immediately.

### Step 2: Phase-A source models (week 1–2)
6. Train 5 seeds on Z_97. Save the **memorized** checkpoint (C2) and the **grokked + cleanup** checkpoint (C1). Record `T_mem`, `T_grok`, and the Fourier metrics for each.

### Step 3: Main grid (week 2–3)
7. For every target q in Section 3.4, run C0, C1, C2, C3 for all 5 seeds. That is 6 targets × 4 conditions × 5 seeds = **120 runs**. Run C4 if time allows (+30).
8. Produce the main figure as soon as the first q is done, so problems show up early.

### Step 4: Representation analysis (week 3–4)
9. Run the metrics of Section 5.3 on the saved checkpoints (offline, CPU is fine).
10. Plot old-frequency retention, Fourier sparsity, and effective learning rate over phase-B steps, overlaid with the val-accuracy curves. The key question: does the representation change **before**, **during**, or **after** the jump in accuracy?

### Step 5: Robustness (week 4)
11. Repeat a reduced grid (C0, C1 × 3 targets × 3–5 seeds) with **a second source prime** (59 or 113) and, if possible, `α = 0.5`. The goal is to check that the *sign* of the effect (R > 1 or R < 1) is the same.
12. Optional: C5 (joint) and C6 (ELR re-warm) for one or two q.

### Step 6: Writing (week 5)
13. Write the paper **yourselves**, starting from the figures. Use AI for lit search, code, and checking, not for prose.

**Compute (rough, re-estimate in Step 1):** one run of ≤ 50k full-batch steps on ≤ 128² × 0.3 ≈ 5k examples takes a few minutes on one GPU. That gives about 150 main runs plus about 60 robustness runs, so on the order of **10–20 GPU-hours**. Several small runs fit on one GPU at the same time.

---

## 8. Planned figures and tables

1. **Main figure:** for each q, phase-B val accuracy vs. step (log x) for C0/C1/C2/C3. Median across seeds, shaded IQR.
2. **Transfer ratio plot:** `R` (log scale, with CI) per q, grouped into prime-smaller / prime-larger / composite. A horizontal line at R = 1.
3. **Forgetting plot:** Z_p val acc and Z_q val acc on the same axes during phase B, with `T_forget` and `T_grok(q)` marked.
4. **Representation dynamics:** old-frequency R², Fourier sparsity, and effective LR over phase-B steps (C1 vs C0).
5. **Table:** median `T_mem`, `T_grok`, `Δ`, `R`, `F`, and the % that grokked, for every (q, condition).
6. **Composite panel (RQ4):** the fraction of subgroup frequencies `gcd(k,q) > 1` in the learned key frequencies, scratch vs. warm.
7. **Robustness panel:** the transfer ratio R for the second source prime.

---

## 9. Related work to cover (and check for overlap!)

Tiberiu stressed that novelty must be crystal clear. **Before writing**, search specifically for "grokking + continual learning / transfer / warm-start / different modulus". Read the abstracts and say explicitly how we differ.

**Grokking and its mechanics (in `papers/`):**
- Power et al. 2022: original grokking.
- Nanda et al. 2023: Fourier/clock algorithm, progress measures.
- Zhong et al. 2023: clock vs. pizza, multiple solutions.
- Gromov 2023: analytic solution, MLP.
- **Furuta et al. 2024: transfer of pre-grokked models across *operations* and multi-task co-grokking. Closest prior work. We differ by changing the *group / modulus* with the operation fixed, by separating symbols from elements, and by the continual-learning framing (forgetting + plasticity).**
- **Lyle et al. 2025: grokking ↔ primacy bias / warm-starting, effective learning rate. Closest continual-learning bridge. Their grokking experiments are stationary, while ours are sequential across groups.**

**Continual learning (to add):**
- Catastrophic forgetting: McCloskey & Cohen 1989; Kirkpatrick et al. 2017 (EWC).
- Warm-starting hurts generalization: Ash & Adams 2020 (shrink-and-perturb).
- Primacy bias: Nikishin et al. 2022.
- Loss of plasticity: Lyle et al. 2023; Dohare et al. 2024 (Nature).
- Critical learning periods: Achille et al. 2019.

**Grokking papers to check for overlap (verify each one yourself):**
- Liu et al. 2022 (effective theory of representation learning) and "Omnigrok".
- Works on transferring embeddings to accelerate grokking (e.g. "Let Me Grok for You", 2025).
- Varma et al. 2023 (circuit efficiency), Doshi et al. 2024 (grokking with corrupted labels), and any work on grokking with multiple / changing moduli.

---

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Phase B never groks within the budget for some q | Treat it as a result (strong primacy bias). Report it censored. Optionally run C6 to see if a plasticity fix recovers it. |
| Everything groks equally fast, so there is no effect | Lower α (harder task, longer grokking) or wd = 0.1, both of which make transfer effects larger. A null result measured properly is still a valid project result. |
| Fourier metrics are noisy for composite q | Report the behavioural metrics as the main result and keep the representation metrics as supporting evidence. |
| Compute is tighter than expected | Cut C2/C4, use 3 seeds, and keep only q ∈ {59, 113, 91, 121}. |
