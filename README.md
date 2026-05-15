<p align="center">
  <img src="assets/hero.png" alt="NewsReWrite: controllable headline rewriting with dual LLM guidance" width="100%"/>
</p>

<h1 align="center">NewsReWrite</h1>
<h3 align="center">LLM-Guided Headline Rewriting: Controllable Clickability Enhancement Without Clickbait</h3>

<p align="center">
  <em>A research framework for treating clickbait not as a separate genre, but as the over-amplification of legitimate engagement mechanisms, and for rewriting news headlines along a continuous, controllable spectrum between neutrality and editorially responsible engagement.</em>
</p>

---

## 1. Research Motivation

Digital news headlines are the single largest determinant of what readers click, share, and ultimately consume. In algorithmically mediated feeds, a headline must compete for attention inside a cognitive window of less than two seconds, while simultaneously remaining a faithful summary of the underlying article. This produces a structural conflict that sits at the core of contemporary journalism:

> **Engagement and editorial integrity are in tension, but they are not in opposition.**

The prevailing computational treatment of this tension is *clickbait detection*, which frames clickbait as a discrete stylistic category to be classified and filtered. This binary view is empirically unsatisfying. Many of the rhetorical mechanisms associated with clickbait, including curiosity gaps, emotional framing, interrogative phrasing, salience cueing, and emphasis scaling, are also legitimate journalistic tools used by reputable publishers every day. The same device that pulls a reader into a long-form investigation can, when amplified beyond proportion, degenerate into manipulative phrasing.

Clickbait, in other words, is not a *style*. It is a *region* on a continuous engagement spectrum. The research question this project addresses is therefore not "is this headline clickbait?", but rather:

> **Can a language model rewrite a neutral headline so that it occupies any chosen point on the engagement spectrum, under explicit constraints on semantic fidelity and proportional emphasis?**

This reframing turns headline optimization from a classification problem into a *controllable generation* problem, where engagement mechanisms become continuous, addressable control signals rather than discrete labels.

---

## 2. Problem Statement

We formulate headline rewriting as a constrained text generation task:

Given a neutral source headline `h₀` and a target engagement profile `a ∈ {0,1}¹⁰` over ten engagement attributes, produce a rewritten headline `h*` that:

1. Preserves the factual content of `h₀` (semantic fidelity).
2. Amplifies the engagement attributes selected in `a` (positive control).
3. Avoids the over-amplification region characterized by clickbait (negative control).
4. Allows the operator to traverse the spectrum between `h₀` and a maximally engaging variant by varying a small set of guidance weights at inference time, with no retraining of the base language model.

This is a controllable decoding problem rather than a fine-tuning problem, which keeps the generator general-purpose and lets the control signal be swapped, retrained, or audited independently of the underlying LLM.

---

## 3. Method

The framework couples a frozen instruction-tuned LLM with two BERT-based auxiliary discriminators, following the **FUDGE** (Future Discriminators for Generation) paradigm. At each decoding step, the LLM proposes a top-k candidate distribution, and the candidates are rescored using two complementary signals derived from the partially generated prefix:

```
          ┌──────────────────────────────────────────────────────┐
          │                    Base LLM (frozen)                 │
          │              Llama-3-8B-Instruct, top-k decoding     │
          └────────────────────────┬─────────────────────────────┘
                                   │ p_LLM(token | prefix)
                                   ▼
                ┌────────────────────────────────────┐
                │  Prefix-aware combined objective   │
                │  s = p_LLM + α·tac − β·cb          │
                └──────────┬──────────────┬──────────┘
                           │              │
        positive guidance  │              │  negative guidance
                           ▼              ▼
        ┌───────────────────────┐  ┌────────────────────────┐
        │ Engagement-Attribute  │  │ Clickbait Scoring      │
        │ Model (BERT, multi-   │  │ Model (BERT, binary,   │
        │ label, 10 attributes) │  │ prefix-aware)          │
        └───────────────────────┘  └────────────────────────┘
```

The combined per-token score is

```
s(t | prefix) = p_LLM(t | prefix) + α · tac(prefix ⊕ t)  −  β · cb(prefix ⊕ t)
```

where `tac(·) ∈ [0,1]` is the mean activation of the selected engagement attributes under the multi-label discriminator, `cb(·) ∈ [0,1]` is the clickbait probability under the binary discriminator, and `(α, β)` are inference-time control weights. The discriminators are trained on **token-level prefixes**, not only on full headlines, so they can deliver a meaningful signal during early decoding steps when the sequence is still incomplete.

The decoding loop additionally applies (i) a dynamic warm-up on `α` over the first few tokens to avoid premature over-steering, and (ii) a length-aware EOS bonus that discourages truncated or over-extended headlines.

---

## 4. Engagement Attribute Space

Engagement is decomposed into ten linguistically distinct, separately controllable attributes:

| # | Attribute | Operational definition |
|---|-----------|------------------------|
| 0 | Information-gap control (Curiosity Gap) | Signals a specific but unnamed piece of withheld knowledge |
| 1 | Emphasis intensity (Exaggeration) | Amplifies scale or impact via intensity modifiers, no new facts |
| 2 | Emotional framing (Emotional Triggers) | Invokes a named emotion through explicit emotional wording |
| 3 | Salience allocation (Sensationalism) | Heightens drama through framing, without emotional vocabulary |
| 4 | Structural emphasis (Lists / Superlatives) | Uses ranking or list scaffolding |
| 5 | Referential underspecification (Ambiguous References) | Deploys vague pronouns or indefinites |
| 6 | Reader relevance cueing (Direct Appeals) | Addresses the reader or a specific audience directly |
| 7 | Narrative continuation cues (Unfinished Narratives) | Presents the event as ongoing or unresolved |
| 8 | Cross-domain framing (Unexpected Associations) | Links two normally disjoint conceptual domains |
| 9 | Interrogative framing (Provocative Questions) | Uses interrogative syntax to challenge an assumption |

Attributes can co-occur, with compatibility rules that forbid linguistically overlapping pairs (for example, Curiosity Gap with Unfinished Narratives). This makes attribute prediction a multi-label classification task and makes attribute steering a sparse, structured control problem.

---

## 5. Dataset Construction

The training corpus is synthesized in a way designed to isolate the *amplification* signal from confounding topical or stylistic variation.

**Source.** Neutral headlines are drawn from the Kaggle *Fake and Real News* corpus. Each source headline is treated as an editorially neutral baseline and assigned a zero attribute vector.

**Synthetic clickbait generation.** For each source headline, a clickbait-oriented variant is produced by GPT-4o under controlled prompting. The prompt:

1. Samples a sparse subset of 1 to 3 engagement attributes per headline.
2. Instructs the model to amplify only those attributes.
3. Forbids introduction of new facts, entities, numbers, or events.
4. Forces lowercase output, no `!` or `?`, and a length cap relative to the original.
5. Requires that each activated attribute be realized through a *distinct, identifiable* linguistic cue, so that both human readers and downstream classifiers can recover which mechanisms were used.

Each sample is therefore a tuple `(headline_text, clickbait_label, attribute_vector)` with structured, attribute-level supervision. Both full headlines and **token-level prefixes** are used as training instances, which is what enables the discriminators to score partial sequences during FUDGE decoding.

**Splits.** The split is performed *before* any synthetic generation to eliminate leakage:

- Train + Validation: 80% (within this, 85% train / 15% validation)
- Test: 20%

---

## 6. Guide Models

### 6.1 Clickbait Scoring Model
- **Task:** binary classification (clickbait vs neutral) over full headlines and prefixes.
- **Architecture:** BERT encoder with a binary classification head.
- **Role at inference time:** provides the negative guidance signal `cb(·)` that penalizes over-amplification.

### 6.2 Engagement Attribute Model
- **Task:** multi-label classification over the 10-dimensional attribute space.
- **Architecture:** BERT encoder with a multi-label sigmoid head.
- **Role at inference time:** provides the positive guidance signal `tac(·)` by averaging predicted activations over the operator-selected target attributes.

Both models are trained on prefix-augmented data, which is the key design choice that makes them usable as *future discriminators* over partial generations.

---

## 7. Evaluation

The evaluation deliberately separates three orthogonal axes, because a successful rewrite must satisfy all three simultaneously.

**Clickbait scoring model.** Accuracy, Precision, Recall, F1, ROC-AUC.

**Engagement attribute model.** Macro-F1, Micro-F1, and per-attribute confusion analysis to surface which mechanisms remain underdetected or systematically conflated.

**Generated rewrites.**
- *Semantic fidelity*: STS (Sentence-Transformers) cosine similarity and BERTScore (P / R / F1) against the source headline.
- *Engagement realization*: attribute-recovery rate using the engagement attribute model on the rewritten output.
- *Editorial responsibility*: clickbait-probability shift relative to a naive maximum-engagement baseline.
- *LLM-as-judge neutralization audit*: a GPT-based pass that flags hallucinated facts, entity drift, and meaning-altering rewrites.

Sweeps over `(α, β, top-k)` produce a Pareto front along the engagement-fidelity-responsibility trade-off, which is the empirical object of interest.

---

## 8. Repository Structure

```
NewsReWrite/
├── Dataset_generation/
│   ├── Split_Data.py                       # leakage-safe train/val/test split
│   ├── generate_clickbait.py               # GPT-4o-driven synthetic clickbait generation
│   ├── clickbait_methods.json              # canonical attribute catalog
│   ├── compute_tactics_distribution.py     # attribute coverage diagnostics
│   ├── create_prefix_dataset_train_val.py  # prefix expansion for train/val
│   ├── create_prefix_dataset_test.py       # prefix expansion for test
│   ├── clickbait_prefix_trainval.py        # binary-classifier prefix dataset
│   └── clickbait_prefix_test.py            # binary-classifier prefix test set
│
├── models/
│   ├── bert_train_binary_prefix.py         # train clickbait scoring model
│   ├── bert_test_binary_prefix.py          # evaluate clickbait scoring model
│   ├── bert_train_tactics_prefix.py        # train engagement-attribute model
│   ├── bert_test_tactics_prefix.py         # evaluate engagement-attribute model
│   └── fudge_controlled_generation.py      # FUDGE decoding loop with dual guidance
│
├── evaluation/
│   └── evaluate_neutralization.py          # fidelity + responsibility audit (STS, BERTScore, LLM-judge)
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 9. Getting Started

```bash
git clone https://github.com/ApartsinProjects/NewsReWrite.git
cd NewsReWrite
pip install -r requirements.txt
```

Create a `.env` file at the project root with:

```
OPENAI_API_KEY=...
OPENAI_API_URL=https://api.openai.com/v1
OPENAI_API_MODEL=gpt-4o
```

A reproducible end-to-end run consists of (1) `Dataset_generation/Split_Data.py`, (2) `Dataset_generation/generate_clickbait.py`, (3) the prefix-expansion scripts, (4) the two BERT training scripts in `models/`, and (5) `models/fudge_controlled_generation.py` for guided decoding. `evaluation/evaluate_neutralization.py` then runs the post-hoc fidelity audit.

---

## 10. Why This Framing Matters

Most existing work on clickbait treats it as a problem of *filtering bad content*. This project treats it as a problem of *calibrating a continuous engagement dial*, which has three consequences worth stating explicitly:

1. **The control signal is auditable.** The two BERT discriminators expose *which* mechanism is being amplified and *how strongly*, rather than producing a single opaque "engagement score".
2. **The base generator stays general.** Because steering happens at decoding time, the same LLM can be repurposed for other constrained-style tasks without retraining.
3. **Editorial responsibility becomes a tunable parameter.** The same system can produce a neutral paraphrase, a moderately engaging rewrite, or an aggressively engaging one, and the trade-off is explicit rather than implicit in the training data.

The intended contribution is therefore both methodological (a concrete instantiation of attribute-level controllable decoding for journalistic text) and conceptual (recasting clickbait as a region on an engagement manifold rather than a binary label).

---

## 11. Authors

* **Linoy Halifa**
* **Sagiv Bar**

## 12. Project Leads

* **Sasha Apartsin**
* **Yehudit Aperstein**
