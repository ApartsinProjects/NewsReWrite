# B6: DExperts and GeDi decoding-time baselines (reviewer R2-M4)

Reviewer R2-M4 asked for comparisons against the related decoding-time
controllers DExperts and GeDi. These are now IMPLEMENTED (not stubs).

Both use a small `meta-llama/Llama-3.2-1B` controller that shares the base
generator's 128k Llama-3 tokenizer, so their logits align with
`Meta-Llama-3-8B-Instruct`. The control corpora are the two halves of the
synthetic dataset: the clickbait headlines (the engagement-bearing class) and
the neutral source headlines. Both produce rewrites in the exact b2 CSV schema,
so `score_rewrites.py` scores them unchanged and `b8_method_comparison/
compare_methods.py` contrasts every method with significance + a Pareto plot.

## DExperts (Liu et al. 2021)

- `train_dexperts.py`: fine-tunes two 1B causal LMs, an expert on clickbait
  headlines and an anti-expert on neutral headlines.
- `run_dexperts.py`: at each step,
  `final_logits = base_logits + alpha*(expert_logits - antiexpert_logits)`,
  restricted to the base top-k, greedy. Sweeps alpha.

## GeDi (Krause et al. 2021)

- `train_gedi.py`: fine-tunes one class-conditional 1B LM on control-code
  prefixed headlines (`<clickbait> ...`, `<neutral> ...`).
- `run_gedi.py`: per candidate token,
  `log P(desired | prefix, v) = log_softmax([lp_desired, lp_other])[0]`,
  and `score(v) = log P_base(v) + omega * log P(desired | v)`. Sweeps omega.

## Interpretation (for the paper, not the code)

DExperts and GeDi here are single-axis controllers (clickbait vs neutral),
whereas FUDGE uses two discriminators (clickbait scorer + tactics model). The
comparison is expected to show that FUDGE's dual-signal control traces a better
fidelity-vs-clickbait tradeoff than the single-axis baselines on the same
held-out set. That is the point of the R2-M4 comparison.

## Modal run order

```
modal run app.py::train_dexperts        # T4, trains expert + anti-expert
modal run app.py::run_dexperts --n-items 300
modal run app.py::run_score --in-subdir b6_dexperts
modal run app.py::train_gedi            # T4, trains class-conditional LM
modal run app.py::run_gedi --n-items 300
modal run app.py::run_score --in-subdir b6_gedi
modal run app.py::compare_methods       # FUDGE vs prompt-only vs DExperts vs GeDi
```

## Caveat

`meta-llama/Llama-3.2-1B` is gated. The HF account behind the `huggingface`
Modal secret must have accepted its license at
https://huggingface.co/meta-llama/Llama-3.2-1B. If not, the trainers exit with
an actionable message.
