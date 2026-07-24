# Diagnostic: base-model transfer of the dual guides (not a paper result)

Second-base-model generality probe on **Qwen2.5-7B-Instruct** (guides reused
unchanged, same 150 Reuters neutrals). Independent DistilBERT clickbait detector.

## Plain `log` objective, alpha=4, beta=2 (paper operating point)
| cell | clickbait | BERTScore | perplexity |
|---|---|---|---|
| (0,0) | 0.041 | 0.464 | 132 |
| (4,0) | 0.063 | 0.443 | 133 |
| (0,2) | 0.071 | 0.424 | 157 |
| (4,2) | 0.082 | 0.406 | 159 |

**Transfers:** engagement guide raises tactic realization (0.30->0.35), and
fidelity/fluency are *better* than Llama (BERTScore 0.41-0.46 vs 0.25-0.28;
perplexity 132-159 vs 175-275). Clickbait stays low (0.04-0.08).
**Does not transfer:** the clickbait brake. (0,2)-(0,0) = +0.03 (wrong sign).

## `lognorm` objective (per-step std-normalized LLM term), alpha=4, beta=2
| cell | clickbait |
|---|---|
| (0,0) | 0.041 (identical; sigma-scaling preserves argmax with no guides) |
| (0,2) | 0.113 |
| (4,0) | 0.121 |
| (4,2) | 0.223 |

brake: (0,2)-(0,0) = +0.072, (4,2)-(4,0) = +0.101. **Over-steers** (clickbait up
everywhere, rewrite length 83->166 chars). The brake fails harder, not less.

## Root cause
1. Dividing the LLM log-prob term by its per-step std makes the guides ~2-3x
   stronger, so alpha=4,beta=2 is now over-aggressive (a re-tune issue).
2. Deeper: the brake term beta*log(1-cb) is nearly flat for small cb
   (log(1-cb) ~= -cb). Qwen already produces low-clickbait text (~0.04), so the
   brake has almost no gradient to act on, while the engagement push dominates
   and raises clickbait. **The brake's effect is intrinsically base-model-
   dependent**: it needs a base model that actually produces clickbait to
   suppress. Normalization cannot manufacture that headroom.

## Conclusion
Report as an honest limitation, not a generality claim. The engagement guide and
high fidelity/fluency transfer to a second base model; the clickbait brake does
not transfer at reused weights and depends on the base model's own clickbait
propensity. The `lognorm` code is kept as an experimental option (does not affect
the default `log` objective or any reported number).
