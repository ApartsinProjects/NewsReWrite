# Strengthening plan — Tier 1/2/3 additions (excluding human raters)

Goal: harden the paper against the reviewer concerns and reviewer-2's statistics
point, and broaden the generality claims, using (a) data already on disk and
(b) a small number of Modal generation runs that reuse the existing b1–b8
pipeline and the already-trained guides. Effort is given as **agent execution
time + cloud wall-clock + $**, not human-developer time.

## Summary

| # | Item | Tier | Type | Agent time | Cloud | $ | Lands in paper | Narrative risk |
|---|------|------|------|-----------|-------|---|----------------|----------------|
| 5 | Factuality-judge calibration table | 3 | local | ~10 min | — | 0 | Table 8 + §4.11 | none |
| 6 | Tactic-agnostic sweep (prompt-plus-guidance, measured) | 3 | local | ~10 min | — | 0 | §5.1 (iv) + mini-table | none (honest limitation) |
| 8 | Effect sizes + multiple-comparison correction | 3 | local | ~30 min | — | 0 | Tables 5/6 + stats note | none |
| 7 | Engagement-continuum figure (one headline across α) | 3 | local | ~20 min | — | 0 | Figure 7 | low (illustrative) |
| 9 | Tactic-judge vs rubric labels validation | 3 | local(+tiny API) | ~20 min | — | ~$1 | §3.7 sentence | low |
| 4 | Second domain (rewrite human-authored non-Reuters neutrals) | 2 | Modal | ~45 min orch | ~1–1.5 h | ~$10–20 | Table 9 + §4.12 | medium (may transfer weaker → honest) |
| 3 | Second base generator (Mistral-7B / Qwen2.5-7B) | 2 | Modal | ~45 min orch | ~1–2 h | ~$10–20 | Table 10 / added cols | medium |
| 2 | Fair dual-control baseline (brake for DExperts/GeDi) | 1 | Modal | ~1 h orch | ~3–5 h | ~$20–40 | Table 6 rows + §4.9 | **high** (could narrow our Pareto edge) |

Totals if all run: ~4–5 h of my orchestration, ~6–9 h cloud wall-clock (mostly
unattended), ~$50–80 compute.

---

## Tier 3 — free wins from data already on disk (no new compute)

### 5. Factuality-judge calibration table
- **Why:** the new-fact judge that underpins the "no fabricated facts" claim
  (§4.11) is currently one sentence. Its calibration is strong and buried.
- **Data:** `results/control/control_validation.json` (n=120): new-fact rate on
  faithful paraphrases 0.025, faithful questions 0.025, fact-injected controls
  1.0; NLI separates faithful vs unfaithful at AUROC 0.997 (RoBERTa) / 1.00
  (DeBERTa).
- **Steps:** build a 3-row table from the JSON; add one sentence in §4.11.
- **Output → paper:** Table 8 "Calibration of the new-fact judge and NLI".

### 6. Tactic-agnostic sweep = measured "prompt-plus-guidance"
- **Why:** turns limitation (iv) from an assertion into a number, pre-empting
  "the guides do nothing without the prompt".
- **Data:** `results/b2_sweep_neutral/` (guides applied with a generic,
  tactic-agnostic prompt). Tactic realisation moves only 0.30 → 0.33 as α goes
  0 → 4, versus 0.44 → 0.51 with the tactic-naming prompt; clickbait still falls
  with β.
- **Steps:** compute the α-sweep means for both prompt regimes; add a 2-column
  mini-table and one paragraph to §5.1.
- **Output → paper:** small table + sharpened limitation (iv).

### 8. Effect sizes + multiple-comparison correction
- **Why:** reviewer 2: "no statistical analysis (single runs, no variance, no
  significance testing)". We already added bootstrap CIs and Wilcoxon p-values;
  effect sizes + correction complete it.
- **Data:** per-item CSVs (`b2`, `b4`, `b6_dexperts`, `b6_gedi`).
- **Steps:** for every reported Wilcoxon test add rank-biserial / Cliff's δ
  effect size; Holm–Bonferroni-correct the family of cross-method p-values;
  note the correction in the captions.
- **Output → paper:** effect-size column in the significance rows + one stats
  sentence in §3.7; full table to the deposit.

### 7. Engagement-continuum figure
- **Why:** makes graded control visceral in a way tables do not.
- **Data:** `results/b2_sweep/` — pick one source headline present across the
  grid; show its rewrite at α = 0,1,2,3,4 (β = 0), then the same at (4,2).
- **Steps:** select a clean item with complete cells; render a small labelled
  panel (source → increasing α → braked).
- **Output → paper:** Figure 7 (engagement continuum).

### 9. Tactic-judge vs rubric-intended labels
- **Why:** the LLM tactic-judge is currently an unvalidated oracle.
- **Data:** `results/b2/judge_raw.jsonl` + the intended tactic per item.
- **Steps:** compare the judge's tactic-present decisions to the intended tactic
  on items where the tactic is unambiguous; report agreement (F1/κ). (Optional
  tiny re-judge of the 150 C1 items, ~$1.) Note this is strongest once real
  human labels exist.
- **Output → paper:** one validating sentence in §3.7.

---

## Tier 2 — generality (Modal, reuse existing pipeline + guides)

### 4. Second domain — rewrite human-authored, non-Reuters neutral headlines
- **Why:** answers reviewer-2 minor #2 (Reuters-only restriction) directly, and
  needs **no new data**: the human-authored *non-clickbait* headlines from
  Chakraborty (`data/external/chakraborty16/non_clickbait_data`) and Webis are
  already local and are a different domain/register from Reuters newswire.
- **Method:** sample ~300 of these neutral non-Reuters headlines; run the
  existing FUDGE rewriting (same guides, α = 4, β = 2, plus the (0,0) baseline)
  via the `b2_held_out_rewrite_eval` app; score with the same independent
  metrics (external clickbait detector, BERTScore, NLI, perplexity, LLM judge).
- **Output → paper:** Table 9 "Cross-domain rewriting on human-authored
  non-Reuters neutrals" + §4.12 paragraph; if strong → also a generality line in
  the abstract/conclusion.
- **Risk (wins-only):** guides trained on Reuters may transfer less well; either
  way it is informative. If strong → headline; if weak → honest, quantified
  limitation.

### 3. Second base generator (model-agnosticism)
- **Why:** shows the framework is not Llama-specific; the guides are BERT-based
  and reusable, so only the base generator changes.
- **Method:** swap the base model in `b2_held_out_rewrite_eval` to
  Mistral-7B-Instruct-v0.3 (fallback Qwen2.5-7B-Instruct); re-run the same 300
  held-out items at (0,0) and (4,2) with the same guides; score identically;
  compare the Pareto behaviour to Llama-3-8B.
- **Output → paper:** Table 10 (or added rows to Table 6) "FUDGE on a second
  base generator".
- **Risk:** low–medium; expected to reproduce the pattern.

---

## Tier 1 — fair dual-control baseline (highest impact, highest risk)

### 2. Give DExperts and GeDi a clickbait brake
- **Why:** the strongest possible answer to reviewer-2 concern #4 ("is the
  dual-guidance machinery worth it over simpler alternatives"). Right now the
  baselines steer positively only, so the comparison is dual-vs-single.
- **Method (two options):**
  1. *Principled:* train a small clickbait "expert" LM (Llama-3.2-1B on
     clickbait text) and add a negative term to DExperts; add a not-clickbait
     control code to GeDi; re-run the strength sweeps; score identically.
  2. *Lighter:* add a post-hoc clickbait rerank/reject to the baseline candidate
     pool (reject high-clickbait continuations) as a dual-control-equivalent.
     Cheaper, less principled; can be reported as an approximation.
- **Output → paper:** extra rows in Table 6 (DExperts+brake, GeDi+brake) + a
  paragraph in §4.9 and a sentence in the conclusion.
- **Risk (wins-only, must flag):** a braked baseline could close the clickbait
  gap and narrow FUDGE's Pareto edge to fidelity/fluency. If FUDGE still
  dominates (likely, because the baselines only reach the tactic level by
  over-steering, which wrecks fidelity/fluency regardless of a brake) → strong
  result. If a braked baseline catches up → we report it honestly and re-scope
  the claim to "comparable control at higher fidelity/fluency with a single
  integrated mechanism". Recommend running the lighter option 2 first as a cheap
  probe before committing to option 1.

---

## Sequencing

1. **Now, local, no approval needed:** #5, #6, #8, #7, #9 (Tier 3). Rebuild HTML,
   re-run the number audit, push. ~1–1.5 h agent time, $0.
2. **Modal batch A (generality):** #4 then #3. Reuse the deployed apps; cap
   concurrency; idempotent + resumable. ~$20–40, ~2–3.5 h cloud.
3. **Modal batch B (highest risk):** #2 — run option 2 (rerank probe) first;
   only escalate to option 1 if the probe is promising. ~$20–40.

## Integration after each batch (standing checklist)
- Re-run the bootstrap-CI + number-audit script against the new per-item data.
- Rebuild `paper/current` HTML + `docs/index.html`; keep new material
  yellow-highlighted; re-run the 5-point audit (nothing lost, highlights,
  reviewer coverage, no leftovers, reads as a paper).
- Update the Zenodo deposit manifest with any new result files.
- Regenerate the DOCX/PDF from the final HTML (one pass at the end, not per
  batch).

## GPU / cost guardrails (per project conventions)
- Cloud-first (Modal); local 6 GB RTX 2060 only for the CPU-scale scoring already
  used for figures. One GPU task at a time locally.
- Cap Modal concurrency ~6–8; idempotent "skip if present" prompts so throttled
  runs resume via `resumeFromRunId`.
- Every run writes per-item scores + manifests for a repeatable number audit.
