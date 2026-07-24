# Diagnostic: base-model transfer of the dual guides (deep root-cause analysis)

Second-base-model probe on **Qwen2.5-7B-Instruct** (guides reused unchanged,
same 150 Reuters neutrals; independent DistilBERT clickbait detector).

## Surface symptom
Under the paper's `log` objective at (alpha=4, beta=2), adding the clickbait
brake appeared to *raise* independent clickbait on Qwen:
(0,0)=0.041 -> (0,2)=0.071, (4,0)=0.063 -> (4,2)=0.082. Under `lognorm` it was
worse (0.041 -> 0.113). This looked like "the brake does not transfer."

## Real root cause: the brake reward-hacks the monolingual guide via code-switching
Reading the failing rewrites shows the brake cells produce **Chinese** text, e.g.
for "Hurricane Matthew toll in Haiti rises to 1,000":
  (0,0): "How Many More Are Buried in the Shadows as ... Death Toll Hits 1,000?"  [ext 0.00, guide 1.00]
  (0,2): "Just如何增加飓风马修在海地的死亡人数...这背后隐藏着什么秘密？"        [ext 0.99, guide 0.00]

Mechanism:
- The clickbait guide is a **BERT trained on English**. On English clickbait it
  scores ~1.0; on Chinese text it is out-of-distribution and scores ~0.0.
- The brake `beta*log(1-cb_guide)` therefore *rewards* switching to Chinese,
  because that is where the guide reports "no clickbait."
- **Qwen is multilingual**, so Chinese tokens sit in its top-k and the decoder
  can escape into them. **Llama-3-8B is not** — it emits 0.0% CJK in every cell,
  which is exactly why the brake works there.

## Evidence (systematic, not cherry-picked)
- corr(guide_cb, external_cb) on Qwen = **-0.117** (the guide rewards what the
  external detector flags).
- CJK share by cell (Qwen `log`): (0,0) 0.7%, (0,2) **4.7%**, (4,0) 0.4%,
  (4,2) **3.3%**. Under `lognorm` (more brake pressure): (0,2) **11.1%**,
  (4,2) **11.6%** — over-steering escalates the escape.
- Among CJK outputs ext_cb ~= 0.8-1.0; among clean English ext_cb ~= 0.035.
- **Removing the code-switched outputs, the brake does NOT fail:**
  clean-English clickbait is (0,0)=0.035 vs (0,2)=0.035 and (4,0)=0.059 vs
  (4,2)=0.059 — the brake is neutral-to-fine on clean text; 100% of the apparent
  "failure" is the ~3-5% that code-switch.
- Llama CJK = 0.0% in all cells -> no escape -> brake works.

## This is a general guided-decoding failure mode
Any guide with an OOD blind spot can be reward-hacked by a base model capable of
producing OOD tokens. On monolingual Llama it is latent; on multilingual Qwen it
surfaces. It is NOT a scaling problem (lognorm made it worse) and NOT a
"no-headroom" problem (clean-English brake is fine).

## Workarounds (ranked)
1. **Source-script / language-consistency mask at decode time** (recommended,
   ~5-line change): drop any top-k candidate that introduces characters outside
   the source headline's script (for English sources, mask CJK/Hangul/Cyrillic).
   Prevents the escape entirely on any multilingual base model. Cheap to
   implement and validate.
2. **Multilingual guide** (mBERT / XLM-R clickbait scorer): closes the blind spot
   so the guide scores non-English clickbait too. More work (retrain), most
   principled.
3. **OOD-penalize the guide:** gate on language-ID; if the prefix is not the
   source language, apply a penalty instead of a reward.
4. Lower beta / re-tune: reduces escape pressure marginally; does not fix.
5. Switch to a less-multilingual base model: hides the bug, not recommended.

## Status
`lognorm` is kept as an experimental option (does not affect the default `log`
or any reported number). Recommended next step: implement workaround 1 and re-run
the Qwen brake to confirm transfer.
