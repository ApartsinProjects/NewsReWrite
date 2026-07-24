# C2 Codebook: Rewrite quality

Study: rater sees a neutral source headline and a rewritten headline. 
Rate the REWRITE on three independent 1-5 Likert scales. You are NOT 
told which generation condition or which method produced the rewrite; 
multiple methods and multiple parameter settings are mixed together 
and shuffled per rater.

Items per rater: 1500. Independent raters: 3. Every rater 
sees every item (full overlap). Presentation order is randomized per 
rater; do not compare rewrites across raters by row position, only by 
task_id.

## Scales

### engagement_1_to_5
How likely would a busy reader click this rewrite in a news feed, 
relative to the source?

  1 = far less engaging than the source
  2 = somewhat less engaging
  3 = about the same as the source
  4 = somewhat more engaging
  5 = far more engaging

### faithfulness_1_to_5
How well does the rewrite preserve the factual content of the source? 
Introducing a new noun, entity, number, or event counts as a fidelity 
loss even if it sounds plausible.

  1 = the rewrite contradicts or adds facts absent from the source
  2 = one clear novel claim or spin present
  3 = mild embellishment, no new facts
  4 = fully faithful with different wording
  5 = fully faithful and preserves emphasis

### clickbait_1_to_5
How much does the rewrite feel like clickbait, in the sense of 
manipulative or misleading engagement optimization?

  1 = pure informational rewrite, no manipulation
  2 = mild rhetorical framing but responsible
  3 = borderline
  4 = noticeable clickbait patterns
  5 = clearly manipulative clickbait

## Rating rules

- Rate the rewrite as it stands, without imagining a better version.
- The three scales are independent: a rewrite can score high on 
  engagement AND high on faithfulness AND low on clickbait, or any 
  other combination.
- If the rewrite is identical to the source, mark engagement 3 and 
  faithfulness 5.
- Use rater_notes for anything you want to flag; leave blank otherwise.

## File format

One row per item. Columns:
- task_id: opaque 12-character identifier.
- presentation_order: your rater-specific row order (already shuffled).
- source_headline, rewrite: the two headlines.
- engagement_1_to_5, faithfulness_1_to_5, clickbait_1_to_5: integers 1-5.
- rater_notes: free-text, optional.

Save the file with the same name; do not add or remove rows.
