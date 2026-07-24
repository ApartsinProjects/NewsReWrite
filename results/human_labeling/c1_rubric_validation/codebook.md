# C1 Codebook: Tactic-label validation

Study: rater sees a neutral source headline and a rewrite. For each 
of ten engagement tactics, mark 1 if the rewrite realizes the tactic 
and 0 otherwise. Multiple tactics may co-occur.

Items per rater: 150. Independent raters: 3. Every rater 
sees every item (full overlap). Presentation order is randomized per 
rater; do not compare across raters by row position, only by task_id.

## Rating rules

- Base your judgment on the text of the rewrite alone, not on the 
  source. The source is provided only for context.
- A tactic is realized only if its distinguishing linguistic cue is 
  actually present in the rewrite text. Do not infer intent.
- If the rewrite copies the source verbatim (or nearly so), mark all 
  tactics 0.
- If in doubt between two overlapping tactics (e.g. Curiosity Gap vs. 
  Ambiguous References), pick the one whose canonical cue is closer to 
  the surface wording. See the tactic definitions below.
- Use the rater_notes column to flag ambiguous cases; leave blank 
  otherwise.

## Tactic definitions

### curiosity_gap
Signals that a specific but unnamed piece of information is being withheld.

### exaggeration
Amplifies importance or magnitude with intensity modifiers, no new facts.

### emotional_trigger
Uses explicit emotional wording (fear, outrage, hope, concern...).

### sensationalism
Heightens drama or spectacle without explicit emotional wording.

### lists_or_superlatives
Uses list scaffolding or extreme ranking words (top, first, largest).

### ambiguous_references
Deploys vague pronouns or indefinite references (this, they, something).

### direct_appeals
Addresses the reader or a specific audience directly (you, voters, parents).

### unfinished_narratives
Presents the event as ongoing or unresolved (what happens next...).

### unexpected_associations
Links two normally disjoint concepts or domains in a surprising way.

### provocative_questions
Uses interrogative phrasing that challenges an assumption.

## File format

One row per item. Columns:
- task_id: opaque 12-character identifier.
- presentation_order: your rater-specific row order (already shuffled).
- source_headline, rewrite: the two headlines to compare.
- tactic__<name>: put 1 or 0 in each of the ten tactic columns.
- rater_notes: free-text, optional.

Save the file with the same name; do not add or remove rows.
