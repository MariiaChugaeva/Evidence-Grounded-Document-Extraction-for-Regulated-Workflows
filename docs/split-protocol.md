# Split protocol

## 1. Roles

| Split | Sheets | Role | May inform method choices |
|---|---|---|---|
| `dev` | 1–20 | development: label lists, cue lexicons, rules, prompts, thresholds | yes |
| `locked` | 21–29 | **audit** split, no longer a test set | only as error analysis, never as a selection criterion |
| `test` | to be collected | final holdout, one run per frozen method version | no |

`locked` was a held-out test set for exactly one run (`rules-v3`, 2026-09-13, see
`docs/locked-runs.csv`). Its labels and its failure modes have since been read and one of
them — `sds_28`, a stabilised substance read as a mixture — motivated `rules-v4`. From that
moment it stopped being a test set: any number it produces for a method chosen after that
reading is optimistically biased. It is kept for regression checks and for reporting where
the system fails, and its numbers are reported as audit numbers with that caveat attached.

`rules-v4` has therefore **not** been run on `locked`, and will not be run on it to support a
method choice. The `sds_28` correction is demonstrated by a unit test over the annotated
premise texts, which were already public in `data/annotations/source/`.

## 2. The new holdout

- At least 20 sheets, every supplier new with respect to sheets 1–30, at least 12 template
  families new with respect to sheets 1–30.
- Drawn from the strata in `docs/sampling-plan.md`, with S3 (stabilised substances), S5
  (open-ended composition) and S7 (difficult text layer) represented by at least 3 sheets
  each, because those are the classes the derivation rules are weakest on.
- Assigned to the holdout before annotation, by supplier and template family, not by how the
  sheets look.

## 3. Freeze order

1. The method is frozen: extractor version bumped, lexicons closed, thresholds fixed, commit
   tagged `freeze/<version>`.
2. The holdout is annotated. Annotation is blind: the extractor is not run on the holdout,
   and no prediction file for it exists at this point.
3. One evaluation run per frozen version. `scripts.evaluate` refuses the non-development
   splits without `--allow-locked`, checks that the prediction file declares the split it is
   being scored against and covers exactly the documents of that split, optionally checks a
   prediction id supplied with `--expect-id`, and appends the run to `docs/locked-runs.csv`
   with the date, the split, the extractor version, the git commit and the prediction id.
4. If a second run is needed on the same version — a crash, a wrong flag — the first run is
   still logged, and the reason is written into this file.

The four `rules-v3` rows in `docs/locked-runs.csv` are backfilled: the log did not exist in
September, so the date, the commit and the prediction ids were reconstructed from the git
history of the committed prediction files. Every row after them is written by the evaluator.

## 4. What counts as tuning on a held-out split

All of the following, and each is a reason to retire the split:

- adding a label, a form phrase or a completeness phrase because a held-out sheet used it;
- changing a threshold after seeing a held-out score;
- choosing between two extractor versions, two text layers or two reading orders by their
  held-out scores;
- re-annotating a held-out sheet because the extractor disagreed with it.

The last one is the subtle one: a genuine annotation error found on a held-out sheet must be
fixed, but the fix is recorded, and the split's numbers from before the fix are not comparable
with the ones after it.
