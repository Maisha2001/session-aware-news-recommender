# Frozen final chronological evaluation

The official MIND-small validation archive is reserved as this project's final
holdout. It is distinct from the chronological development partition inside
the training archive. All model weights, popularity counts, feature rules,
policies, exclusions and audit tolerances were frozen before opening it.
No model was refitted, selected or tuned using final outcomes.

The [aggregate report](../results/heldout-v1/report.json),
[independent audit](../results/heldout-v1/audit.json) and
[protocol](../results/heldout-v1/protocol.json) preserve the measured results.
Earlier baseline, retrieval, session and service reports remain unchanged.
Their `final_holdout_opened: false` fields describe those earlier stages.

## Causal replay

Final impressions must occur strictly after the entire training archive.
The evaluator retains every logged candidate and positive, including unavailable
or unretrieved positives in catalog denominators. Missing metadata, duplicate
impressions and invalid chronology cause an error instead of silent exclusions.
Archive-local impression IDs receive a `final:` namespace.

Training metadata wins when an article ID appears in both archives. New articles
use the existing fixed title/category hashing and frozen content encoder; there
is no vocabulary adaptation. Existing vectors and fitting counts remain fixed.
An article becomes retrievable only after its first observed appearance in an
earlier history or candidate slate. Current and equal-time observations cannot
enter the retrieval pool. Earlier unlabeled exposures may add later candidates.
Metadata snapshots have no publication or edit timestamps, so this availability
rule is a conservative proxy rather than proof of original publication time.

Each query uses its provided earlier history, capped at 20 articles. Session
context contains at most ten earlier same-user groups of clicked articles,
strictly more than five minutes old. A gap greater than 30 minutes resets the
session. All equal-time impressions see the same past; their clicks are combined
as an unordered centroid afterward. Click times and true session boundaries are
not supplied by MIND, so these timing rules are explicit simulation assumptions.

**Earlier final labels can affect later session features.** Current, equal-time
and future labels cannot affect a query. This is a frozen, causal replay with
delayed feedback, not an entirely label-free final simulation. Prepared contexts
are independently checked from raw TSVs. The predictor does not access current
labels; every prediction and its hash is frozen before metric aggregation.

## Comparisons and limitations

Logged ranking compares frozen popularity, learned content, long-history-only
ranking, session ranking, and the session freshness/variety presets. All logged
candidates are retained, with no evaluation negative resampling. Scores are
uncalibrated; AUC applies to the four scoring arms, while presets are evaluated
as rankings. `mean_clicked_rr` averages reciprocal ranks over all clicked items;
`first_hit_rr` is reported separately.

Catalog retrieval evaluates popularity100 and content100. Session policies rank
their deduplicated union, without injecting clicked items. Recall counts only
known logged positives; unexposed article relevance is unknown. The different
candidate pools prevent attributing every difference to the ranker alone.

Reports include cold/warm users, empty histories, cold-positive and stricter
unseen-content-positive articles, with/without session context, daily results,
fitting-exposure strata, article-macro recall and slate diversity/age. “Cold”
means absent from fitting, not permanently unseen throughout the replay.
Undefined zero-positive ratios and single-class AUC are omitted with explicit
metric counts; those impressions are otherwise retained.

There is one training seed, no significance claim, no randomized exposure or
propensity correction, and no evidence of live engagement lift. The sampled
observed catalog is not the full publisher catalog. Catalog coverage remains a
limitation even when logged-candidate ranking improves.

## Reproduce with licensed data

Use the pinned [retrieval](retrieval.md) and [session](session-ranking.md)
environments and artifacts. Keep final data sealed until your protocol and
model hashes are fixed. For an already completed run, inspect its saved evidence;
do not overwrite or rerun it to obtain a preferable result.

Create a private `models.json` containing paths to the frozen content,
long-only and session weight files:

```json
{
  "content": "/private/retrieval-model/epoch-3.weights.h5",
  "long_only": "/private/session-model/long_only-epoch-3.weights.h5",
  "session": "/private/session-model/session-epoch-3.weights.h5"
}
```

Before preparation, save your frozen choices, model/source hashes and actual
freeze timestamp as a private `protocol.json`, following the published protocol.
Keep it beside (not inside) the new run directory until preparation creates that
directory. The evaluator does not silently create or enforce an operator's
protocol; the published run separately verified all frozen inputs before every stage.
After preparation, copy that already-frozen record into the run directory for tracking.
The included `models.json` binds the prepared data to the frozen content weights;
verify all three model hashes yourself before prediction.

In a new output directory, run each stage once:

```sh
python -m newsrec.heldout prepare --run /private/final-run --models /private/models.json --database /private/news.sqlite --archive /private/MINDsmall_dev.zip --original-prepared /private/session-data
python -m newsrec.heldout predict --run /private/final-run --models /private/models.json
python -m newsrec.heldout metrics --run /private/final-run
python scripts/audit_heldout.py --training-behaviors /private/mind/train/behaviors.tsv --run /private/final-run --output /private/final-run/audit.json
```

Use four CPU threads (`TF_NUM_INTRAOP_THREADS=4`, `OMP_NUM_THREADS=4`), one
inter-op thread and `TF_ENABLE_ONEDNN_OPTS=0`, as in the recorded environment.
This workflow copies only the two required TSV members into its private output;
root members and the official `MINDsmall_dev/` wrapper are supported. The initial
root-only extraction guard failed before reading outcomes and was corrected
with an authored wrapper/ambiguity regression test. Original failure evidence,
source hashes and the unchanged protocol are retained.

Existing stage directories and reports are refused. Reproduction can differ
slightly across numerical libraries/hardware; the published audit declares
explicit tolerances. The checked CPU reloads are not a cross-platform or CUDA
bitwise retraining guarantee. Raw data, per-impression predictions and model
weights remain private under the data access policy.

## Local experiment tracking

Use a separate tracking environment:

```sh
python -m pip install -e . -r requirements/tracking.txt
```

The pinned SQLAlchemy version avoids an observed MLflow3.5 import incompatibility
with SQLAlchemy2.1. Tracking uses an explicit local SQLite store and local file
artifacts. Telemetry is disabled in the tracking process. No training, scoring,
cloud service or remote model API is invoked.

```sh
python -m newsrec.tracking --store /private/news-tracking --report /private/final-run/report.json --audit /private/final-run/audit.json --protocol /private/final-run/protocol.json --models /private/models.json --receipt /private/tracking-receipt.json
```

An optional `--artifacts` JSON maps artifact names to additional local files,
such as a wheel, environment record and earlier aggregate reports. The command
verifies model hashes against the prediction manifest, records metrics and
parameters, then reads them back and checks every artifact hash. A receipt
prevents accidentally creating duplicate runs. Inspect a failed receipt before
repairing a tracking run; it does not justify rescoring the benchmark.

For the local [MLflow interface](https://mlflow.org/docs/latest/ml/tracking/quickstart/),
set `MLFLOW_DISABLE_TELEMETRY=true` and `DO_NOT_TRACK=true` in that terminal and run:

```sh
mlflow server --backend-store-uri sqlite:////private/news-tracking/tracking.sqlite --host 127.0.0.1 --port 5025
```

On Windows, use a URI such as `sqlite:///C:/private/news-tracking/tracking.sqlite`.
Keep the interface on loopback; stop it with Ctrl+C when finished. See the
[MLflow API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html)
and [telemetry controls](https://mlflow.org/docs/latest/community/usage-tracking/).
