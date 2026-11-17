# Evaluation protocol

The current pipeline is an offline baseline, not a completed recommender.
Measured results are in `results/baseline-v1`. Authored-fixture tests are correctness
checks; the separate independent audit uses all real development impressions.

Use an explicit chronological cutoff within MIND-small's training archive.
Impressions strictly before it are fitting data; impressions at or after it are
development data. The cutoff is November 13, 2019 at midnight in the dataset's
unspecified timezone. It was proposed before access and frozen at 2026-11-17
01:00:21 UTC after training timestamp inspection, before scoring. Fitting has
95,071 impressions; development has 61,894. No cutoff was selected from scores.

Popularity counts only clicked candidates in fitting impressions. It never
adds development clicks or provided histories to fitting counts. The untrained
comparison orders IDs by a deterministic SHA256 key. Ranking includes all logged
candidates; negatives are not resampled. Equal popularity uses the same hash
order, independent of logged position and outcomes.

Catalog retrieval chooses the top 20 and 100 by frozen popularity among articles
observed strictly before each impression time. Earlier label-free development
exposures can make articles available; same-time and future exposures cannot.
Availability uses earlier candidates and histories, not whether an article was
clicked. A newly observed article has zero fitted popularity. This is a conservative
observed-catalog proxy, not a publication-time catalog or the full live news catalog.

Report macro recall, nDCG and mean clicked reciprocal rank for impressions with
positives. Mean clicked reciprocal rank averages `1/rank` over every clicked item;
first-hit reciprocal rank is separate. AUC uses half credit for equal scores and
requires both classes. Report the actual denominator with every mean and count
excluded zero-positive or one-class impressions. Catalog recall retains unavailable
positives in its denominator and reports their availability separately.

Cold users have no fitting impressions. A cold-positive-article impression has at
least one positive absent from all fitting candidates. A zero-click article exposed
during fitting is not cold under that definition. Report daily slices as well as
aggregate scores. Missing metadata never removes outcomes from baseline metrics.

Logged impressions reflect an existing exposure policy and active-user sampling.
Unclicked articles are not universal dislikes; offline metrics do not demonstrate
engagement lift, unbiased preferences, or a causal policy improvement. Histories
lack individual timestamps and article text lacks version history. The evaluator
does not consume text, demographics, external services, or a final archive.

The report hashes the local database and deterministic ranked prediction stream.
Only aggregate reports are intended for publication. Input provenance must be
checked against independently recorded download hashes before using a report as
real benchmark evidence. Runtime measures this evaluator, not an online service.

## Evidence and limitations of this run

All 156,965 source impressions and 5,843,444 candidate entries were imported.
There are no missing article records, zero-positive development impressions or
single-class development impressions. Every aggregate AUC and ranking mean uses
all 61,894 development impressions. There are 15,172 fit-unseen user impressions,
46,722 fit-seen user impressions, and 37,961 cold-positive-article impressions.
The cold-positive segment overlaps the two user segments. Empty history occurs
in 1,503 impressions. All 313 unavailable positive events remain in denominators.
Macro positive availability is 99.71%; it is not an event-weighted proportion.

Frozen popularity nDCG@10 falls from 0.2972 on November 13 to 0.2633 on November 14;
its cold-positive-article AUC is 0.4513. These results expose the limitations of
stale popularity when new articles arrive. They do not establish a causal reason
or prove that a future model will improve performance. Hash order is one fixed,
untuned comparison, not an average over random trials or a significance test.

The independent auditor parses the original TSVs without importing the pipeline,
recomputes rank-sum AUC and positive-rank metrics, and maintains a chronological
active top-100 catalog. It checks every count, denominator, prediction order and
96 values across 16 metric groups; maximum numerical difference was 5.59e-14
with 1e-10 absolute tolerance. Only aggregate evidence and hashes are public.
CI checks evidence hashes and source correspondence; licensed data must be
obtained separately to rerun the full audit.

Ingestion took 66.63 seconds. Its first memory measurement captured only the
Windows venv launcher and is invalid for the worker; no ingestion memory claim
is made. Evaluation's corrected process-tree monitor sampled every 100 ms and
retained Windows peak working sets: 57,425,920 bytes for the worker and 61,644,800
bytes maximum combined sampled residency. The evaluator itself took 25.12 seconds
(25.32 seconds including CLI startup). These exclude the independent audit and
other applications and are not serving-latency measurements. The auditor took
11.82 seconds. All work used CPU; no GPU model has been trained.
