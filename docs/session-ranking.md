# Session ranking and slate controls

The ranker combines the frozen content vectors, a reader's supplied longer-term
history, earlier same-session feedback and an article-age proxy. It includes a
separately trained long-term-only comparison and fixed freshness/diversity
controls. This is an offline research workflow. The [local service](service.md) reuses
these frozen models in the [browsing demo](demo.md).

## Run

Use the same separate Python 3.13 environment and process thread settings as
[learned retrieval](retrieval.md). Keep model output paths short on Windows.
First complete the documented training-archive pipeline and retrieval run. The
commands below reuse its frozen index and predictions; they do not retrain the
content encoder or open the official validation archive.

```sh
python -m newsrec.session_data --database /private/news.sqlite --index /private/retrieval/learned_content --output /private/session-data
python scripts/audit_session.py context --behaviors /private/mind/train/behaviors.tsv --data /private/session-data --index /private/retrieval/learned_content --output /private/context-audit.json
python -m newsrec.train_session --data /private/session-data --output /private/session-model
python -m newsrec.evaluate_session --data /private/session-data --model /private/session-model --retrieval-report /private/retrieval/report.json --output /private/session-results
python scripts/audit_session.py metrics --behaviors /private/mind/train/behaviors.tsv --data /private/session-data --report /private/session-results/report.json --retrieval-report /private/retrieval/report.json --output /private/session-metric-audit.json
python scripts/verify_session.py
```

Prepared contexts, IDs, vectors, weights and predictions remain private. New
output directories are required. Each epoch saves weights and optimizer state;
the command does not automatically resume an interrupted checkpoint. Model
access is reproducible through licensed local data and these commands; no model
or raw-data download is redistributed. See [data access and terms](data.md).

## What counts as a session

The [MIND schema](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md)
provides impression timestamps, displayed articles, click outcomes and historical
clicked IDs. It does not provide the time of each resulting click. **Feedback
availability here is simulated, not measured.**

A gap greater than 30 minutes between a user's impressions resets the session.
Click outcomes from an earlier impression become available only when more than
five minutes have elapsed. All impressions at the same timestamp see the same
past. Their positive articles form one unordered mean-vector group, so the model
does not infer click order within an impression or between equal-time impressions.
The last ten available groups feed a 32-unit GRU. Each group also carries its
normalized log elapsed time. Current or future outcomes cannot enter its context.

The supplied last 20 historical articles form a separate mean-vector profile.
That history has no per-click timestamp here. During development evaluation,
weights stay frozen while earlier development click outcomes may enter later
session context under the five-minute rule. The long-term-only trained arm
disables this replayed-feedback feature. It is the comparison for the assumed
feedback availability; neither arm establishes performance at other real delays.

Only 5,000 of 95,071 fitting impressions and 3,511 of 61,894 development
impressions have available same-session context. Every remaining impression is
still evaluated. Results for both groups are reported separately.

## Fitting and comparisons

The content encoder and its 51,282 article vectors are unchanged from the
retrieval experiment. Both rankers use a 64-unit and then a 16-unit dense layer
over candidate content, long-term content, session representation, interaction
features and normalized first-observation age. No user-ID, item-ID or displayed
position embedding is fitted.

Each of the 141,558 fitting positives is paired with eight logged nonclicked
articles from its own impression. Sampling uses replacement only when fewer
than eight negatives are available. Other clicked articles are not negatives.
No fitting impression lacks a negative in the measured dataset. The objective
is sampled softmax; its output is an **uncalibrated ranking score**, not a click
probability.

The two architectures use the same fixed seed 20261117, three epochs, batch size
128, Adam learning rate 0.001 and gradient clip norm 1.0. One arm zeros the session
representation; the other learns it. Both use their final epoch without
development-based tuning or stopping. The comparison includes the consequences
of fitting separate models, not merely removing a feature at inference time.

Logged-candidate evaluation retains every displayed candidate. End-to-end
evaluation uses the deduplicated union of the previously frozen learned top 100
and popularity top 100, restricted to articles first observed strictly before
the query. No clicked article is injected to make the candidate pool easier.
All known clicked articles stay in recall and nDCG denominators, even when they
are unavailable or missing from the retrieved pool. Pool recall is the upper
bound imposed on the downstream ranker for these known positives.

## Freshness and diversity

Controls rerank the top ten, then retain the model order for remaining candidates.
Relevance is transformed to a rank scale, from 1 for the highest score to 0 for
the lowest, with deterministic SHA256 ties. The presets are fixed comparisons:

| Preset | Freshness weight | Redundancy penalty |
| --- | ---: | ---: |
| relevance | 0 | 0 |
| fresh | 0.15 | 0 |
| balanced | 0.10 | 0.15 |

Freshness is `exp(-first_seen_age_hours / 24)`: a 24-hour decay scale, with a
half-life of approximately 16.64 hours. The original protocol's verbal
"half-life" description was corrected before scoring; the formula and behavior
never changed. The [clarification](../results/session-v1/protocol-documentation-clarification.json)
preserves the exact documentation-only change and original fingerprint.

Redundancy combines equal weights for maximum positive cosine similarity to
already selected articles and the share of selected articles in the same
category. It is a soft preference, not a hard category quota. First-observation
age is a proxy: actual publication and edit timestamps are unavailable. Diversity
uses the frozen learned content geometry and dataset categories. Neither measure
guarantees editorial quality or reader satisfaction.

## Exposure bias and verification

The [MIND paper](https://msnews.github.io/assets/doc/ACL2020_MIND.pdf) describes
logged recommendations, not randomized exposure. Unexposed articles have unknown
relevance and are never assigned negative training labels. Catalog hit metrics
use only known logged clicks; full-catalog AUC is not reported. Without logging
propensities or randomized data, these results cannot support unbiased inverse-
propensity estimates, causal claims or live engagement lift.

Exposure-frequency bins use fitting exposures only: unseen, 1-99, 100-999 and
1,000 or more. Event recall within each bin and macro recall across distinct
clicked articles show sensitivity to frequently exposed items. These descriptive
comparisons do not remove selection bias. Cold-user, cold-positive-article,
empty-history, available-session and daily metrics remain visible.

An independent raw-TSV audit checks all 156,965 context rows, feedback boundaries,
candidate/label identities and frozen vectors. A separate metric audit recomputes
every aggregate, checks the exact retrieved union and time eligibility, and
independently recomputes controls on 128 ID-selected queries. Model reloads are
checked on 256 impressions per arm. The authored tests cover tied timestamps,
strict feedback delay, inactivity resets, own/future-label isolation, masked
padding, event order, logged-only negatives, control tradeoffs and the complete
train/evaluate/audit workflow.

See [results](../results/session-v1/report.json),
[training](../results/session-v1/training.json),
[protocol](../results/session-v1/protocol.json),
[context audit](../results/session-v1/context-audit.json),
[metric audit](../results/session-v1/metric-audit.json) and
[runtime](../results/session-v1/runtime.json). Runtime measurements are local
offline stages and batches, not API latency. One seed and one configuration do
not establish significance or cross-platform bitwise retraining. The official
validation archive was sealed at this stage; the subsequent
[final evaluation](final-evaluation.md) preserves these models and results.
