# Recommendation service

The local FastAPI service uses the frozen content encoder and session ranker,
an exact FAISS catalog, PostgreSQL receipts and feedback, and an optional Redis
cache. It supports article insertion without retraining. Scores remain
uncalibrated ranking scores; the earlier offline results do not establish live
engagement improvements. The [browsing application](demo.md) and [final evaluation](final-evaluation.md)
reuse the same frozen models.

## Setup

Use a separate Python 3.13 environment. Install the project and pinned service
dependencies, then set the process thread limits used in the measured run:

```sh
python -m pip install -e . -r requirements/service.txt -r requirements/test.txt
# Set these variables in the shell that starts the application.
export TF_USE_LEGACY_KERAS=1
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS=4
export TF_ENABLE_ONEDNN_OPTS=0
```

On PowerShell, use `$env:NAME='value'` instead of `export NAME=value`.
The [data guide](data.md), [retrieval workflow](retrieval.md) and
[ranking workflow](session-ranking.md) describe how to obtain licensed local
data and produce the required private artifacts. Existing verified artifacts
can be reused directly. Raw articles, individual histories and model weights
are not redistributed by this repository.

Set strong, private `NEWS_DB_PASSWORD` and `NEWS_REDIS_PASSWORD` environment
variables before starting the two isolated containers:

```sh
docker compose -f deploy/compose.yml up -d
```

PostgreSQL listens only on `127.0.0.1:55435`; Redis listens only on
`127.0.0.1:56385`. The Compose file pins image digests, resource ceilings and
restart behavior. Its dedicated PostgreSQL volume preserves receipts, feedback
and article vectors. Redis is a disposable, memory-limited cache.

Set these private application variables, substituting URL-encoded passwords:

```text
NEWS_DATABASE_URL=postgresql://portfolio_news:<password>@127.0.0.1:55435/portfolio_news
NEWS_REDIS_URL=redis://:<password>@127.0.0.1:56385/0
NEWS_API_TOKEN=<private random token of at least 24 characters>
NEWS_MODEL_MANIFEST=/private/serving-model/manifest.json
```

Keep credentials out of version control and command logs. Import the existing
training-archive catalog and frozen model artifacts once into an empty dedicated
database. The importer verifies prepared-data and model hashes and refuses an
existing catalog or output directory:

```sh
python -m newsrec.service_setup --database /private/news.sqlite --prepared /private/session-data --content-model /private/retrieval-model --session-model /private/session-model --output /private/serving-model
python -m uvicorn newsrec.api:create_app --factory --host 127.0.0.1 --port 8015
```

Every endpoint requires `Authorization: Bearer <NEWS_API_TOKEN>`. The interactive
OpenAPI reference is available at `/docs`. Use `/health` to check PostgreSQL,
cache availability, model identity and the local index generation. The catalog
is loaded lazily on the first uncached recommendation in each worker.

## Requests and feedback

`POST /recommendations` accepts a fresh UUID, a caller-assigned reader ID,
an integer replay timestamp, up to 20 historical article IDs, a policy
(`relevance`, `fresh` or `balanced`) and a limit from 1 to 20:

```json
{
  "request_id": "21a7b0e5-454f-4f5e-a791-297cc24c954d",
  "user_id": "local_reader_1",
  "at": 1573776000,
  "history": [],
  "policy": "balanced",
  "limit": 10
}
```

The response includes article titles, categories, ranking scores, content
similarities, first-observation age, candidate count, model identity and catalog
and feedback revisions. Similarities describe vector alignment, not a causal
explanation. Candidates are the union of up to 100 learned and 100 frozen
popularity results available strictly before the replay timestamp. Dismissed
articles are removed from that union; it is not backfilled. First observation
is a dataset availability proxy, not the original publication time.

`POST /feedback` accepts `feedback_id`, `request_id`, `user_id`, `article_id`,
`kind` (`click` or `dismiss`) and `observed_at`. The article must have appeared
in that user's persisted response. Each served request/article pair accepts
one immutable feedback event. Duplicate IDs with the same payload are safe;
changed payloads return 409. A reader's clock cannot move backwards.

Clicks enter the last ten unordered impression-time groups only when the
original impression is strictly more than 300 seconds old and the feedback
observation is strictly before the new query. A gap greater than 1,800 seconds
between recommendation requests resets the session. Dismissals apply to later
queries and persist across sessions. Longer-term history is supplied by the
caller; this service does not silently extend it from feedback.

These are **caller-declared replay times**, not verified live click times.
The shared token authorizes a trusted local caller to use any supplied reader
ID. Public identity, per-user authorization, rate limiting, consent and retention
management are outside this research service. Keep it on loopback.

## Catalog and consistency

`POST /articles` takes an `articles` array of up to 256 objects with
`article_id`, `title`, `category`, optional `subcategory` and `first_seen`.
The fixed encoder creates a vector from content. New items have zero fitting
clicks; insertion does not modify the frozen popularity counts or any weights.
Catalog updates are append-only. Exact repeated content is a no-op; changes to
an existing ID return 409. A conflicting batch rolls back all its inserts.

PostgreSQL is authoritative. Each recommendation transaction locks its reader,
holds a catalog-generation read lock while scoring, and stores a complete
immutable receipt. A writer commits article vectors and a new generation
together. Each process builds a complete in-memory snapshot for that generation
before scoring; a process lock serializes its model calls. This favors bounded
consistency over high throughput. Rebuilding 51,000-plus vectors is not an
incremental distributed ANN update system.

Redis keys include the model, catalog and reader-feedback revisions, session
start, timestamp, reader, history, policy and limit. Cached payloads have a
60-second TTL and a checksum for accidental corruption. Every new request still
visits PostgreSQL. Redis failure falls back to model scoring; PostgreSQL failure
returns 503. A checksum is not protection against a malicious cache writer.

Retrying an existing request UUID returns its original receipt, including its
original revisions and cache flag, even after catalog changes. Use a new UUID to
request current recommendations. This provides idempotent effects within the
single SQL database, not distributed exactly-once delivery.

## Validation

The [service protocol](../results/service-v1/protocol.json) was frozen before
HTTP verification. The [report](../results/service-v1/report.json) records
label-free parity with preserved development predictions, actual HTTP latency,
cache behavior, two-worker consistency, restart and dependency outage checks.
Only aggregate evidence is public. Authored new-article and feedback scenarios
test behavior; they are not new measured recommendation-quality results.

For local integration tests, create a separate `portfolio_news_test` database,
set `NEWS_TEST_DATABASE_URL` to it, and set `NEWS_REDIS_URL`. Tests truncate only
the dedicated test schema and delete only their own generated cache keys.
Without those variables the relevant service tests skip. CI supplies ephemeral
PostgreSQL and Redis instances; it has no private benchmark data or model weights.

```sh
python -m pytest -q
python scripts/verify_service.py
docker compose -f deploy/compose.yml stop
```

Stopping preserves the dedicated volume. Do not use a volume-removal or global
Docker cleanup command to stop this application.
