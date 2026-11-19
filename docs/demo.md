# Fieldnotes: local browsing demo

Fieldnotes connects the frozen recommendation models to a usable news feed.
Choose an interest, inspect article summaries and ranking details, explicitly
save or dismiss stories, and compare offline model results. Everything runs
locally. Article content comes only from the original training archive;
the final holdout contributes aggregate comparison statistics only.

## Authored interface preview

For a quick interface demonstration without data or model files, run from a clone:

```sh
python -m pip install -e . -r requirements/browser.txt
python -m uvicorn examples.preview:create_preview --factory --host 127.0.0.1 --port 8025
```

Open http://127.0.0.1:8025. The preview clearly labels its invented stories,
deterministic ordering and illustrative comparison numbers. It is not a model
evaluation. Stop it before starting the real demo on the same port.

## Start the model-backed demo

Complete the [service setup](service.md) in a separate Python 3.13 environment.
Reuse an existing verified catalog and serving-model manifest; run the importer
only for a new, empty database. Start the two dedicated containers and API as
described there. Keep `NEWS_API_TOKEN` private in the server environment.

In a second terminal using the same environment, set:

```text
NEWS_DEMO_DATABASE=/private/news.sqlite
NEWS_MODEL_MANIFEST=/private/serving-model/manifest.json
NEWS_RESULTS_DIRECTORY=/path/to/session-aware-news-recommender/results
NEWS_SERVICE_URL=http://127.0.0.1:8015
NEWS_API_TOKEN=<the same private server token>
```

Then run:

```sh
python -m uvicorn newsrec.demo:create_demo --factory --host 127.0.0.1 --port 8025
```

Open **http://127.0.0.1:8025**. The demo verifies the original SQLite database
against the serving manifest before loading its article summaries. A different
database, unavailable API or missing model files fails visibly. The shared API
token stays in the Python proxy and is never sent to browser JavaScript.

## Explore

1. **Your starting lens:** choose a category. Five deterministic, previously
   observed stories form an authored interest profile. “Surprise me” starts with
   no history. These are synthetic readers, not real users' browsing histories.
2. **Your next choices:** “More like this” records an explicit positive signal;
   “Less like this” dismisses a displayed story. Opening a summary alone does
   not create feedback.
3. **Refresh feed:** advance the simulated clock six minutes. Positive feedback
   becomes eligible only after the service's fixed five-minute delay. A dismissed
   story disappears from subsequent feeds. Relevance, freshness and variety
   settings reorder the same retrieved candidate union.
4. **Read summary:** inspect the stored abstract, uncalibrated ranking score,
   content alignment and first-observation age. Similarity is descriptive, not
   a causal explanation or click probability. First observation is not publication.
5. **Model results:** switch between chronological development and the reserved
   final holdout. Logged ranking and full observed-catalog retrieval use different
   candidate pools; the table explains that distinction.

The replay starts at noon on November 14, 2019 and remains within that dataset
day. Its displayed time is not today's time or a claim about the dataset's
geographic timezone. A new profile or “Start a new session” creates a new
synthetic reader and resets the replay clock. This is a research demonstration,
not a live news product or an online engagement experiment.

## State and operation

The browser uses a same-site, HTTP-only session cookie and a per-session action
token. HTML renders dynamic article text with text nodes. Host/origin checks and
a restrictive content policy keep this loopback application self-contained;
there are no external fonts, scripts or image requests. Desktop and 390-pixel
mobile layouts support the same controls, with keyboard dialog dismissal.

The demo keeps at most 128 browser sessions in memory and runs with one worker.
A restart or eviction ends that browser session. PostgreSQL retains the service's
synthetic receipts and explicit feedback. No automatic retention/deletion policy
or public user authentication is implemented. Do not expose either service to
the internet. Replay time and reader identity are supplied by the trusted local
caller, as described in the service guide.

Stop both Python processes with Ctrl+C, then stop only this application's
Compose services:

```sh
docker compose -f deploy/compose.yml stop
```

Stopping preserves the dedicated PostgreSQL volume. Browser screenshots from
the measured local run remain private because they contain licensed news text.
The published aggregate verification describes the exercised controls and
screen sizes without redistributing that content.
