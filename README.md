# PII Anonymisation API

HTTP service that detects and redacts sensitive entities in text. One document
per call or a small batch, with every call recorded to a database.

## Repository structure

```
pii-api/
├── app/                        # the service
│   ├── main.py                 # app factory, startup, engine loading
│   ├── config.py               # settings (env / .env), validated at boot
│   ├── routes.py               # all endpoints, thin handlers
│   ├── service.py              # engines, detection, persistence
│   ├── schemas.py              # Pydantic request/response models
│   ├── models.py               # SQLModel table: request_logs
│   ├── database.py             # engine + session handling
│   ├── entities.py             # Entity, overlap resolution, masking
│   ├── tokens.py               # token estimation for the input limit
│   ├── logging_config.py       # JSON logs (never containing request text)
│   ├── evaluation/             # metrics + MLflow, used by the notebook
│   │   ├── metrics.py
│   │   ├── runner.py
│   │   └── tracking.py
│   └── detectors/
│       ├── presidio.py         # Presidio with the Belgian recognisers
│       ├── belgian.py          # mod-97 IBAN + rijksregisternummer
│       └── llm.py              # zero-shot and few-shot extraction
├── tests/
│   ├── conftest.py             # fakes: no model loading, no API calls
│   └── test_api.py
├── notebooks/
│   └── benchmark.ipynb         # evaluation dataset -> engines -> MLflow
├── data/
│   ├── evaluation.jsonl        # consolidated evaluation dataset
│   ├── few_shot_examples.json  # shots for the few-shot engine
│   └── requests.db             # SQLite, created at first run (gitignored)
├── archify/                    # architecture diagrams (gitignored)
├── mlflow.db, mlartifacts/     # MLflow runs and artefacts (created by benchmarks)
├── streamlit_app.py            # demo client, talks to the API over HTTP
├── dashboard.py                # evaluation dashboard, reads requests.db directly
├── Dockerfile                  # the API
├── Dockerfile.streamlit        # the demo (streamlit_app.py only)
├── docker-compose.yml          # API + demo, wired together
├── pyproject.toml
└── uv.lock
```

Two rules keep it navigable: `app/detectors/` is the only place that knows how
entities are found, and `routes.py` handlers stay thin — validate, call the
service, record, respond. Anything longer belongs in `service.py`.

## Setup

```bash
uv sync
uv run python -m spacy download en_core_web_lg
uv run uvicorn app.main:create_app --factory --reload
```

Docs at http://localhost:8000/docs (only when `ENVIRONMENT=dev`, the default;
in `prod` the docs are off and `/` redirects to `/health`). The service starts
without a Cohere key — `presidio-be` works and the LLM engines report as
unavailable.

### Configuration

Settings are read from the environment or a `.env` file at the project root
(gitignored; there is no `.env.example`). Every setting has a default, so a
`.env` is optional.

| Variable | Default | |
| --- | --- | --- |
| `COHERE_API_KEY` | unset | Enables the LLM engines |
| `DEFAULT_ENGINE` | `presidio-be` | Used when a request names no engine |
| `LLM_MODEL` | `command-a-03-2025` | Pinned; a floating alias changes behaviour between deploys |
| `SPACY_MODEL` | `en_core_web_lg` | |
| `SCORE_THRESHOLD` | `0.4` | Presidio confidence cut-off |
| `MAX_INPUT_TOKENS` | `8000` | Per document |
| `MAX_BATCH_SIZE` | `50` | Documents per batch |
| `MAX_BATCH_TOKENS` | `40000` | Total across a batch |
| `CHARS_PER_TOKEN` | `4.0` | Token estimate ratio |
| `DATABASE_URL` | `sqlite:///./data/requests.db` | |
| `STORE_INPUT_TEXT` / `STORE_OUTPUT_TEXT` | `true` / `true` | See *What gets stored* |
| `ENVIRONMENT` | `dev` | `dev` or `prod` |
| `LOG_LEVEL` | `INFO` | |

### Demo and dashboard

```bash
uv sync --group demo
uv run streamlit run streamlit_app.py    # demo client
uv run streamlit run dashboard.py        # evaluation dashboard
```

The demo (Single, Batch and Activity tabs) is a thin HTTP client with no
detection logic of its own, so what it shows is what the API actually returns.

The dashboard is different: it reads `data/requests.db` directly and joins
logged predictions to `data/evaluation.jsonl` on `doc_id`, reusing
`app.evaluation.metrics` so its numbers match the notebook. It shows headline
metrics, per-label recall and precision, and each engine's errors, with
filters for engine, dataset subset and strict/partial matching. Requests with
no matching gold record are excluded from accuracy. Run it locally; it is not
part of the Docker setup.

### Docker

```bash
docker compose up --build      # API on :8000, demo on :8501
```

`./data` is mounted, so the SQLite database survives rebuilds. Build with
`SPACY_MODEL=en_core_web_sm` to cut roughly 500 MB off the image.

## Engines

| Name | What it is |
| --- | --- |
| `presidio-be` | Presidio plus mod-97 validated Belgian IBAN and rijksregisternummer. No API calls. |
| `llm-zero-shot` | LLM extraction from instructions alone. Requires `COHERE_API_KEY`. |
| `llm-few-shot` | LLM extraction with five worked examples (`data/few_shot_examples.json`). |

Labels: `PERSON`, `ORG`, `JOB`, `EMAIL_ADDRESS`, `LOCATION`, `AMOUNT`,
`DATE_TIME`, `UNIVERSITY`, `PHONE_NUMBER`, `URL`, `IBAN`, `SSN`.

Stock Presidio is deliberately not exposed. It is still built at startup
(so it shows up in `/health`'s engine list), but the API rejects it as an
engine name. On Belgian data it is strictly
worse than `presidio-be` — it has no national-number recogniser at all, and
its IBAN pattern silently drops valid matches when the next word happens to be
four letters long — so offering both would only add a slower, less accurate
option. `app/detectors/belgian.py` documents exactly what the recognisers fix.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/anonymize` | One document |
| `POST` | `/v1/anonymize/batch` | Several documents, one call each |
| `GET` | `/v1/logs` | Recent calls, newest first (`limit` up to 500, optional `status` filter); payload columns omitted |
| `GET` | `/v1/stats` | Totals, inferences run vs served from cache, token usage, avg/p95 latency, calls per engine |
| `GET` | `/v1/engines` | What this deployment can serve, with each LLM engine's prompt and `prompt_id` |
| `GET` | `/health` | Loaded engines and database status |

Errors: `413` over the token or batch limit, `503` engine unavailable, `502`
detection failed.

```bash
curl -X POST localhost:8000/v1/anonymize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Marie Dubois works at Ordina. IBAN BE68 5390 0754 7034."}'
```

```json
{
  "request_id": "3da9bee65dcf4e58",
  "redacted_text": "<PERSON> works at <ORG>. IBAN <IBAN>.",
  "entity_count": 3,
  "counts_by_label": {"PERSON": 1, "ORG": 1, "IBAN": 1},
  "entities": null,
  "engine": "presidio-be",
  "input_tokens": 34,
  "latency_ms": 234.17,
  "cached": false,
  "served_from_log_id": null,
  "served_from_timestamp": null
}
```

Batch takes `{"items": [{"id": ..., "text": ...}], "engine": ..., "include_entities": ...}`
and returns `200` with a per-item `status`, so one bad document does not
discard the rest:

```json
{
  "request_id": "...",
  "results": [
    {"id": "doc-1", "status": "ok", "redacted_text": "<PERSON> called.", "cached": false},
    {"id": "doc-2", "status": "error", "error": "detector exploded"}
  ],
  "succeeded": 1, "failed": 1,
  "engine": "presidio-be",
  "total_input_tokens": 12,
  "latency_ms": 41.6
}
```

Batch item `id`s double as `doc_id`s, so batches are deduplicated too.

## What gets stored

One row per **document**, not per HTTP request — a batch of twenty writes
twenty rows sharing a `request_id`, so latency and errors are attributable to
the document that caused them.

| Column | |
| --- | --- |
| `request_id`, `doc_id` | Correlation; `doc_id` is the caller's batch id |
| `created_at` | UTC timestamp |
| `endpoint`, `engine` | What was called |
| `input_tokens`, `input_chars` | Size of the input |
| `input_text`, `output_text` | Payloads, both individually switchable |
| `entities_json` | Detected spans, so a replay returns the same spans |
| `prompt_id`, `text_hash`, `dedupe_key` | Deduplication key and its parts |
| `served_from_log_id`, `served_from_timestamp` | Set when answered from an earlier row |
| `status`, `entity_count`, `counts_by_label` | Outcome |
| `latency_ms` | Detection time |
| `error_type`, `error_message` | Populated on failure |

SQLite by default so the service runs with no infrastructure; point
`DATABASE_URL` at Postgres for anything shared.

**`STORE_INPUT_TEXT` deserves a decision rather than a default.** Persisting
the raw input means the database now holds exactly the data this service
exists to remove — a useful debugging aid and a new place for a breach. It is
on in the example config because it is useful in development; turn it off in
production unless you have a specific reason. `STORE_OUTPUT_TEXT` is safe to
leave on, since that is the redacted form.

Writing the log never breaks the response: if the database is unreachable the
caller still gets their redacted text and the failure is logged. An
anonymisation service that returns 500 because its audit log is down has
traded its primary job for its secondary one.

## Input limits

Limits are in **tokens**, not characters. What costs money and fills a context
window is tokens, and the ratio varies: `BE68 5390 0754 7034` tokenises far
worse than ordinary prose, so a character cap either lets expensive input
through or rejects cheap input.

The count is an estimate (`app/tokens.py`), deliberately rounding up. An exact
count needs the provider's tokeniser, which means either an extra API call per
request — paying to find out whether you want to pay — or pinning tokeniser
weights to a model version. Not worth it for a guard rail whose job is to
reject the obviously-too-large.

Over-limit returns `413`, and the attempt is recorded: a caller repeatedly
hitting the ceiling is exactly what the log is for. The batch budget is
checked before any document is processed, so an over-budget batch costs
nothing rather than billing for the half it got through.

## Design notes

**Sequential, not concurrent.** A batch is a loop. Detection blocks, and
FastAPI runs synchronous handlers in a worker thread, so the event loop stays
free without any asyncio in our code. If throughput becomes a problem, more
replicas is simpler than making one request concurrent.

**Engines load at startup.** The spaCy pipeline takes several seconds. Loading
it lazily means the first caller after every deploy pays for it.

**One spaCy pipeline, shared.** An `AnalyzerEngine` built with no arguments
loads its own copy of the model. Two analysers that way is ~1 GB of resident
memory for two identical copies, so the pipeline is built once and passed in.

**Settings come from app state, not `get_settings()`.** `get_settings()` is
`lru_cache`d and reads the environment, so settings passed into `create_app`
would be silently ignored — which had already made `STORE_INPUT_TEXT=false`
store the text anyway. App state is the single source of truth for what an
instance is configured to do.

**Detected spans are opt-in.** `include_entities` is off by default; the spans
reveal exactly what was removed and where.

## Development

```bash
uv run pytest                  # 28 tests, no model loading, no API calls
uv run ruff check app tests streamlit_app.py
```

Tests inject a fake detector and an in-memory database, so the suite runs in
under half a second.

## Known limitations

- **English only.** Presidio runs with `language="en"`; Dutch and French input
  loses most entities. Largest practical gap for a Belgian deployment.
- **`ORG`, `JOB`, `UNIVERSITY`, `AMOUNT`** are outside Presidio's default
  entity set. The LLM engines are the answer to those labels.
- **No auth or rate limiting.** Both belong in front of this service, but
  neither exists — do not expose it publicly as-is.
- **Single process.** Each worker loads its own model copy; scale with
  replicas and size memory accordingly.

## Benchmarking and MLflow

`app/evaluation/` computes metrics and logs them to MLflow. The runner drives
the **running API over HTTP** rather than importing engines directly, so the
numbers describe what the service actually serves.

```bash
uv sync --group notebook
uv run uvicorn app.main:create_app --factory    # terminal 1
uv run jupyter lab notebooks/benchmark.ipynb    # terminal 2
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### Evaluation dataset

One consolidated file, `data/evaluation.jsonl` — 50 documents, 183 spans, each
record tagged with the subset it came from:

| `source` | Docs | Spans | What it is |
| --- | --- | --- | --- |
| `provided` | 10 | 49 | The supplied set, annotations corrected |
| `belgian` | 20 | 80 | Realistic Belgian documents, all 12 labels |
| `edge` | 20 | 54 | Format variants, distractors, Dutch and French |

**Score per source, not pooled.** The subsets have very different label mixes,
so a single blended number is not interpretable — the edge set is `PERSON`-heavy,
which Presidio handles well, so it scores *higher* than the realistic Belgian
set despite being harder by construction. `run_benchmark(..., source="edge")`
restricts a run to one subset, `runner.sources()` lists them, and the
dashboard filters by subset.

Records are `{id, source, text, entities:[{start, end, label}]}` — spans, not
masked strings, because per-entity scoring needs character offsets. The file
is edited directly; there is no build step.

**Keep the few-shot examples disjoint from the evaluation data.** Drawing
shots from evaluation data leaks answers.

### Metrics, and why these ones

Span-level F1 answers "how many spans did you get right". Anonymisation asks
"is this document safe to release", and the two come apart:

- **`document_leakage_rate`** — fraction of documents with at least one
  unredacted entity. The headline safety number, and **not derivable from span
  recall**: at 0.95 recall with five entities per document, roughly 22% of
  documents still leak something. One missed national number makes the whole
  document unsafe, so the unit of risk is the document, not the span.
- **`direct_identifier_recall`** — recall over `IBAN`, `SSN`, `EMAIL_ADDRESS`,
  `PHONE_NUMBER`, `PERSON` only. Aggregate recall of 0.90 means nothing if
  every miss is a national number.
- **`micro_f2`** — F-beta with β=2, weighting recall four times precision. A
  missed IBAN is a breach; a false positive is an over-redacted word. F1
  weights them equally, which is the wrong trade here.
- **`over_redaction_per_doc`** — the counterweight. Redacting everything
  scores perfect recall and destroys the data.
- **`latency_p50_ms` / `latency_p95_ms`** — means hide the tail.

Also logged: precision, recall, document and gold-span counts,
`replayed_documents`, `latency_total_s` and `total_input_tokens`. The set is
kept short on purpose. A dashboard with forty numbers is one nobody reads, so
per-label figures go in the `per_label.json` artefact instead of becoming
forty separate MLflow metrics.

Matching is `strict` (exact boundaries) by default; `mode="partial"` accepts
any overlap with the right label, which separates boundary errors from real
misses.

**One run per engine**, over the whole evaluation dataset (or one subset),
nested under a parent run so MLflow's compare view lines them up. Each run
also logs:

- `prompt.txt` — the system prompt the engine actually ran, read back from
  the API rather than reconstructed from source, plus a `prompt_hash` param so
  prompt changes are visible at a glance (MLflow truncates params at 500
  characters, so the prompt itself cannot be one)
- `per_label.json` — precision, recall, F1 and support per entity type
- `errors.json` — every failing document with **the source text alongside the
  misses**, since a label and an offset rarely explain why something failed

### Not repeating inference

Deduplication lives in `request_logs`, not in a side cache. Supplying a
`doc_id` makes a repeat of the same document, engine and prompt answer from a
previous row instead of running inference again:

```json
{"text": "...", "doc_id": "provided-01", "engine": "llm-few-shot"}
```

A full re-run of the 50-document evaluation set goes from 50 inferences to 0.

**The key is `doc_id + engine + prompt_id`.** `prompt_id` is a fingerprint of
the prompt the engine actually ran, so editing a prompt invalidates exactly
that engine's results and nothing else — no manual clearing. Text is
deliberately not in the key, since a `doc_id` is assumed stable; the input
hash is stored anyway, and a document that changed under a reused id is
refused rather than replayed with a stale answer.

**A replay still writes a row.** It carries `served_from_log_id` and the
original `created_at`, so the table stays a complete record of traffic while a
dashboard can separate work from reuse:

```sql
WHERE served_from_log_id IS NULL      -- inferences actually run
WHERE served_from_log_id IS NOT NULL  -- served from an earlier result
```

`/v1/stats` reports both as `inferences_run` and `served_from_cache`.

Errors are never reused — a failed row is not a result. Omit `doc_id` to force
a fresh call. Benchmark latency metrics count only measured calls, so a fully
replayed run reports `latency_measured_documents: 0` rather than a
distribution of fake zeroes.

### Starting over

```bash
rm -rf mlflow.db mlartifacts mlruns
rm -f data/requests.db          # also clears deduplication
```

Two stores, so delete both: `mlflow.db` holds the runs and metrics,
`mlartifacts/` holds `prompt.txt`, `per_label.json` and `errors.json`.

`tracking.setup()` pins the artifact directory next to the database. Without
that, MLflow defaults to `./mlruns` relative to the working directory of
whichever process logged the run — so a notebook whose kernel starts in
`notebooks/` writes artifacts to `notebooks/mlruns/` while the database sits
at the project root. Two stores, one of which you would not think to delete.
If you have a stray `mlruns/` from an earlier run, that is where it came from.

Deleting runs from the MLflow UI only soft-deletes them; they stay in the
database until `mlflow gc`.

**Error artefacts contain sensitive text.** The missed and spurious spans
include the source text itself, and `run_benchmark` and `log_evaluation` both
default to `log_errors=True` (the notebook sets it explicitly). Pass
`log_errors=False` to skip `errors.json`, or `redact_error_text=True` to
`log_evaluation` to keep labels and offsets while dropping the content. Do
not run benchmarks with `log_errors=True` on real data.