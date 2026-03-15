# Cost Control LLM Router

A production-grade intelligent routing platform that sends every prompt to the cheapest LLM capable of handling it — automatically, without the developer ever picking a model.

Instead of sending every prompt to the strongest model which isn't a cost-effective decision. 

---

![Dashboard Overview](screenshots/dashboard_overview.png)

## What It Does

- Accepts a prompt via a single API endpoint
- Classifies the prompt by difficulty and assigns a confidence score
- Routes to the cheapest model that can handle it well
- Escalates to a stronger model if confidence is too low to trust the decision
- Calls the selected model and returns the response
- Calculates actual cost vs what GPT-4o would have charged, and logs the savings
- Learns from human feedback over time and adjusts routing rules automatically

---

## Enterprise Architecture Decisions

## Dashboard

![Routing Decision Panel](screenshots/routing_decision.png)

*The system classified "Explain data types in Kotlin" as `code_generation` with 95% confidence, routed to `llama-3-70b`, and saved $0.011565 vs GPT-4o.*

![Feedback and Recent Requests](screenshots/feedback_table.png)

*Click any row to auto-fill the feedback panel. Feedback history drives adaptive routing rule changes.*

### Pydantic — Input Validation at the API Boundary

- Every incoming request is parsed and validated by a Pydantic model before any business logic runs
- Catches bad input — wrong types, missing fields, oversized prompts — immediately at the edge, not deep inside the pipeline where errors are harder to trace
- The same Pydantic models serve as typed contracts between every internal stage: `ClassificationResult`, `RoutingDecision`, `TokenUsage`, `CostBreakdown`
- FastAPI reads these models to auto-generate Swagger docs — the documentation is always in sync with the actual code, no separate spec file needed

### LLM-Based Classifier — Not a Traditional ML Model

- The classifier is an LLM call, not a trained BERT or sklearn model
- A traditional ML classifier needs thousands of labelled training examples, a training pipeline, and retraining every time prompt patterns shift — the LLM classifier needs none of this
- Understands semantic nuance out of the box — "explain recursion with a Python example" correctly gets both `code_generation` and `multi_step_reasoning` tags with zero training data
- Every classification decision includes a `reasoning` field — a plain-English sentence explaining why the tags were assigned, making every routing decision auditable by any engineer without data science knowledge
- A fallback classifier activates if the LLM classifier fails all retries — it returns `high_context_analysis` with zero confidence, forcing escalation to the strongest model so response quality is never compromised

### The Router — Three Layers of Priority

- **Layer 1 — Adaptive overrides:** If historical feedback has triggered a rule change for this tag, that override takes priority over everything else
- **Layer 2 — Base routing table:** Static mapping from difficulty tag to model tier, defined once and changed only deliberately
- **Layer 3 — Confidence-based escalation:** If the classifier confidence is below 0.5, the router escalates one tier regardless of the base table — it is always better to slightly overspend than to return a poor answer
- The router is stateless — overrides are fetched fresh from PostgreSQL on every request, so rule changes from feedback take effect immediately with no restart required
- Works correctly in multi-instance deployments because all state lives in the database, never in memory

### The Adaptive Engine thresholds

- Minimum 10 feedback samples before any rule changes
- Underpowered rate above 30% → escalates one tier
- Overkill rate above 40% → demotes one tier

### The Difficulty Tags

- simple_summarization
- information_extraction
- code_generation
- multi_step_reasoning
- high_context_analysis

### CRUD — Repository Pattern, All SQL in One File

- Every database read and write goes through `app/db/crud.py` — no other module writes SQL directly
- Changing the database schema means changing one file, not hunting through route handlers and business logic
- Key operations: `log_request` (full audit record per call), `save_feedback` (linked to request via foreign key, unique constraint enforced at DB level), `get_adaptive_overrides` (fetched before every routing decision), `get_feedback_stats_for_tag` (drives the adaptive engine), `save_routing_adjustment` (audit trail of every rule change)
- Uses `flush()` not `commit()` inside write functions — the commit wraps the full HTTP request lifecycle, so if anything fails midway, the entire transaction rolls back automatically

### LLM Clients — Abstract Base 

- All three model tiers implement the same abstract interface — `complete(prompt, max_tokens, temperature)` — and return a normalised `LLMResponse` object
- Provider-specific response shapes (Groq, OpenAI, Anthropic all differ) are translated at the client boundary — the rest of the system never knows which provider was used
- Client instances are singletons via `lru_cache` — one instance per model per process, preserving internal HTTP connection pools across requests
- Latency is measured in the base class `complete_with_timing` method, ensuring consistent measurement regardless of which provider is used

### Cost Calculator — Decimal Precision, Not Float

- Uses Python's `Decimal` type throughout — floating point arithmetic (`0.1 + 0.2 = 0.30000000000000004`) compounds into materially wrong financial reports at scale
- Input and output tokens are priced separately because providers charge them differently — GPT-4o charges 3x more for output than input
- The GPT-4o baseline is always calculated using real production pricing even during development on Groq's free tier — savings figures shown in the dashboard reflect real enterprise economics

### PostgreSQL Logging 

- Every request is logged before the response is returned: prompt, tags, confidence, model chosen, routing reason, tokens, actual cost, baseline cost, savings, latency
- DB write failures never cause a 502 for the user — logging is best-effort, user experience is not
- `routing_adjustments` table records every adaptive rule change with the rates that triggered it and a plain-English reason — a full history of how the system learned
- Each request gets a UUID generated at entry that appears in every log line throughout the pipeline — grep one UUID in production logs to trace the complete lifecycle of any single request


The dashboard auto-open behaviour is available , apart from that we can use Swagger UI for testing - http://127.0.0.1:8000/docs

---

## Models

| Tier    | Model                        | Used For                     |
|---------|------------------------------|------------------------------|
| Cheap   | `llama-3.1-8b-instant`       | Summarization, extraction    |
| Medium  | `llama-3.3-70b-versatile`    | Code generation, reasoning   |
| Strong  | `mixtral-8x7b-32768`         | Complex analysis             |

All three run on Groq's free tier. Swap model IDs in `.env` to use GPT-4o, Claude, or any OpenAI-compatible provider in production — no code changes needed.

---

## Project Structure

```
cost_control_llm_router/
│
├── main.py
├── .env
├── requirements.txt
├── cost_control_llm_router_dashboard.html
│
└── app/
    ├── config.py
    ├── schemas/
    │   └── pydantic_models.py
    ├── db/
    │   ├── models.py
    │   ├── session.py
    │   └── crud.py
    ├── classifier/
    │   ├── prompt_classifier.py
    │   └── fallback_classifier.py
    ├── router/
    │   └── model_router.py
    ├── llm_clients/
    │   ├── base.py
    │   ├── groq_base.py
    │   ├── phi_client.py
    │   ├── llama_client.py
    │   ├── mixtral_client.py
    │   └── client_factory.py
    ├── cost/
    │   └── cost_calculator.py
    ├── feedback/
    │   └── adaptive_engine.py
    └── api/
        └── routes/
            ├── completion.py
            ├── feedback.py
            └── analytics.py
```

---

## Prerequisites

- Python 3.11+
- PostgreSQL
- Git

---

## Step 1 — Get a Free Groq API Key

1. Go to [console.groq.com](https://console.groq.com)
2. Sign up with Google — no credit card needed
3. API Keys → Create API Key
4. Copy the key — starts with `gsk_`, shown only once

---

## Step 2 — Set Up PostgreSQL

### Mac (Terminal)

```bash
brew install postgresql
brew services start postgresql
psql postgres
```

```sql
CREATE USER llmrouter WITH PASSWORD 'password';
CREATE DATABASE llm_router_db OWNER llmrouter;
\q
```

### Ubuntu / Linux (Terminal)

```bash
sudo apt install postgresql -y
sudo service postgresql start
sudo -u postgres psql
```

```sql
CREATE USER llmrouter WITH PASSWORD 'password';
CREATE DATABASE llm_router_db OWNER llmrouter;
\q
```

### PyCharm

1. View → Tool Windows → Database
2. `+` → Data Source → PostgreSQL → Host `localhost`, Port `5432`, User `postgres`
3. Test Connection → OK
4. Right-click connection → New → Query Console

```sql
CREATE USER llmrouter WITH PASSWORD 'password';
CREATE DATABASE llm_router_db OWNER llmrouter;
GRANT ALL PRIVILEGES ON DATABASE llm_router_db TO llmrouter;
```

Connect to `localhost:5432` and run the same three SQL statements above.

---

## Step 3 — Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/cost_control_llm_router.git
cd cost_control_llm_router
python3 -m venv .venv

# Mac / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Step 4 — Create .env

```env
DATABASE_URL=postgresql+asyncpg://llmrouter:password@localhost:5432/llm_router_db

GROQ_API_KEY=gsk_your_key_here

PHI3_MODEL_ID=llama-3.1-8b-instant
LLAMA3_MODEL_ID=llama-3.3-70b-versatile
GPT4O_MODEL_ID=mixtral-8x7b-32768

PHI3_INPUT_COST_PER_1M=0.50
PHI3_OUTPUT_COST_PER_1M=0.50
LLAMA3_INPUT_COST_PER_1M=0.90
LLAMA3_OUTPUT_COST_PER_1M=0.90
GPT4O_INPUT_COST_PER_1M=5.00
GPT4O_OUTPUT_COST_PER_1M=15.00

HIGH_CONFIDENCE_THRESHOLD=0.80
LOW_CONFIDENCE_THRESHOLD=0.50
```

---

## Step 5 — Run

```bash
uvicorn main:app --reload --port 8000
```

Dashboard opens automatically at `http://127.0.0.1:8000/dashboard`

---

## API Quick Reference

```bash
# Complete a prompt
curl -X POST http://127.0.0.1:8000/api/v1/complete \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your prompt here", "caller_id": "my-app"}'

# Submit feedback
curl -X POST http://127.0.0.1:8000/api/v1/feedback \
  -H "Content-Type: application/json" \
  -d '{"request_id": "UUID-HERE", "rating": 4, "underpowered": false, "overkill": false}'

# Cost summary
curl http://127.0.0.1:8000/api/v1/analytics/cost-summary?lookback_days=30

# Routing rules
curl http://127.0.0.1:8000/api/v1/analytics/routing-table
```
![Swagger UI](screenshots/swagger_complete.png)

*Full interactive API docs available at `http://127.0.0.1:8000/docs`*

![Swagger All Endpoints](screenshots/swagger_endpoints.png)


---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Router only — no API calls, instant
pytest tests/test_router.py -v

# Cost calculator — pure math, no API calls
pytest tests/test_cost_calculator.py -v

# Classifier — makes real Groq API calls
pytest tests/test_classifier.py -v -s
```

---

## Tech Stack

| Layer      | Technology          |
|------------|---------------------|
| API        | FastAPI             |
| Database   | PostgreSQL          |
| ORM        | SQLAlchemy async    |
| Validation | Pydantic v2         |
| LLM        | Groq (free tier)    |
| Retries    | Tenacity            |
| Logging    | Structlog           |
| Server     | Uvicorn             |

---

## Environment Variables

| Variable                  | Description                              | Default / Example              |
|---------------------------|------------------------------------------|--------------------------------|
| `DATABASE_URL`            | PostgreSQL connection string             | `postgresql+asyncpg://...`     |
| `GROQ_API_KEY`            | Groq API key                             | `gsk_...`                      |
| `PHI3_MODEL_ID`           | Tier 1 model                             | `llama-3.1-8b-instant`         |
| `LLAMA3_MODEL_ID`         | Tier 2 model                             | `llama-3.3-70b-versatile`      |
| `GPT4O_MODEL_ID`          | Tier 3 model                             | `mixtral-8x7b-32768`           |
| `HIGH_CONFIDENCE_THRESHOLD` | Trust base routing above this          | `0.80`                         |
| `LOW_CONFIDENCE_THRESHOLD`  | Escalate below this                    | `0.50`                         |
| `*_INPUT_COST_PER_1M`     | Input token price per million            | `0.50`                         |
| `*_OUTPUT_COST_PER_1M`    | Output token price per million           | `15.00`                        |
