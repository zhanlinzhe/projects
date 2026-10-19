# Email Cascade Classifier

A small end-to-end MLOps project: a two-stage email classifier served with FastAPI,
versioned with DVC, tracked with MLflow, packaged with Docker and delivered with GitHub Actions.

The first stage decides whether an email is spam. Only emails that pass the spam filter
reach the second stage, which classifies them as `work`, `personal` or `notification`.

```
                 POST /predict {"text": "..."}
                            |
                            v
                 +---------------------+
                 |  spam_model.joblib  |  TF-IDF + classifier
                 +---------------------+
                     |            |
              is spam|            |not spam
                     v            v
            category=spam   +-----------------------+
                            | normal_model.joblib   |
                            +-----------------------+
                                      |
                      work / personal / notification
```

## API

| Method | Path      | Request                  | Response |
|--------|-----------|--------------------------|----------|
| GET    | `/health` | –                        | `{"status": "ok", "models_loaded": true}` |
| POST   | `/predict`| `{"text": "..."}`        | `{"is_spam": true, "category": "spam"}` |

`POST /predict` examples:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won a free iPhone, click here to claim it."}'
# {"is_spam":true,"category":"spam"}

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Can you review the Q3 budget spreadsheet before Friday?"}'
# {"is_spam":false,"category":"work"}
```

Input rules: `text` must be a string, non-blank after trimming, and at most 5000 characters.
Invalid input returns HTTP 422; missing models return HTTP 503.

Interactive docs: <http://localhost:8000/docs>.

## Repository layout

```
app/                 FastAPI service (entrypoint: app.main:app)
tests/               pytest suite with boundary cases
ml/data_pipeline.py  deterministic synthetic dataset generator (v1 / v2)
ml/train.py          training + MLflow tracking + model export
ml/report.py         builds ml/README_experiments.md from the run summaries
ml/configs/          TF-IDF and model hyperparameters
ml/registry/         current production model card
data/                DVC-tracked dataset (raw CSV is not committed)
data.dvc             DVC pointer for the current dataset version
artifacts/           exported joblib models used by the API and the image
.github/workflows/   CI (pull requests) and CD (staging / production)
Dockerfile           single-stage runtime image based on python:3.11-slim
```

## Local development

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt -r requirements-ml.txt -r requirements-dev.txt

python ml/data_pipeline.py --version v2 --out data/emails.csv
python ml/train.py --model lr --run-name improved-lr --save-final

uvicorn app.main:app --reload
pytest -q
ruff check .
```

The DVC remote is configured locally (see `ml/README_data.md`); the SAS token lives in
`.dvc/config.local`, which is git-ignored.

## MLOps lifecycle

| Concern | Tool | Where |
|---|---|---|
| Data versioning | DVC + Azure Blob Storage | `data.dvc`, tags `data-v1` / `data-v2` |
| Experiment tracking | MLflow (local file backend) | `mlruns/` (git-ignored), `ml/train.py` |
| Model registry | Markdown model card | `ml/registry/current_model.md` |
| Provenance | Commit ↔ data ↔ run ↔ artifact | `ml/PROVENANCE.md` |

Every MLflow run logs the git commit SHA, the DVC data hash, the hyperparameters,
accuracy / F1 metrics, a confusion matrix and the trained model artifacts.

MLflow 3 keeps the local `mlruns/` file store behind `MLFLOW_ALLOW_FILE_STORE`;
`ml/train.py` sets it automatically, and `mlflow ui` needs the same flag:

```bash
export MLFLOW_ALLOW_FILE_STORE=true      # PowerShell: $env:MLFLOW_ALLOW_FILE_STORE='true'
mlflow ui --backend-store-uri mlruns
```

## Testing

`tests/test_predict.py` contains three functional tests, each with several parametrized
edge cases:

1. spam emails → `is_spam=true`, `category=spam`;
2. normal emails → the correct `work` / `personal` / `notification` sub-category;
3. invalid payloads (empty, whitespace-only, oversized, non-string, null, missing field)
   → HTTP 422;
4. plus a `/health` check that both models are loaded.

```bash
pytest -q
```

## Branching and delivery flow

```
feature/* --PR--> dev --PR--> staging --PR--> main
                     |            |              |
                     CI           CD staging     CD production
```

| Branch | Purpose | Pipeline |
|---|---|---|
| `feature/*` | one module per team member | CI runs on the pull request into `dev` |
| `dev` | integration branch | CI on every PR; no image build on push |
| `staging` | pre-production validation | CD builds `email-cascade-classifier:staging` from staging secrets |
| `main` | production | CD builds `email-cascade-classifier:prod` from production secrets |

Every pull request into `dev`, `staging` or `main` must pass three required checks
(`Lint (ruff)`, `Tests (pytest)`, `Docker build`) before it can be merged.

### From commit to production

1. A developer pushes a `feature/*` branch and opens a pull request into `dev`.
2. CI installs dependencies, runs `ruff check .`, `pytest -q` and `docker build`.
3. After review and green checks, the feature branch is merged into `dev`.
4. When `dev` is stable, it is merged into `staging`; the CD workflow creates a staging
   `.env` from `STAGING_ENV_FILE` and builds the staging image.
5. After staging validation, `staging` is merged into `main`; CD creates a production
   `.env` from `PROD_ENV_FILE` and builds the production image.

No deployment target is requested by the course, so CD stops after a successful image build.

## Team

| Member | Role | Ownership |
|---|---|---|
| Zhan Linzhe | ML engineer | `ml/`, `data/`, `artifacts/`, DVC, MLflow |
| Chu Ci | Backend engineer | `app/`, `tests/` |
| Wu Haoyang | DevOps engineer | `Dockerfile`, `.github/workflows/`, `README.md`, `DEPLOYMENT.md` |

See [DEPLOYMENT.md](DEPLOYMENT.md) for environments, secrets and branch protection.

<!-- ci smoke -->
