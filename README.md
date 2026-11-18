# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy GPS tracks to a directed road network. For each GPS point it builds a set of candidate road segments, a trained classifier scores each candidate, and an HMM/Viterbi decoder uses network connectivity and shortest-path routing to reconstruct a continuous, topologically valid route.

The trained model runs behind a **FastAPI backend service on Laptop 1**. An evaluator interface built with **Streamlit runs on Laptop 2** communicating over HTTP.

*UE24CS352A Machine Learning Hackathon, PES University.*

---

## Architecture Overview

```
                      +--------------------------------------------------+
                      |               LAPTOP 1: FASTAPI API              |
                      |                                                  |
                      |  POST /predict {"trip_id": "00000000"}           |
                      |  1. load_trip(with_route=False)                  |
                      |  2. RoadNetwork spatial candidate lookup         |
                      |  3. Feature extraction & classifier scoring      |
                      |  4. HMM / Viterbi continuous decoding            |
                      |  5. Server-side reference route evaluation       |
                      +--------------------------------------------------+
                                        ▲                │
                   HTTP POST /predict   │                │ JSON PredictResponse
                   (Trip selection)     │                │ (GPS + Route Polylines
                                        │                │  + Metrics + Timings)
                                        │                ▼
                      +--------------------------------------------------+
                      |              LAPTOP 2: STREAMLIT UI              |
                      |                                                  |
                      |  • Trip Selector from GET /trips                 |
                      |  • Interactive Folium Map (GPS, HMM, Classifier) |
                      |  • Top-1 Accuracy & Route Continuity metrics     |
                      |  • Pipeline Latency Breakdown cards              |
                      +--------------------------------------------------+
```

---

## Repository Layout

| Folder | Contents & Ownership |
|---|---|
| `pathsnap/` | Core library: loading (`io.py`), projection (`geo.py`), network (`network.py`), candidate generation (`candidates.py`), labels (`labels.py`), features (`features.py`), models (`models.py`), metrics (`metrics.py`), routing (`routing.py`), HMM (`hmm.py`), pipeline (`pipeline.py`), evaluation (`evaluate.py`), interactive Folium maps (`viz.py`) |
| `scripts/` | Executable entry points: `build_dataset.py`, `train_models.py`, `bayes_opt.py`, `eval_hmm.py`, `make_maps.py` |
| `notebooks/` | Exploratory data analysis (`p4_eda.ipynb`) covering §4 inventory, sampling rates, gaps, network size, connectivity |
| `configs/` | `default.yaml`: single source of truth for paths, thresholds, and hyperparameters |
| `api/` | FastAPI backend service (`api/main.py`, `api/schemas.py`) for Laptop 1 |
| `ui/` | Streamlit evaluator application (`ui/app.py`, `ui/requirements.txt`) for Laptop 2 |
| `artifacts/` | Serialized model (`model.joblib`), metadata (`model_meta.json`), HMM tuning (`hmm_config.json`) |
| `results/` | Experiment tracking CSVs, plots (`results/plots/`), interactive HTML maps (`results/maps/`) |
| `report/` | Technical report markdown sections (`report/sections/`) and final deliverable |
| `docs/` | `WORK_PLAN.md` (role breakdown) and `CONTRACTS.md` (frozen API/data interfaces) |
| `tests/` | Contract tests (`tests/test_contracts.py`) and end-to-end integration tests (`tests/test_e2e.py`) |

---

## Quickstart & Setup

### 1. Environment Installation
```bash
python3 -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Dataset Setup
Extract the dataset into the repository root (gitignored):
```
dataset/
├── train/          # 80 training trips (00000001, ..., 00000090)
└── test/           # 10 student-testing trips (00000000, 00000007, ...)
```

Verify setup and contract tests:
```bash
pytest -q tests/test_contracts.py
# Must print: 4 passed
```

---

## Reproducing Pipeline Stages

```bash
# 1. Exploratory Data Analysis & Map Generation
python -m scripts.make_maps         # Generates Folium HTML maps in results/maps/
python scripts/build_p4_eda_nb.py   # Compiles notebooks/p4_eda.ipynb with rendered figures

# 2. Candidate Generation & Feature Extraction
python -m scripts.build_dataset     # Candidate tables -> data/processed/*.parquet

# 3. Model Training & Comparison (5+ models with GroupKFold)
python -m scripts.train_models      # Outputs results/model_comparison.csv & loss curves

# 4. Bayesian Hyperparameter Optimisation
python -m scripts.bayes_opt         # Optuna TPE tuning -> artifacts/model.joblib

# 5. HMM Sequential Decoding & Evaluation
python -m scripts.eval_hmm          # HMM vs classifier comparison -> results/hmm_comparison.csv
```

---

## Two-Laptop Demo Setup

### Demo Checklist
1. **Network Connectivity:** Ensure both laptops are connected to the same local Wi-Fi or phone hotspot.
2. **Obtain Laptop 1 IP Address:**
   - **Windows:** Run `ipconfig` in PowerShell/CMD and locate the IPv4 Address (e.g., `192.168.1.45`).
   - **macOS/Linux:** Run `ipconfig getifaddr en0` or `hostname -I`.
3. **Firewall:** Ensure port `8000` on Laptop 1 is open to inbound TCP traffic.
4. **Terminal Visibility:** Keep the Laptop 1 terminal visible during evaluation so evaluators can inspect live request logs.

### Running Laptop 1 (FastAPI Backend)
```bash
# If using the default evaluation directory (dataset/test):
uvicorn api.main:app --host 0.0.0.0 --port 8000

# When provided with the official evaluator directory on demo day:
PATHSNAP_EVAL_DIR=/path/to/evaluator_trips uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Running Laptop 2 (Streamlit Evaluator UI)
```bash
# Install UI-specific dependencies (if running standalone):
pip install -r ui/requirements.txt

# Launch UI targeting Laptop 1's IP address:
PATHSNAP_API_URL=http://<LAPTOP_1_IP>:8000 streamlit run ui/app.py
```

---

## HTTP API Endpoints

| Method | Endpoint | Description | Request Body | Response Model |
|---|---|---|---|---|
| `GET` | `/health` | Service health status, active model name, available trip count | *None* | `Health` |
| `GET` | `/trips` | List all available evaluation trip IDs and observation counts | *None* | `TripList` |
| `POST` | `/predict` | Full map-matching inference and server-side evaluation | `{"trip_id": "00000000"}` | `PredictResponse` |

*Security & Data Integrity Note:*  
Reference route geometry is **never** transmitted over the network or exposed to the client UI. The API returns only raw GPS points, predicted route polylines, and aggregated evaluation metrics calculated strictly server-side.
