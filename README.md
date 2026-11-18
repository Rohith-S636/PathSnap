# PathSnap: Distributed Map-Matching & Route Reconstruction

PathSnap matches noisy GPS tracks to a road network. For each GPS point it builds a set of candidate road segments, a trained classifier scores each candidate, and an HMM/Viterbi decoder uses network connectivity to pick a continuous route. The trained model runs behind a FastAPI service (Laptop 1). An evaluator UI (Laptop 2) asks it for predictions.

UE24CS352A Machine Learning hackathon, PES University.

## Repository layout
| Folder | Contents |
|---|---|
| `pathsnap/` | Core library: loading, projection, road network, candidates, labels, features, models, metrics, routing, HMM, pipeline, evaluation, Folium maps |
| `scripts/` | Entry points: build dataset, train/compare models, Bayesian optimisation, HMM evaluation, map generation |
| `notebooks/` | Exploratory data analysis |
| `configs/` | `default.yaml`, the single source of paths and constants |
| `api/` | FastAPI backend (Laptop 1) |
| `ui/` | Streamlit evaluator interface (Laptop 2) |
| `artifacts/` | Final model, model metadata, HMM configuration |
| `results/` | Experiment CSVs, plots, Optuna trials, Folium HTML maps |
| `report/` | Technical report sources and PDF |
| `docs/` | Interface contracts and team work plan |
| `tests/` | Contract and end-to-end tests |

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# put the provided data at ./dataset/{train,test}/<trip_id>/<trip_id>.{track,nodes,arcs,route}
pytest -q
```

## Reproduce the results
```bash
python -m scripts.build_dataset     # candidate tables -> data/processed/
python -m scripts.train_models      # 5+ models, GroupKFold CV + student-test comparison -> results/
python -m scripts.bayes_opt         # Optuna TPE on the selected model -> artifacts/model.joblib
python -m scripts.eval_hmm          # classifier-only vs classifier+HMM -> results/hmm_comparison.csv
python -m scripts.make_maps         # Folium maps -> results/maps/
```

## Run the system
**Laptop 1 (API):**
```bash
PATHSNAP_EVAL_DIR=/path/to/evaluation_trips uvicorn api.main:app --host 0.0.0.0 --port 8000
```
**Laptop 2 (UI):**
```bash
pip install -r ui/requirements.txt
PATHSNAP_API_URL=http://<laptop-1-ip>:8000 streamlit run ui/app.py
```

## Data handling
- Models and decoder parameters are fitted on the 80 training trips and compared on the 10 student-testing trips. Evaluator trips are never used during development.
- The prediction pipeline never reads `.route`. Reference routes are used only to derive training labels and to compute metrics on the server.
