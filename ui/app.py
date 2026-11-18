"""Evaluator UI — runs on Laptop 2.  Owner: P4

Run:  PATHSNAP_API_URL=http://<laptop1-ip>:8000 streamlit run ui/app.py
Talks to the API only (never reads dataset files). Shows the trip list from GET /trips,
sends the chosen trip to POST /predict, draws GPS points + predicted route on a Folium map,
and shows the metrics/timings returned by the API.
"""
import os

import requests
import streamlit as st

API = os.environ.get("PATHSNAP_API_URL", "http://localhost:8000")

st.set_page_config(page_title="PathSnap", layout="wide")
st.title("PathSnap — Map-Matching & Route Reconstruction")
st.caption(f"API: {API}")

try:
    trips = requests.get(f"{API}/trips", timeout=30).json()["trips"]
except Exception as e:  # noqa: BLE001
    st.error(f"Cannot reach API at {API}: {e}")
    st.stop()

# TODO(P4): trip selector (show n_points), "Run" button -> POST /predict,
#           folium map via streamlit_folium.st_folium (GPS points, predicted route, optional
#           classifier-only route, legend + LayerControl), metrics table, timings, error handling.
st.write(trips)
