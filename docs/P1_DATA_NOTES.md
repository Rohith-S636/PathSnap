# P1 data notes: what's ready and what we found

## What you can use now
```bash
python -m scripts.build_dataset      # ~10 s for all 90 trips -> data/processed/ (gitignored, so every member runs this)
pytest -q                            # contract + P1 tests
```
| Output | Contents |
|---|---|
| `data/processed/candidates_{train,test}.parquet` | Candidate table (`schema.KEY_COLS + CANDIDATE_BASE_COLS + label`). Up to 8 arcs within 60 m per GPS point. |
| `data/processed/points_{train,test}.parquet` | One row per GPS point (`schema.POINT_COLS`): projected x/y, `ref_arc`, `ref_in_candidates`, `n_candidates`. Use this for **top-1 accuracy and miss rate**, because points with no candidates don't appear in the candidate table. |
| `results/data/build_stats_{train,test}.csv` | Per-trip quality stats (committed, used for report §2) |
| `results/data/candidate_sweep.csv` | Miss rate and table size for each k × radius combination (train only) |

**Per-trip API for P2/P3:**
```python
trip = load_trip(dir, tid)                 # no route
net  = RoadNetwork(trip)                   # 0.2 s even for 307k arcs
xy   = net.project_track(trip.track)       # (N,2) metres
net.query_arcs_bulk(xy, radius, k)         # point_idx, arc_id, dist, proj_frac, rank
net.successor_arcs(a); net.out_arcs(node); net.reverse_arc[a]   # directed connectivity, -1 = one-way
net.arc_len_m, net.arc_bearing, net.arc_a, net.arc_b, net.node_xy, net.out_degree, net.in_degree
net.arc_polyline_lonlat(arc_ids)           # [(lat, lon), ...] for Folium / API
net.arcs_near_track(xy, 100)               # "relevant road network" for maps
```

## Key decisions (be ready to explain these in the viva)
1. **Per-trip AEQD projection** centred on the track centroid. Trips span lon −100…137, so a single UTM zone is impossible. Compared with exact WGS84 ellipsoidal distances, the error is **under 1 cm over 62 km** (trip 00000013). Note that haversine is *not* a good reference here: its spherical-Earth assumption alone is off by about 0.3%.
2. **Spatial index:** a shapely STRtree over arc segments, queried with `dwithin`, then exact point-to-segment distance.
3. **Per-point labels from `.route`** using monotone DP alignment (`labels.py`). Each point is assigned a route position that never decreases, minimising capped distance. Points more than 50 m from their aligned arc get −1. Median point-to-label distance is about 4.6 m (normal GPS noise).
4. **Label sanity check:** while moving faster than 3 m/s, the GPS heading (smoothed over ±3 s) agrees with the labelled arc direction within 45° for a **median of 98.6% of points across all 88 kept trips** (minimum 81%, train 00000002). See the `heading_agreement` column in `build_stats_*.csv`. The single 1 s step heading is too noisy to use (step of 3 m vs noise of about 5 m). **Tip for P2:** compute heading over a window, not consecutive points.
5. **Candidate settings k=8, r=60 m**, chosen on train (`candidate_sweep.csv`). Miss rate is 0.46% on train and 0.26% on test. k=12 would give 0.19%, but with 27% more rows and 2.25× the Viterbi transitions. A radius above 60 m gives nothing, since labels stop at 50 m anyway.
6. **No-candidate convention:** such points have *no rows* in the candidate table (0.7% of train points). `candidates.points_without_candidates()` lists them, and HMM assigns them −1.

## Data-quality findings (for report §2)
- **Two train trips have a systematic GPS offset** relative to their own network and route: **00000054** is about 138 m north (89% of its points are more than 50 m from the route) and **00000033** is about 115 m south (59%). The offset varies along the trip (std 25–60 m), so it can't be corrected. They are excluded by rule (`labels.min_labelled_share: 0.5`, not by hardcoded IDs). The next-lowest trip is at 88%, so the threshold isn't borderline. Train goes from 80 trips to **78** for modelling.
- **Time gaps:** the largest are 153,064 s (00000090, about 42 h) and 52,810 s (00000078). Tracks are split into HMM segments at gaps over 60 s (`segment_id`).
- **Directed arcs:** a two-way street is two arcs with identical geometry, so twins always come with equal `dist_m`. Only direction (heading vs `arc_bearing_deg`) can tell them apart. One-way share per trip ranges from 0% to 65%.
- **Reference route contiguity** is 98–100%, so the reference itself has small breaks, and labels are derived rather than observed. State this when reporting accuracies.
- **Class balance:** about 16% of candidate rows are positive (1 correct arc out of about 6 candidates per point).
- Many trips are slow (few points above 8 m/s), so mixed vehicles, cycling or walking speeds are likely.
