"""Build candidate tables for train + student-test partitions.  Owner: P1
Output: data/processed/candidates_train.parquet, candidates_test.parquet (schema.py columns + features + label).
Run:    python -m scripts.build_dataset [--trips 00000046 ...]  (start with one small trip: 00000046, 649 arcs)
Uses ONLY dataset/train and dataset/test. Calls features.add_features (P2) once it exists.
"""
raise NotImplementedError
