"""Folium maps (mandatory for the report) and shared plotting helpers.  Owner: P4

Folium uses [lat, lon] — our raw files are lon, lat. Swap carefully.
Required layers (report section 6): raw GPS points, nearby road network, reference route,
predicted route; distinct colours, legend, LayerControl, fit_bounds to the track.
"""
from __future__ import annotations


def trip_map(track_lonlat, network_lines=None, reference_line=None, predicted_line=None,
             classifier_line=None, title: str = ""):
    """Return a folium.Map. Every *_line is a list of (lat, lon). Save with m.save(path)."""
    raise NotImplementedError
