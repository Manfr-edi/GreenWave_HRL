from __future__ import annotations

"""Minimal preprocessing utilities required by the standalone RL module."""

import os
import xml.etree.ElementTree as ET

import pandas as pd

from libraries.constants import EDGE_DATA_FILE_PATH


def _hour_columns_for_slot(time_slot: str) -> list[str]:
    """Return one or more hourly CSV columns covered by a requested time slot."""

    first_hour = int(time_slot[:2])
    last_hour = int(time_slot[6:8])
    if last_hour - first_hour <= 1:
        return [time_slot]

    columns = []
    for hour in range(first_hour, last_hour):
        next_hour = hour + 1
        end_label = "24" if next_hour == 24 else f"{next_hour % 24:02d}"
        columns.append(f"{hour:02d}:00-{end_label}:00")
    return columns


def generateEdgeDataFile(input_file: str, date: str = "01/02/2024", time_slot: str = "00:00-01:00", duration: str = "3600"):
    """Generate the SUMO `edgedata` XML consumed by `routeSampler.py`.

    The input CSV must contain one row per detector edge, a `data` date column,
    an `edge_id` column, and hourly count columns such as `07:00-08:00`.
    """

    df = pd.read_csv(input_file, sep=";")
    if "data" not in df.columns or "edge_id" not in df.columns:
        raise ValueError("Traffic-flow CSV must contain at least 'data' and 'edge_id' columns.")
    hour_columns = _hour_columns_for_slot(time_slot)
    missing_columns = [column for column in hour_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Traffic-flow CSV does not contain the requested time slot columns: {missing_columns}.")

    df = df[df["data"].astype(str).str.contains(str(date), regex=False)]
    if df.empty:
        raise ValueError(f"No traffic-flow rows found for date '{date}'.")

    root = ET.Element("data")
    interval = ET.SubElement(root, "interval", begin="0", end=str(duration))

    for _, row in df.iterrows():
        edge_id = str(row["edge_id"])
        count = sum(float(row[column]) for column in hour_columns)
        ET.SubElement(interval, "edge", id=edge_id, entered=str(int(count)))

    os.makedirs(os.path.dirname(EDGE_DATA_FILE_PATH), exist_ok=True)
    tree = ET.ElementTree(root)
    ET.indent(tree, "  ")
    tree.write(EDGE_DATA_FILE_PATH, encoding="UTF-8", xml_declaration=True)
    print(f"Edge data XML saved at '{EDGE_DATA_FILE_PATH}'")
