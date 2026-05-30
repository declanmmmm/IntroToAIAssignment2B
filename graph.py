import pandas as pd
from manualedges import MANUAL_EDGES
import numpy as np
import math
import re

def haversine(lat1, lon1, lat2, lon2):
    # calculate distance in km between two lat/long points
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def get_roads(location):
    roads = re.findall(
        r'([A-Z_\.]+(?:_RD|_ST|_HWY|_AVE|_GV|_FWY|_ARTERIAL))',
        str(location)
    )
    return list(set(roads))


def build_graph(xlsx_path="Datasets/Dataonly.xlsx"):
    df = pd.read_excel(xlsx_path)
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)
    df["SCATS Number"] = df["SCATS Number"].astype(str).str.strip()

    coords = df.groupby("SCATS Number")[["Location", "NB_LATITUDE", "NB_LONGITUDE"]].first()
    coords["NB_LATITUDE"] = pd.to_numeric(coords["NB_LATITUDE"], errors="coerce")
    coords["NB_LONGITUDE"] = pd.to_numeric(coords["NB_LONGITUDE"], errors="coerce")

    # drop sites with missing coordinates
    coords = coords[coords["NB_LATITUDE"] != 0].dropna()
    coords["roads"] = coords["Location"].apply(get_roads)

    nodes = {}
    for site_id, row in coords.iterrows():
        nodes[site_id] = {
            "lat": row["NB_LATITUDE"],
            "lon": row["NB_LONGITUDE"],
            "location": row["Location"]
        }

    edges = {}
    for site_id in nodes:
        edges[site_id] = []

    # build groups for every road name
    road_groups = {}
    for site_id, row in coords.iterrows():
        for road in row["roads"]:
            if road not in road_groups:
                road_groups[road] = []
            road_groups[road].append(site_id)

    # connect sites on the same road
    for road, site_ids in road_groups.items():
        if len(site_ids) < 2:
            continue
        group = coords.loc[site_ids]
        sorted_sites = group.sort_values(["NB_LATITUDE", "NB_LONGITUDE"])
        ordered_ids = sorted_sites.index.tolist()
        for i in range(len(ordered_ids) - 1):
            a = ordered_ids[i]
            b = ordered_ids[i + 1]
            dist = haversine(
                nodes[a]["lat"],
                nodes[a]["lon"],
                nodes[b]["lat"],
                nodes[b]["lon"]
            )
            if dist < 5.0:
                if not any(x[0] == b for x in edges[a]):
                    edges[a].append((b, dist))
                if not any(x[0] == a for x in edges[b]):
                    edges[b].append((a, dist))

    # also connect any sites within 1km that aren't already connected
    # this picks up intersections between different roads
    site_ids = list(nodes.keys())
    for i in range(len(site_ids)):
        for j in range(i + 1, len(site_ids)):
            a = site_ids[i]
            b = site_ids[j]
            dist = haversine(
                nodes[a]["lat"], nodes[a]["lon"],
                nodes[b]["lat"], nodes[b]["lon"]
            )
            already_connected = any(x[0] == b for x in edges[a])
            if dist < 1.0 and not already_connected:
                edges[a].append((b, dist))
                edges[b].append((a, dist))

    for a, b in MANUAL_EDGES:
        if a in nodes and b in nodes:
            dist = haversine(nodes[a]["lat"], nodes[a]["lon"], nodes[b]["lat"], nodes[b]["lon"])
            if not any(x[0] == b for x in edges[a]):
                edges[a].append((b, dist))
                edges[b].append((a, dist))

    

    return nodes, edges


if __name__ == "__main__":
    nodes, edges = build_graph()
    print(f"nodes: {len(nodes)}")
    total_edges = sum(len(v) for v in edges.values()) // 2
    print(f"edges: {total_edges}")
    print()
    for site, neighbours in edges.items():
        print(f"{site} ({nodes[site]['location']}) -> {[(n, round(d,2)) for n,d in neighbours]}")

