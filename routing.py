import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from graph import build_graph, haversine
from traveltime import get_travel_time


LOOKBACK = 8
traffic_cols = [f"V{str(i).zfill(2)}" for i in range(96)]


def load_dataset(xlsx_path="Datasets/Dataonly.xlsx"):
    df = pd.read_excel(xlsx_path)
    df.columns = df.iloc[0]
    df = df.drop(0).reset_index(drop=True)
    df["SCATS Number"] = df["SCATS Number"].astype(str).str.strip()
    return df


def time_to_interval(time_str):
    # convert "HH:MM" to interval index 0-95
    h, m = map(int, time_str.split(":"))
    return (h * 60 + m) // 15


def predict_flow(site_id, interval, df, model_dir="saved_models", model_name="lstm"):
    model_path = f"{model_dir}/{model_name}_{site_id}.keras"
    if not os.path.exists(model_path):
        # fall back to average if no model saved
        group = df[df["SCATS Number"] == site_id]
        raw = group[traffic_cols].astype(float).values.flatten()
        avg = np.nanmean(raw[max(0, interval-8):interval]) if interval > 0 else np.nanmean(raw)
        return avg * 4  # convert to per hour

    group = df[df["SCATS Number"] == site_id]
    raw = group[traffic_cols].astype(float).values.flatten()
    series = pd.Series(raw).ffill().bfill().values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    # get the 8 intervals before the target interval
    if interval < LOOKBACK:
        seq = scaled[:LOOKBACK]
    else:
        seq = scaled[interval - LOOKBACK:interval]

    model = load_model(model_path)
    X = seq.reshape(1, LOOKBACK, 1)
    pred_scaled = model.predict(X, verbose=0)
    pred = scaler.inverse_transform(pred_scaled)[0][0]

    return max(0, pred) * 4  # convert to per hour


def get_edge_cost(site_a, site_b, dist_km, interval, df, model_name="lstm"):
    flow = predict_flow(site_a, interval, df, model_name=model_name)
    return get_travel_time(flow, dist_km)


# UCS - finds shortest path by travel time
def ucs(nodes, edges, origin, destination, interval, df, model_name="lstm"):
    frontier = [(0, origin, [origin])]
    visited = set()
    created = 1

    while frontier:
        frontier.sort(key=lambda x: x[0])
        cost, current, path = frontier.pop(0)

        if current in visited:
            continue
        visited.add(current)

        if current == destination:
            return path, cost, created

        for neighbour, dist in edges.get(current, []):
            if neighbour not in visited:
                edge_cost = get_edge_cost(current, neighbour, dist, interval, df, model_name)
                frontier.append((cost + edge_cost, neighbour, path + [neighbour]))
                created += 1

    return None, float("inf"), created


# A* - uses straight line distance to destination as heuristic
def astar(nodes, edges, origin, destination, interval, df, model_name="lstm"):
    def heuristic(node):
        if node not in nodes or destination not in nodes:
            return 0
        dist = haversine(nodes[node]["lat"], nodes[node]["lon"],
                         nodes[destination]["lat"], nodes[destination]["lon"])
        return dist / 60 * 3600  # assume speed limit for heuristic (seconds)

    frontier = [(heuristic(origin), 0, origin, [origin])]
    visited = set()
    created = 1

    while frontier:
        frontier.sort(key=lambda x: x[0])
        f, cost, current, path = frontier.pop(0)

        if current in visited:
            continue
        visited.add(current)

        if current == destination:
            return path, cost, created

        for neighbour, dist in edges.get(current, []):
            if neighbour not in visited:
                edge_cost = get_edge_cost(current, neighbour, dist, interval, df, model_name)
                g = cost + edge_cost
                frontier.append((g + heuristic(neighbour), g, neighbour, path + [neighbour]))
                created += 1

    return None, float("inf"), created


# find top-k paths by temporarily blocking edges from previous paths
def find_top_k(nodes, edges, origin, destination, interval, df, k=5, algorithm="astar", model_name="lstm"):
    results = []
    blocked = set()

    search_fn = astar if algorithm == "astar" else ucs

    for _ in range(k):
        # temporarily remove blocked edges
        filtered_edges = {}
        for node, neighbours in edges.items():
            filtered_edges[node] = [(n, d) for n, d in neighbours if (node, n) not in blocked]

        path, cost, created = search_fn(nodes, filtered_edges, origin, destination, interval, df, model_name)

        if path is None:
            break

        results.append((path, cost, created))

        # block one edge from this path to force a different route next time
        if len(path) >= 2:
            mid = (len(path) - 2) // 2

            blocked.add((path[mid], path[mid + 1]))
            blocked.add((path[mid + 1], path[mid]))

    return results


def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s}s"


if __name__ == "__main__":
    print("loading dataset and graph...")
    df = load_dataset()
    nodes, edges = build_graph()

    origin = input("enter origin SCATS site: ").strip().zfill(4)
    destination = input("enter destination SCATS site: ").strip().zfill(4)
    time_str = input("enter time (HH:MM): ").strip()
    model_name = input("model to use (lstm/gru/cnn_lstm): ").strip() or "lstm"

    if origin not in nodes:
        print(f"site {origin} not found in graph")
        exit()
    if destination not in nodes:
        print(f"site {destination} not found in graph")
        exit()

    interval = time_to_interval(time_str)
    print(f"\nfinding top 5 routes from {origin} to {destination} at {time_str}...")

    for algo in ["astar", "ucs"]:
        print(f"\n--- {algo.upper()} ---")
        routes = find_top_k(nodes, edges, origin, destination, interval, df, k=5, algorithm=algo, model_name=model_name)

        if not routes:
            print("no routes found")
            continue

        for i, (path, cost, created) in enumerate(routes, 1):
            print(f"\nroute {i}: {' -> '.join(path)}")
            print(f"  estimated travel time: {format_time(cost)}")
            print(f"  nodes created: {created}")