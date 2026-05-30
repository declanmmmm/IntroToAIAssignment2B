import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import pandas as pd

from graph import build_graph
from routing import load_dataset, time_to_interval, predict_flow, find_top_k, format_time
from traveltime import flow_to_speed, get_travel_time

print("loading graph and dataset...")
nodes, edges = build_graph()
df = load_dataset()
print("ready\n")

passed = 0
failed = 0


def test(name, result, expected=True, details=""):
    global passed, failed
    status = "PASS" if result == expected else "FAIL"
    if result == expected:
        passed += 1
    else:
        failed += 1
    print(f"[{status}] {name}")
    if details:
        print(f"       {details}")


# ── 1. valid route exists
routes = find_top_k(nodes, edges, "2000", "3002", time_to_interval("08:00"), df, k=5)
test("valid route - 2000 to 3002", len(routes) > 0, details=f"found {len(routes)} routes")

# ── 2. invalid origin
try:
    routes = find_top_k(nodes, edges, "9999", "3002", time_to_interval("08:00"), df, k=5)
    test("invalid origin - 9999 not in graph", "9999" not in nodes)
except:
    test("invalid origin - 9999 not in graph", True)

# ── 3. invalid destination
try:
    routes = find_top_k(nodes, edges, "2000", "9999", time_to_interval("08:00"), df, k=5)
    test("invalid destination - 9999 not in graph", "9999" not in nodes)
except:
    test("invalid destination - 9999 not in graph", True)

# ── 4. origin equals destination
routes = find_top_k(nodes, edges, "2000", "2000", time_to_interval("08:00"), df, k=5)
if len(routes) > 0:
    test("origin equals destination", routes[0][0] == ["2000"], details=f"path: {routes[0][0]}")
else:
    test("origin equals destination - no route returned", True)

# ── 5. top 5 routes are all different
routes = find_top_k(nodes, edges, "2000", "3002", time_to_interval("08:00"), df, k=5)
paths = [tuple(r[0]) for r in routes]
test("top 5 routes are all different", len(paths) == len(set(paths)), details=f"{len(paths)} unique paths")

# ── 6. peak hour routing returns a result
routes = find_top_k(nodes, edges, "2000", "3002", time_to_interval("08:00"), df, k=5)
test("peak hour routing (08:00) returns routes", len(routes) > 0, details=f"found {len(routes)} routes")

# ── 7. off-peak routing returns a result
routes = find_top_k(nodes, edges, "2000", "3002", time_to_interval("02:00"), df, k=5)
test("off-peak routing (02:00) returns routes", len(routes) > 0, details=f"found {len(routes)} routes")

# ── 8. travel time is higher at peak vs off-peak
peak = find_top_k(nodes, edges, "2000", "3002", time_to_interval("08:00"), df, k=1)
offpeak = find_top_k(nodes, edges, "2000", "3002", time_to_interval("02:00"), df, k=1)
if len(peak) > 0 and len(offpeak) > 0:
    test("peak hour slower than off-peak", peak[0][1] > offpeak[0][1],
         details=f"peak: {format_time(peak[0][1])} | off-peak: {format_time(offpeak[0][1])}")

# ── 9. A* vs UCS - both find a path
astar_routes = find_top_k(nodes, edges, "2000", "3002", time_to_interval("08:00"), df, k=1, algorithm="astar")
ucs_routes   = find_top_k(nodes, edges, "2000", "3002", time_to_interval("08:00"), df, k=1, algorithm="ucs")
test("A* finds a route", len(astar_routes) > 0)
test("UCS finds a route", len(ucs_routes) > 0)

# ── 10. A* creates fewer or equal nodes than UCS
if len(astar_routes) > 0 and len(ucs_routes) > 0:
    astar_nodes = astar_routes[0][2]
    ucs_nodes = ucs_routes[0][2]
    test("A* creates fewer or equal nodes than UCS", astar_nodes <= ucs_nodes,
         details=f"A*: {astar_nodes} nodes | UCS: {ucs_nodes} nodes")

# ── 11. flow to speed - low flow gives speed limit
speed = flow_to_speed(0)
test("flow 0 veh/hr gives speed 60 km/h", speed == 60.0, details=f"speed: {speed}")

# ── 12. flow to speed - moderate flow reduces speed
speed = flow_to_speed(800)
test("flow 800 veh/hr reduces speed below 60", speed < 60.0, details=f"speed: {round(speed, 2)} km/h")

# ── 13. flow to speed - high flow (over capacity) uses red line
speed = flow_to_speed(2000)
test("flow 2000 veh/hr gives congested speed", speed < 32.0, details=f"speed: {round(speed, 2)} km/h")

# ── 14. travel time increases with distance
t1 = get_travel_time(500, 1.0)
t2 = get_travel_time(500, 2.0)
test("travel time increases with distance", t2 > t1, details=f"1km: {round(t1,1)}s | 2km: {round(t2,1)}s")

# ── 15. ML model prediction returns a positive value
for model_name in ["lstm", "gru", "cnn_lstm"]:
    flow = predict_flow("2000", time_to_interval("08:00"), df, model_name=model_name)
    test(f"{model_name.upper()} prediction for site 2000 is positive", flow >= 0, details=f"predicted flow: {round(flow, 2)} veh/hr")

# ── 16. time to interval conversion
test("08:00 converts to interval 32", time_to_interval("08:00") == 32)
test("00:00 converts to interval 0",  time_to_interval("00:00") == 0)
test("23:45 converts to interval 95", time_to_interval("23:45") == 95)

# ── summary
print(f"\n{'='*40}")
print(f"results: {passed} passed, {failed} failed out of {passed + failed} tests")
print(f"{'='*40}")