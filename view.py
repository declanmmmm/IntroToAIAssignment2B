import matplotlib.pyplot as plt
from graph import build_graph

nodes, edges = build_graph()

plt.figure(figsize=(12, 10))

# draw edges
for node, neighbours in edges.items():

    x1 = nodes[node]["lon"]
    y1 = nodes[node]["lat"]

    for neighbour, _ in neighbours:

        x2 = nodes[neighbour]["lon"]
        y2 = nodes[neighbour]["lat"]

        plt.plot(
            [x1, x2],
            [y1, y2],
            color="lightgray",
            linewidth=1
        )

# draw nodes
for site, data in nodes.items():

    plt.scatter(
        data["lon"],
        data["lat"],
        s=50
    )

    plt.annotate(
        site,
        (data["lon"], data["lat"]),
        fontsize=8
    )

plt.title("SCATS Road Network")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)

plt.show()