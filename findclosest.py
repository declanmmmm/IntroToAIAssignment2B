from graph import build_graph, haversine


def find_closest_nodes(nodes, target_site, top_n=5):
    distances = []

    for other_site in nodes:

        if other_site == target_site:
            continue

        dist = haversine(
            nodes[target_site]["lat"],
            nodes[target_site]["lon"],
            nodes[other_site]["lat"],
            nodes[other_site]["lon"]
        )

        distances.append((other_site, dist))

    distances.sort(key=lambda x: x[1])

    return distances[:top_n]


def main():

    print("Building graph...")

    nodes, edges = build_graph()

    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {sum(len(v) for v in edges.values()) // 2}")

    print("\nSearching for isolated nodes...\n")

    isolated = []

    for site, neighbours in edges.items():
        if len(neighbours) == 0:
            isolated.append(site)

    print(f"Found {len(isolated)} isolated nodes\n")

    if len(isolated) == 0:
        print("No isolated nodes found.")
        return

    for site in isolated:

        print("=" * 80)
        print(f"ISOLATED SITE: {site}")
        print(f"LOCATION: {nodes[site]['location']}")
        print()

        closest = find_closest_nodes(nodes, site)

        print("5 Closest SCATS Sites:")
        print()

        for other_site, distance in closest:

            print(
                f"{other_site:<6} "
                f"{distance:>6.2f} km   "
                f"{nodes[other_site]['location']}"
            )

        print()


if __name__ == "__main__":
    main()