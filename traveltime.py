import math

SPEED_LIMIT = 60
INTERSECTION_DELAY = 30 / 3600

def flow_to_speed(flow_per_hour):
    # from the assignment doc: flow = -1.4648375 * speed^2 + 93.75 * speed
    # rearranged to solve for speed using quadratic formula
    a = 1.4648375
    b = -93.75

    def solve_red(flow):
        disc = b**2 - 4 * a * flow
        if disc < 0:
            return 1
        return (-b - math.sqrt(disc)) / (2 * a)

    def solve_green(flow):
        disc = b**2 - 4 * a * flow
        if disc < 0:
            return SPEED_LIMIT
        return (-b + math.sqrt(disc)) / (2 * a)

    if flow_per_hour <= 1500:
        speed = solve_green(flow_per_hour)
    else:
        # mirror around capacity point (1500)
        # e.g. 2000 veh/hr -> 1500 - 500 = 1000 -> use red line at 1000
        mirrored = 3000 - flow_per_hour
        mirrored = max(mirrored, 0)
        speed = solve_red(mirrored)

    speed = min(speed, SPEED_LIMIT)
    speed = max(speed, 1)
    return speed


def get_travel_time(flow_per_hour, distance_km):
    speed = flow_to_speed(flow_per_hour)
    travel_time = distance_km / speed
    travel_time += INTERSECTION_DELAY
    return travel_time * 3600


if __name__ == "__main__":
    for flow in [0, 351, 800, 1500, 2000, 2500, 3000]:
        speed = flow_to_speed(flow)
        print(f"flow: {flow} veh/hr -> speed: {speed:.2f} km/h")

    print()
    t = get_travel_time(800, 1.5)
    print(f"travel time for 800 veh/hr over 1.5km: {t:.1f} seconds")