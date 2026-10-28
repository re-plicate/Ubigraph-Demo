"""Deterministic ordered assignment and four-neighbour space-time A* reservations."""
import copy
import heapq
import math

from .scene import HEIGHT, HEIGHT_CELLS, HOMES, NODES, OBSTACLES, RESERVE, SLOT, WIDTH


def cell(drone):
    return round(drone.x), round(drone.y)


def distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(start, goal, blocked, start_tick=0, vertices=None, edges=None, horizon=600, goal_hold=0, final_stop=False):
    vertices, edges = vertices or set(), edges or set()
    if goal in blocked or start in blocked:
        raise ValueError("目标或当前位置被障碍/其他无人机占用")
    first = (*start, start_tick)
    queue, cost, parent = [(distance(start, goal), start_tick, *start)], {first: 0}, {}
    while queue:
        _, t, x, y = heapq.heappop(queue)
        current = (x, y, t)
        until = horizon if final_stop else min(horizon, t + goal_hold)
        if (x, y) == goal and not any((x, y, future) in vertices for future in range(t, until + 1)):
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            return list(reversed(path))
        if t >= horizon:
            continue
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)):
            nx, ny, nt = x + dx, y + dy, t + 1
            nxt = nx, ny, nt
            if not (0 <= nx < WIDTH and 0 <= ny < HEIGHT_CELLS) or (nx, ny) in blocked:
                continue
            if nxt in vertices or ((nx, ny), (x, y), nt) in edges:
                continue
            nc = cost[current] + 1
            if nc < cost.get(nxt, math.inf):
                cost[nxt], parent[nxt] = nc, current
                heapq.heappush(queue, (nc + distance((nx, ny), goal), nt, nx, ny))
    raise ValueError("在规划时限内找不到可行路径，请减少目标或调整场景")


def route_length(path):
    return sum(distance(a, b) for a, b in zip(path, path[1:]))


def estimate_order(start, order, blocked, home, return_home):
    here, total = start, 0
    for target in order:
        goal = NODES[target]["x"], NODES[target]["y"]
        total += len(astar(here, goal, blocked)) - 1
        here = goal
    if return_home:
        total += len(astar(here, home, blocked)) - 1
    return total


def assign(drones, targets, dwell_seconds, return_home, all_drones=None):
    """Input target order already expresses external priority. No acquisition trigger."""
    orders = {d.id: [] for d in drones}
    reasons = []
    for target in targets:
        choices = []
        for drone in drones:
            blocked = OBSTACLES | {cell(other) for other in (all_drones or drones) if other.id != drone.id}
            # Include all registered home cells of other vehicles as parked reservations.
            blocked |= {p for key, p in HOMES.items() if key != drone.id}
            # Existing priority order is preserved; equal-priority insertion is not necessary.
            order = orders[drone.id] + [target]
            try:
                old = estimate_order(cell(drone), orders[drone.id], blocked, HOMES[drone.id], return_home)
                length = estimate_order(cell(drone), order, blocked, HOMES[drone.id], return_home)
                last = NODES[order[-1]]
                return_length = 0 if return_home else len(astar((last["x"], last["y"]), HOMES[drone.id], blocked)) - 1
            except ValueError:
                continue
            duration = (length + 2) * SLOT + len(order) * max(1, math.ceil(dwell_seconds / SLOT)) * SLOT
            # Always require enough energy to return, even if the mission ends hovering.
            reserve_return = 0 if return_home else .4 * return_length + .06 * (return_length + 1) * SLOT + 3
            energy = .4 * length + .06 * duration + 3 + reserve_return
            if drone.battery - energy < RESERVE:
                continue
            choices.append((length * SLOT + len(order) * dwell_seconds, length - old, drone.id, order))
        if not choices:
            reasons.append({"target": target, "reason": "没有满足路径与返航电量约束的可用无人机"})
        else:
            _, _, identity, order = min(choices)
            orders[identity] = order
    return orders, reasons


def build_routes(drones, orders, dwell_seconds, return_home, all_drones, mode="collection", height=HEIGHT):
    vertices, edges, routes = set(), set(), {}
    starts = {d.id: cell(d) for d in all_drones}
    dwell_ticks = max(1, math.ceil(dwell_seconds / SLOT))
    for drone in sorted(drones, key=lambda d: d.id):
        order = orders.get(drone.id, [])
        if not order and mode == "collection":
            continue
        here, tick = cell(drone), 0
        blocked = OBSTACLES | {p for key, p in starts.items() if key != drone.id}
        blocked |= {p for key, p in HOMES.items() if key != drone.id}
        path = [{"x": here[0], "y": here[1], "tick": 0, "phase": "ready", "z": drone.z}]
        jobs = []

        def hold(count, phase, z):
            nonlocal tick
            for _ in range(count):
                tick += 1
                if (*here, tick) in vertices:
                    raise ValueError("采集/起降位置存在时间冲突")
                path.append({"x": here[0], "y": here[1], "tick": tick, "phase": phase, "z": z})

        def travel(goal, phase, hold_after=0, final_stop=False):
            nonlocal tick, here
            found = astar(here, goal, blocked, tick, vertices, edges, goal_hold=hold_after, final_stop=final_stop)
            for x, y, t in found[1:]:
                path.append({"x": x, "y": y, "tick": t, "phase": phase, "z": height})
            here, tick = goal, found[-1][2]

        if mode == "land":
            hold(1, "landing", 0.)
        else:
            if drone.z < .05:
                hold(1, "taking_off", height)
            if mode == "takeoff":
                if len(path) == 1:
                    hold(1, "hovering", height)
            elif mode == "hover":
                hold(dwell_ticks, "hovering", height)
            else:
                for index, target in enumerate(order):
                    goal = NODES[target]["x"], NODES[target]["y"]
                    travel(goal, "travelling", dwell_ticks if mode == "collection" else 0,
                           not return_home and index == len(order) - 1)
                    if mode == "collection":
                        arrival = tick
                        hold(dwell_ticks, "collecting", height)
                        jobs.append({"node": target, "arrival_tick": arrival, "end_tick": tick,
                                     "status": "pending", "receipt": None})
                if return_home or mode == "return":
                    travel(HOMES[drone.id], "returning")
                    hold(1, "landing", 0.)
        if len(path) == 1:
            hold(1, "hovering", height)
        for point in path:
            vertices.add((point["x"], point["y"], point["tick"]))
        for a, b in zip(path, path[1:]):
            edges.add(((a["x"], a["y"]), (b["x"], b["y"]), b["tick"]))
        final = path[-1]
        for t in range(final["tick"], 601):
            vertices.add((final["x"], final["y"], t))
        routes[drone.id] = {"path": path, "jobs": jobs, "targets": order,
                            "duration_seconds": tick * SLOT, "distance_m": route_length([(p["x"], p["y"]) for p in path])}
    return routes


def energy_requirements(routes, all_drones, include_future_return=True):
    """Budget execution, joint-mission hovering, and an obstacle-aware future return.

    Parked airborne vehicles also need energy while other vehicles occupy the
    shared airspace. Projected copies never change the live fleet.
    """
    if not include_future_return:
        return {key: .4 * r["distance_m"] + .06 * r["duration_seconds"] + 3 for key, r in routes.items()}
    projected = [copy.copy(d) for d in all_drones]
    duration = max((r["duration_seconds"] for r in routes.values()), default=0.)
    need = {}
    for d in projected:
        if d.id in routes:
            route = routes[d.id]
            end = route["path"][-1]
            d.x, d.y, d.z = end["x"], end["y"], end["z"]
            need[d.id] = .4 * route["distance_m"] + .06 * route["duration_seconds"] + 3
            if d.z > .05:
                need[d.id] += .06 * (duration - route["duration_seconds"])
        elif d.z > .05 and not d.fault:
            need[d.id] = .06 * duration
    airborne = [d for d in projected if d.z > .05 and not d.fault]
    if airborne:
        back = build_routes(airborne, {}, 0, True, projected, mode="return")
        for key, route in back.items():
            need[key] = need.get(key, 0.) + .4 * route["distance_m"] + .06 * route["duration_seconds"] + 3
    return need
