"""Hybrid solver, gradient-free, fill-the-grid form.

    placement[(row, col)][module] = letter
"""

import math
import random
import itertools

from visualize_filled import visualize_solution, visualize_comparison, animate_paths

# ---------------------------------------------------------------------------
# Instance
# ---------------------------------------------------------------------------

def build_toy_instance(seed=0, num_students=50, n=2, m=6):
    rng = random.Random(seed)
    catalog = {
        1: list("ABCDEF"),
        2: list("ABCD"),
        3: list("ABCD"),
        4: list("ABCDE"),
        5: list("FGHI"),
        6: list("ABCDEFGHIJKL"),
    }
    robots = []
    for _ in range(num_students):
        sched = {c: rng.choice(catalog[c]) for c in (1, 2, 3, 4, 5)}
        options = [x for x in catalog[6] if x not in (sched[4], sched[5])]
        sched[6] = rng.choice(options)
        robots.append(sched)
    return n, m, list(catalog), catalog, robots


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def positions_of(placement, module, letter):
    return [p for p, cells in placement.items() if cells.get(module) == letter]

def robot_cost(req, order, placement):
    INF = float("inf")
    dp = {p: 0 for p in positions_of(placement, order[0], req[order[0]])}
    if not dp:
        return INF
    for module in order[1:]:
        nxt = {}
        for q in positions_of(placement, module, req[module]):
            nxt[q] = min(dp[p] + manhattan(p, q) for p in dp)
        if not nxt:
            return INF
        dp = nxt
    return min(dp.values())

def average_cost(robots, order, placement):
    total = 0.0
    for r in robots:
        c = robot_cost(r, order, placement)
        if c == float("inf"):
            return float("inf")       # infeasible placement: reject
        total += c
    return total / len(robots)


# ---------------------------------------------------------------------------
# Feasibility
# ---------------------------------------------------------------------------

def needed_letters(robots, catalog):
    return {m: {r[m] for r in robots} for m in catalog}

def covers(placement, needed, catalog):
    for m in catalog:
        present = {cells[m] for cells in placement.values()}
        if not needed[m].issubset(present):
            return False
    return True


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

def random_fill(N, M, catalog, robots, rng):
    pos_list = [(r, c) for r in range(N) for c in range(M)]
    placement = {p: {} for p in pos_list}
    needed = needed_letters(robots, catalog)

    for m in catalog:
        if len(needed[m]) > len(pos_list):
            raise ValueError(f"module {m}: not enough rooms")
        slots = pos_list[:]
        rng.shuffle(slots)
        for i, letter in enumerate(needed[m]):
            placement[slots[i]][m] = letter

    for p in pos_list:
        for m in catalog:
            if m not in placement[p]:
                placement[p][m] = rng.choice(catalog[m])
    return placement


# ---------------------------------------------------------------------------
# Coordinate descent (per-cell)
# ---------------------------------------------------------------------------

def coordinate_descent(robots, order, placement, catalog, rng, sweeps=6):
    pos_list = list(placement.keys())
    needed = needed_letters(robots, catalog)
    cur = average_cost(robots, order, placement)

    for _ in range(sweeps):
        improved = False
        rng.shuffle(pos_list)
        for p in pos_list:
            for m in catalog:
                old = placement[p][m]
                best_letter, best_cost = old, cur
                for letter in catalog[m]:
                    if letter == old:
                        continue
                    placement[p][m] = letter
                    if not covers(placement, needed, catalog):
                        continue
                    c = average_cost(robots, order, placement)
                    if c < best_cost - 1e-9:
                        best_cost, best_letter = c, letter
                placement[p][m] = best_letter
                if best_letter != old:
                    cur, improved = best_cost, True
        if not improved:
            break
    return placement, cur


# ---------------------------------------------------------------------------
# Simulated annealing (per-cell + order swap)
# ---------------------------------------------------------------------------

def anneal(robots, order, placement, catalog, rng,
           iters=800, T0=1.0, T1=0.01):
    pos_list = list(placement.keys())
    modules = list(catalog)
    needed = needed_letters(robots, catalog)

    cur = average_cost(robots, order, placement)
    best_cost = cur
    best_order = order
    best_placement = {p: dict(c) for p, c in placement.items()}

    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))

        if rng.random() < 0.7:
            p = rng.choice(pos_list)
            m = rng.choice(modules)
            old = placement[p][m]
            new_letter = rng.choice([x for x in catalog[m] if x != old])
            placement[p][m] = new_letter
            if not covers(placement, needed, catalog):
                placement[p][m] = old
                continue
            new = average_cost(robots, order, placement)
            delta = new - cur
            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur = new
                if new < best_cost:
                    best_cost, best_order = new, order
                    best_placement = {p2: dict(c) for p2, c in placement.items()}
            else:
                placement[p][m] = old
        else:
            ol = list(order)
            i, j = rng.sample(range(len(ol)), 2)
            ol[i], ol[j] = ol[j], ol[i]
            new_order = tuple(ol)
            new = average_cost(robots, new_order, placement)
            delta = new - cur
            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur = new
                order = new_order
                if new < best_cost:
                    best_cost, best_order = new, order
                    best_placement = {p2: dict(c) for p2, c in placement.items()}

    return best_placement, best_order, best_cost


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def solve(N, M, categories, catalog, robots, rng, n_orders=10):
    all_orders = list(itertools.permutations(categories))
    rng.shuffle(all_orders)
    # (cost, order, placement_final, placement_heur, placement_cd)
    best = (float("inf"), None, None, None, None)

    for candidate in all_orders[:n_orders]:
        print(f"\n=== category order {candidate} ===")

        placement = random_fill(N, M, catalog, robots, rng)
        placement_heur = {p: dict(c) for p, c in placement.items()}
        c_heur = average_cost(robots, candidate, placement)
        print(f"  random_fill : {c_heur:.3f}")

        placement, c_cd = coordinate_descent(
            robots, candidate, placement, catalog, rng)
        placement_cd = {p: dict(c) for p, c in placement.items()}
        print(f"  coord desc  : {c_cd:.3f}")

        placement, order, c = anneal(
            robots, candidate, placement, catalog, rng)
        print(f"  SA polish   : {c:.3f}  order {order}")

        if c < best[0]:
            best = (c, order, placement, placement_heur, placement_cd)

    return best


if __name__ == "__main__":
    rng = random.Random(42)
    N, M, categories, catalog, robots = build_toy_instance(seed=0)

    cost, order, placement, placement_heur, placement_cd = solve(
        N, M, categories, catalog, robots, rng)

    print("\n===== FINAL =====")
    print(f"best average cost   : {cost:.3f}")
    print(f"best category order : {order}")

    fig = visualize_solution(N, M, catalog, robots, order, placement,
                             title=f"Polished  (avg = {cost:.2f})")
    fig.savefig("solution.png", dpi=150)

    fig = visualize_comparison(N, M, catalog, robots, order,
                               placement_heur, placement)
    fig.savefig("comparison.png", dpi=150)

    animate_paths(N, M, catalog, robots, order, placement, "paths.gif")