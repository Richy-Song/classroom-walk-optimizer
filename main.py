"""
Hybrid solver, gradient-free version.

Pipeline per candidate category order:
    (a) heuristic A : random placement + swap-based local search
    (b) heuristic B : greedy min-sum placement from pairwise demand
    (c) polish      : coordinate descent + simulated annealing (both seeds)

No torch, no autograd, no soft relaxation. The category order is handled
by trying several sampled permutations and letting SA swap entries in it.
"""

import math
import random
import itertools

from visualize import visualize_solution, visualize_comparison, animate_paths


# ---------------------------------------------------------------------------
# 1. Problem data
# ---------------------------------------------------------------------------

# def build_toy_instance(seed=0):
#     rng = random.Random(seed)
#     N, M = 4, 4
#     categories = list(range(1, 7))
#     identifiers = list("ABC")
#     labels = [(c, i) for c in categories for i in identifiers]
#     robots = [{c: (c, rng.choice(identifiers)) for c in categories}
#               for _ in range(6)]
#     return N, M, categories, labels, robots

def build_toy_instance(seed=0, num_students=50, n=2, m=6):
    """
    Applied IB-style instance.

    Course letters per module (category):
        1: A,B,C,D,E,F  = English LAL SL/HL, English L SL/HL, Chinese SL/HL
        2: A,B,C,D      = English B, Chinese B, Spanish ab, French ab
        3: A,B,C,D      = MAA SL/HL, MAI SL/HL
        4: A,B,C,D,E    = Econ, Hist, Psych, Geog, Phil
        5: F,G,H,I      = Phys, Chem, Bio, CS   (letters F-I stay unique)
        6: A-L          = M4 ∪ M5 ∪ {VA, Music, Theater}

    Module 6 rule: a student's M6 letter must differ from their M4 and M5
    letters, i.e. "another" subject. Different periods give different labels,
    so (4,'A') and (6,'A') are distinct grid labels.
    """
    import random
    rng = random.Random(seed)
    catalog = {
        1: list("ABCDEF"),
        2: list("ABCD"),
        3: list("ABCD"),
        4: list("ABCDE"),
        5: list("FGHI"),
        6: list("ABCDEFGHIJKL"),
    }
    labels = [(c, x) for c in catalog for x in catalog[c]]
    robots = []
    for _ in range(num_students):
        sched = {c: (c, rng.choice(catalog[c])) for c in (1, 2, 3, 4, 5)}
        m4, m5 = sched[4][1], sched[5][1]
        options = [x for x in catalog[6] if x not in (m4, m5)]
        sched[6] = (6, rng.choice(options))
        robots.append(sched)
    return n, m, list(catalog), labels, robots

# ---------------------------------------------------------------------------
# 2. Discrete utilities
# ---------------------------------------------------------------------------

def all_positions(N, M):
    return [(x, y) for x in range(N) for y in range(M)]

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def robot_cost(req, order, placement):
    seq = [placement[req[c]] for c in order]
    return sum(manhattan(seq[i], seq[i + 1]) for i in range(len(seq) - 1))

def average_cost(robots, order, placement):
    return sum(robot_cost(r, order, placement) for r in robots) / len(robots)

def placement_is_valid(placement):
    seen = set()
    for (cat, _), pos in placement.items():
        if (pos, cat) in seen:
            return False
        seen.add((pos, cat))
    return True


# ---------------------------------------------------------------------------
# 3. (a) Heuristic A: random placement + swap local search
# ---------------------------------------------------------------------------

def random_placement(labels, N, M, rng):
    pos_list = all_positions(N, M)
    occupied = set()
    placement = {}
    shuffled = labels[:]
    rng.shuffle(shuffled)
    for l in shuffled:
        cat = l[0]
        free = [p for p in pos_list if (p, cat) not in occupied]
        p = rng.choice(free)
        placement[l] = p
        occupied.add((p, cat))
    return placement

def swap_local_search(robots, order, placement, rng, iters=400):
    labels = list(placement.keys())
    cur = average_cost(robots, order, placement)
    for _ in range(iters):
        a, b = rng.sample(labels, 2)
        placement[a], placement[b] = placement[b], placement[a]
        if not placement_is_valid(placement):
            placement[a], placement[b] = placement[b], placement[a]
            continue
        new = average_cost(robots, order, placement)
        if new < cur:
            cur = new
        else:
            placement[a], placement[b] = placement[b], placement[a]
    return placement, cur


# ---------------------------------------------------------------------------
# 4. (b) Heuristic B: greedy min-sum placement
# ---------------------------------------------------------------------------

def greedy_placement(robots, order, labels, N, M, rng):
    """
    Build pairwise demand W[l1,l2] = #robots that need l1 immediately
    before l2 in this category order. Then place labels one at a time
    (most-connected first) at the position that minimises the weighted
    sum of distances to already-placed neighbours.
    """
    weight = {}
    for r in robots:
        seq = [r[c] for c in order]
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            key = (a, b) if a <= b else (b, a)
            weight[key] = weight.get(key, 0) + 1

    strength = {l: 0 for l in labels}
    for (a, b), w in weight.items():
        strength[a] += w
        strength[b] += w
    ordered_labels = sorted(labels, key=lambda l: -strength[l])

    pos_list = all_positions(N, M)
    center = (N / 2.0, M / 2.0)
    placement = {}
    occupied = set()

    for l in ordered_labels:
        cat = l[0]
        best_pos, best_score = None, float("inf")
        for p in pos_list:
            if (p, cat) in occupied:
                continue
            score = 0.0
            for other_l, other_pos in placement.items():
                key = (l, other_l) if l <= other_l else (other_l, l)
                w = weight.get(key, 0)
                if w:
                    score += w * manhattan(p, other_pos)
            score += 0.01 * manhattan(p, center)   # gentle centre bias
            if score < best_score:
                best_score, best_pos = score, p
        placement[l] = best_pos
        occupied.add((best_pos, cat))
    return placement


# ---------------------------------------------------------------------------
# 5. (c) Polishing: coordinate descent + simulated annealing
# ---------------------------------------------------------------------------

def coordinate_descent(robots, order, placement, N, M, rng, sweeps=6):
    """
    Discrete 'gradient-like' step: for each label, try every feasible
    position and pick the best, keeping others fixed. Rinse and repeat.
    """
    labels = list(placement.keys())
    pos_list = all_positions(N, M)
    cur = average_cost(robots, order, placement)

    for _ in range(sweeps):
        improved = False
        rng.shuffle(labels)
        for l in labels:
            cat = l[0]
            other_positions = {placement[o] for o in labels
                               if o != l and o[0] == cat}
            old_pos = placement[l]
            best_pos, best_cost = old_pos, cur
            for p in pos_list:
                if p in other_positions:
                    continue
                placement[l] = p
                c = average_cost(robots, order, placement)
                if c < best_cost - 1e-9:
                    best_cost, best_pos = c, p
            placement[l] = best_pos
            if best_pos != old_pos:
                cur = best_cost
                improved = True
        if not improved:
            break
    return placement, cur

def anneal(robots, order, placement, rng, iters=800, T0=1.0, T1=0.01):
    cur = average_cost(robots, order, placement)
    best_cost, best_order, best_placement = cur, order, placement.copy()
    labels = list(placement.keys())
    for it in range(iters):
        T = T0 * (T1 / T0) ** (it / max(1, iters - 1))
        if rng.random() < 0.7:
            a, b = rng.sample(labels, 2)
            placement[a], placement[b] = placement[b], placement[a]
            if not placement_is_valid(placement):
                placement[a], placement[b] = placement[b], placement[a]
                continue
            new = average_cost(robots, order, placement)
            delta = new - cur
            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur = new
                if new < best_cost:
                    best_cost, best_order, best_placement = (
                        new, order, placement.copy())
            else:
                placement[a], placement[b] = placement[b], placement[a]
        else:
            ol = list(order)
            i, j = rng.sample(range(6), 2)
            ol[i], ol[j] = ol[j], ol[i]
            new_order = tuple(ol)
            new = average_cost(robots, new_order, placement)
            delta = new - cur
            if delta < 0 or rng.random() < math.exp(-delta / T):
                cur = new
                order = new_order
                if new < best_cost:
                    best_cost, best_order, best_placement = (
                        new, order, placement.copy())
    return best_placement, best_order, best_cost


# ---------------------------------------------------------------------------
# 6. Driver
# ---------------------------------------------------------------------------

def solve(N, M, categories, labels, robots, rng, n_orders=3):
    all_orders = list(itertools.permutations(categories))
    rng.shuffle(all_orders)
    candidates = all_orders[:n_orders]
    best = (float("inf"), None, None)

    for order in candidates:
        print(f"\n=== category order {order} ===")

        # (a) heuristic A
        pA = random_placement(labels, N, M, rng)
        pA, cA = swap_local_search(robots, order, pA, rng)
        print(f"  heur A (random + LS)       : {cA:.3f}")

        # (b) heuristic B
        pB = greedy_placement(robots, order, labels, N, M, rng)
        cB = average_cost(robots, order, pB)
        print(f"  heur B (greedy min-sum)    : {cB:.3f}")

        # (c) polish both with coordinate descent then SA
        pA, cA = coordinate_descent(robots, order, pA, N, M, rng)
        pB, cB = coordinate_descent(robots, order, pB, N, M, rng)
        print(f"  coord-descent (from A)     : {cA:.3f}")
        print(f"  coord-descent (from B)     : {cB:.3f}")

        pA_f, oA_f, cA_f = anneal(robots, order, pA.copy(), rng, iters=800)
        pB_f, oB_f, cB_f = anneal(robots, order, pB.copy(), rng, iters=800)
        print(f"  SA polish (from A)         : {cA_f:.3f}  order {oA_f}")
        print(f"  SA polish (from B)         : {cB_f:.3f}  order {oB_f}")

        c, o, p = (cA_f, oA_f, pA_f) if cA_f <= cB_f else (cB_f, oB_f, pB_f)
        if c < best[0]:
            best = (c, o, p)

    return best


if __name__ == "__main__":
    rng = random.Random(42)
    N, M, cats, labels, robots = build_toy_instance(seed=0)

    cost, order, placement = solve(N, M, cats, labels, robots, rng, n_orders=3)

    print("\n===== FINAL =====")
    print(f"best average cost   : {cost:.3f}")
    print(f"best category order : {order}")
    for l in labels:
        print(f"  {l} -> {placement[l]}")

    placement_before = random_placement(labels, N, M, rng)
    placement_before, _ = swap_local_search(robots, order, placement_before, rng)
    placement_before, _ = coordinate_descent(
        robots, order, placement_before, N, M, rng)
    placement_before, _, _ = anneal(robots, order, placement_before, rng)

    fig = visualize_solution(N, M, labels, robots, order, placement_before,
                            title="Polished solution")
    fig.savefig("solution.png", dpi=150)

    # Optional animation
    animate_paths(N, M, robots, order, placement_before, "paths.gif")