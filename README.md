# classroom-walk-optimizer

Assign IB course sections to a grid of classrooms and choose a common module order so that the **average Manhattan walking distance** over all students is minimized, using gradient-free heuristics (coordinate descent + simulated annealing).

---

## 1. Problem

An N × M grid represents the school's classrooms. There are six timetable **modules** (periods of the day):

| Module | Meaning |
|---|---|
| 1 | Language A (English LAL / English L / Chinese, SL or HL) |
| 2 | Language B (English B / Chinese B / Spanish ab initio / French ab initio) |
| 3 | Mathematics (MAA / MAI, SL or HL) |
| 4 | Individuals & Societies (Economics / History / Psychology / Geography / Philosophy) |
| 5 | Sciences (Physics / Chemistry / Biology / Computer Science) |
| 6 | Elective: any *other* Module 4/5 course, or VA / Music / Theater |

SL/HL is only distinguished where the real curriculum distinguishes it (Modules 1 and 3). Each module has a fixed small **catalog** of course letters; in the toy instance:

```python
catalog = {
    1: list("ABCDEF"),         # 6 Language A variants
    2: list("ABCD"),           # 4 Language B variants
    3: list("ABCD"),           # 4 Math variants
    4: list("ABCDE"),          # 5 I&S subjects
    5: list("FGHI"),           # 4 Sciences
    6: list("ABCDEFGHIJKL"),   # 12 electives (M4 ∪ M5 ∪ Arts)
}
```

A set of **students** (robots) is sampled; each has one letter per module, with the rule that a student's Module 6 letter differs from their Module 4 and Module 5 letters ("another subject").

The school contains exactly N × M rooms, and **every room teaches exactly one course per module**. The decision is:

> For every room and every module, which course letter is taught there?
> Plus: in which order do all students visit the six modules?

Given a placement and an order, each student picks the shortest route that visits one room per module in the required order; path length is summed Manhattan distance. The objective is to minimize the average over all students.

---

## 2. Model

### Decision variables

```python
placement[(row, col)][module] = letter     # every cell filled
order = tuple of modules                   # common visit order
```

Every `(room, module)` slot holds exactly one letter. A course "appears" at every room whose `placement[room][module]` matches it, so the number of parallel sections of a course is an **output**, not an input.

### Cost

For a single student `req` (a dict `module -> letter`), the shortest valid route is a 6-step dynamic program:

```text
dp_1(p)     = 0                                   for every room p teaching req[order_1]

dp_{i+1}(q) = min over p with dp_i(p) < inf of
              ( dp_i(p) + d1(p, q) )              for every room q teaching req[order_{i+1}]
```

where `d1` is Manhattan distance. The student's cost is `min over p of dp_6(p)`; the objective is the mean over students.

This DP handles the "several rooms teach the same course" case for free: the student just uses whichever section is cheapest at each step.

### Feasibility (coverage)

A placement is **feasible** iff every course letter that any student needs appears in **at least one** room for that module:

```text
{ req[m] : student }  ⊆  { placement[p][m] : p in grid }     for every module m
```

This is the only hard constraint. There are no capacity limits in the current model.

---

## 3. Methods

The solver is a hybrid pipeline, run once per candidate category order. All components are gradient-free.

### 3.1 `build_toy_instance(seed, num_students, n, m)`

Constructs an applied instance. Returns:

```python
(N, M, categories, catalog, robots)
```

- `catalog`: the per-module letter set.
- `robots`: `num_students` sampled schedules, respecting the "Module 6 must differ" rule.

The current grid is 6 × 6 and 12 students by default.

### 3.2 Cost and feasibility helpers

| Function | Purpose |
|---|---|
| `manhattan(p, q)` | L1 distance between two grid cells. |
| `positions_of(placement, module, letter)` | All rooms teaching `letter` in `module`. |
| `robot_cost(req, order, placement)` | The 6-step DP above; returns `inf` if the student cannot complete the route. |
| `average_cost(robots, order, placement)` | Mean of `robot_cost` over all students — the objective. |
| `needed_letters(robots, catalog)` | Per-module set of letters that must appear. |
| `covers(placement, needed, catalog)` | Whether every needed letter is present in its module. |

### 3.3 `random_fill(N, M, catalog, robots, rng)` — initialization

1. Shuffle the grid positions.
2. For each module, place **each needed letter** in a distinct position so that coverage is guaranteed immediately.
3. Fill every remaining `(position, module)` slot with a uniformly random letter from the catalog.

Produces a feasible random placement. This is the only initialization; it does not try to be clever.

### 3.4 `coordinate_descent(robots, order, placement, catalog, rng, sweeps=6)` — discrete local search

For a fixed category order, repeatedly sweep the grid and try to improve each cell:

```text
for each sweep:
    shuffle positions
    for each position p:
        for each module m:
            old = placement[p][m]
            try every alternative letter:
                if placement remains feasible and cost decreases:
                    keep the best alternative
            place best_letter at p, m
    stop early if no cell changed
```

Notes:

- The move set is **single-cell substitution**, not a swap. Because the grid is fully filled, a swap is just two substitutions, and the acceptance check is cheaper this way.
- Feasibility (`covers`) is re-checked after each candidate letter; infeasible candidates are rejected immediately without computing the cost.
- Cost is recomputed from scratch on every candidate. This is fine at 6 × 6 × 6 × 12 but is the first thing to cache if you scale up.
- `sweeps` is an upper bound; the loop breaks as soon as a sweep makes no improvement, so it usually stops earlier.

**Role:** takes the raw random fill and turns it into a local optimum under single-cell moves. It is the analogue of gradient descent in the discrete space.

### 3.5 `anneal(robots, order, placement, catalog, rng, iters=800, T0=1.0, T1=0.01)` — simulated annealing

Same single-cell move set as coordinate descent, but with temperature-based acceptance that can escape local minima. Two move types:

- **70% of iterations:** pick a random cell and a random module, replace its letter with a uniformly random different letter from the catalog. If feasibility breaks, revert. Otherwise accept by the Metropolis rule:

```text
Pr[accept] = 1                if Δ <= 0
           = exp(-Δ / T)      if Δ > 0
```

- **30% of iterations:** swap two positions in the current category order, and re-evaluate. Accepted by the same rule. This lets the solver **co-optimize the order** alongside the placement instead of treating it as fixed.

Temperature schedule is geometric:

```text
T(it) = T0 * (T1 / T0)^( it / (iters - 1) )
```

The best-ever `(placement, order, cost)` triple is saved separately from the current state, so the return value is the best solution found, not just the final state.

**Role:** escapes the coordinate-descent local minimum and jointly refines the category order.

### 3.6 `solve(N, M, categories, catalog, robots, rng, n_orders=3)` — driver

For each of `n_orders` randomly chosen category orders:

```text
placement = random_fill(...)                  # initial
placement, c1 = coordinate_descent(...)       # discrete local search
placement, order, c = anneal(...)             # SA, may change order
keep the best (cost, order, placement) across orders
```

The returned tuple is `(best_cost, best_order, best_placement)`.

**Why multiple orders?** There are 6! = 720 possible category orders, and the objective is not separable across them. Trying a few random orders gives most of the benefit of the full enumeration at a small fraction of the cost; on the default 6 × 6 instance, `n_orders=3` runs in well under a second.

---

## 4. Visualization (`visualize_filled.py`)

Four panels, each answering a different question:

| Panel | Question answered |
|---|---|
| **Grid** | Where does each course sit? Every cell shows six letters, one per module, color-coded by module. |
| **Robot paths** | Which route does each student take? Step numbers show the module order. |
| **Congestion heatmap** | Which rooms are bottlenecks? Modes: `endpoint` (stops per cell) and `traversal` (approximate Manhattan segments). |
| **Per-robot cost bars** | Which students are expensive? The mean line is the objective; the spread is a fairness signal. |

Public functions:

```python
visualize_solution(N, M, catalog, robots, order, placement, title="")
visualize_comparison(N, M, catalog, robots, order, before, after)
animate_paths(N, M, catalog, robots, order, placement, filename="paths.gif")
```

All route rendering uses the same `robot_path(...)` DP as the solver, so the picture matches the optimized objective exactly.

---

## 5. Usage

### Install

```bash
pip install numpy matplotlib
```

### Run

```bash
python main_filled.py
```

This prints per-order progress, the final solution, and saves:

- `solution.png` — the four-panel view of the polished solution
- `comparison.png` — polished vs a random baseline for the same order
- `paths.gif` — an animation of students stepping through the module order

### Call from Python

```python
import random
from solver import build_toy_instance, solve
from visualize_filled import visualize_solution

rng = random.Random(42)
N, M, categories, catalog, robots = build_toy_instance(seed=0)

cost, order, placement = solve(N, M, categories, catalog, robots, rng)

fig = visualize_solution(N, M, catalog, robots, order, placement,
                         title=f"Polished  (avg = {cost:.2f})")
fig.savefig("solution.png", dpi=150)
```

### File layout

```text
main_filled.py            # instance builder, cost, feasibility, coordinate descent, SA, driver
visualize_filled.py       # 4-panel solution view, comparison view, GIF animation
```

`main_filled.py` imports `visualize_filled`, so run from a directory where both files are visible.

---

## 6. What the solver actually optimizes

The optimizer can only change two things:

1. **Which letter is assigned to each `(room, module)` slot.**
2. **The common visit order over the six modules.**

Everything else is fixed by the instance: the grid size, the catalog, and every student's schedule.

As a consequence:

- Adding a section of a course is not a "decision" — it is just a room whose module entry happens to equal that course's letter.
- Removing a section means changing that room's letter to something else.
- There are no room capacities, teacher assignments, or teacher-conflict constraints in the current model.
- The number of students per class is unbounded.

If a real school needs any of these, they must be added as **hard constraints**; the current cost function will happily route 40 students through a room built for 20.

---

## 7. Known limitations

- **No room capacity.** A course can be placed in exactly one room regardless of class size.
- **No teacher constraints.** Nothing prevents "the same teacher teaches two courses in two rooms at the same time."
- **Feasibility is checked from scratch** in every move of coordinate descent and SA. This is fine for a toy instance and becomes the bottleneck around 20 × 20 grids.
- **The cost is recomputed from scratch per move.** Caching per-student cost and updating only affected students after a single-cell change would be the first optimization for a real instance.
- **Manhattan distance is a proxy.** It ignores corridors, stairs, entrances, and terrain. For a real building, replace `manhattan` with a precomputed shortest-path matrix on the building graph; the DP is unchanged.
- **The category order is shared by all students.** Real schools may permit parallel electives or free periods; the current model forces one common order.

---

## 8. Why gradient-free

The discrete objective is piecewise constant in `placement` and `order`, so gradient descent does not apply directly. A differentiable relaxation (softmax over rooms, Sinkhorn over the order, smoothed Manhattan) was tried in an earlier version and then removed for two reasons:

1. The relaxation requires **rounding and repair**, which reintroduces a discrete stage anyway.
2. On this instance, **coordinate descent + SA from a random fill** matched the relaxed pipeline's final cost within noise, in a fraction of the code.

The current pipeline is the shortest one that consistently reaches a good local minimum on the toy instance.
