"""
visualize.py — visualization for the fill-the-grid solver.

Data shape assumed everywhere:
    placement[(row, col)][module] = letter      # every cell filled
    robots   = [{module: letter, ...}, ...]
    order    = permutation of modules

Public API:
    visualize_solution(N, M, catalog, robots, order, placement, ...)
    visualize_comparison(N, M, catalog, robots, order, before, after, ...)
    animate_paths(N, M, catalog, robots, order, placement, filename)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ---------------------------------------------------------------------------
# 0. Constants and small helpers
# ---------------------------------------------------------------------------

CATEGORY_COLORS = {
    1: "#e6194B", 2: "#3cb44b", 3: "#4363d8",
    4: "#f58231", 5: "#911eb4", 6: "#f032e6",
}
ROBOT_CMAP = plt.cm.Dark2

def _robot_colors(n):
    return [ROBOT_CMAP(i % 8) for i in range(n)]

def manhattan(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])

def positions_of(placement, module, letter):
    return [p for p, cells in placement.items() if cells.get(module) == letter]


# ---------------------------------------------------------------------------
# 1. Canonical route per robot (the DP from the solver)
# ---------------------------------------------------------------------------

def robot_path(req, order, placement):
    """Return the sequence of cells chosen by the shortest-path DP."""
    INF = float("inf")
    first = positions_of(placement, order[0], req[order[0]])
    if not first:
        return []
    dp = {p: (0, [p]) for p in first}
    for module in order[1:]:
        nxt = {}
        for q in positions_of(placement, module, req[module]):
            p_best = min(dp, key=lambda p: dp[p][0] + manhattan(p, q))
            nxt[q] = (dp[p_best][0] + manhattan(p_best, q),
                      dp[p_best][1] + [q])
        if not nxt:
            return []
        dp = nxt
    return min(dp.values(), key=lambda t: t[0])[1]

def robot_cost(req, order, placement):
    path = robot_path(req, order, placement)
    return sum(manhattan(path[i], path[i + 1]) for i in range(len(path) - 1))


# ---------------------------------------------------------------------------
# 2. Grid drawing
# ---------------------------------------------------------------------------

def _grid_border(ax, N, M):
    ax.set_xlim(-0.5, M - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks(range(M))
    ax.set_yticks(range(N))
    ax.tick_params(labelsize=7)
    for r in range(N):
        for c in range(M):
            ax.add_patch(patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor="lightgray", lw=0.8))

def _grid_labels(ax, N, M, placement, order):
    """Stack every module's letter vertically inside its cell."""
    mods = list(order)
    n = len(mods)
    dy = min(0.28, 0.9 / max(n, 1))
    for (r, c), cells in placement.items():
        for i, m in enumerate(mods):
            if m not in cells:
                continue
            y_off = (i - (n - 1) / 2.0) * dy
            ax.text(c, r + y_off, cells[m],
                    ha="center", va="center", fontsize=5.5,
                    bbox=dict(boxstyle="round,pad=0.10",
                              facecolor=CATEGORY_COLORS[m],
                              alpha=0.90, edgecolor="none"))


# ---------------------------------------------------------------------------
# 3. Path panel
# ---------------------------------------------------------------------------

def _draw_paths(ax, N, M, robots, order, placement, highlight=None):
    _grid_border(ax, N, M)
    colors = _robot_colors(len(robots))
    for ridx, r in enumerate(robots):
        seq = robot_path(r, order, placement)
        if not seq:
            continue
        rows = [p[0] for p in seq]
        cols = [p[1] for p in seq]
        col = colors[ridx]
        active = (highlight is None or highlight == ridx)
        alpha = 1.0 if active else 0.12
        lw = 2.2 if active else 1.0
        ax.plot(cols, rows, "-", color=col, alpha=alpha, lw=lw, zorder=3)
        ax.scatter(cols, rows, color=col, s=36, alpha=alpha,
                   edgecolor="white", linewidth=0.4, zorder=4,
                   label=f"r{ridx}")
        if active:
            for i, (rr, cc) in enumerate(zip(rows, cols)):
                ax.text(cc, rr, str(i + 1),
                        color="white", fontsize=5.5,
                        ha="center", va="center",
                        fontweight="bold", zorder=5)
    ax.legend(fontsize=6, loc="center left",
              bbox_to_anchor=(1.02, 0.5), framealpha=0.85)


# ---------------------------------------------------------------------------
# 4. Congestion
# ---------------------------------------------------------------------------

def _endpoint_counts(robots, order, placement):
    counts = {}
    for r in robots:
        for p in robot_path(r, order, placement):
            counts[p] = counts.get(p, 0) + 1
    return counts

def _traversal_counts(robots, order, placement, N, M):
    grid = np.zeros((N, M))
    for r in robots:
        path = robot_path(r, order, placement)
        visited = set()
        for i in range(len(path) - 1):
            (x0, y0), (x1, y1) = path[i], path[i + 1]
            sx = 0 if x0 == x1 else (1 if x1 > x0 else -1)
            sy = 0 if y0 == y1 else (1 if y1 > y0 else -1)
            x, y = x0, y0
            visited.add((x, y))
            while (x, y) != (x1, y1):
                if x != x1:   x += sx
                elif y != y1: y += sy
                visited.add((x, y))
        for (x, y) in visited:
            grid[x, y] += 1
    return grid

def _draw_congestion(ax, N, M, robots, order, placement, mode="endpoint"):
    if mode == "endpoint":
        counts = _endpoint_counts(robots, order, placement)
        grid = np.zeros((N, M))
        for (r, c), v in counts.items():
            grid[r, c] = v
        title = "Endpoint congestion\n(stops per room)"
    else:
        grid = _traversal_counts(robots, order, placement, N, M)
        title = "Traversal congestion\n(approx. Manhattan paths)"

    im = ax.imshow(grid, cmap="Reds", alpha=0.9, vmin=0)
    for r in range(N):
        for c in range(M):
            v = int(grid[r, c])
            if v > 0:
                ax.text(c, r, str(v), ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_xticks(range(M))
    ax.set_yticks(range(N))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    return im


# ---------------------------------------------------------------------------
# 5. Per-robot cost bars
# ---------------------------------------------------------------------------

def _draw_costs(ax, robots, order, placement):
    costs = [robot_cost(r, order, placement) for r in robots]
    colors = _robot_colors(len(robots))
    bars = ax.bar(range(len(robots)), costs, color=colors, alpha=0.85)
    ax.axhline(np.mean(costs), color="black", ls="--", lw=1,
               label=f"mean = {np.mean(costs):.2f}")
    ax.set_xlabel("robot index")
    ax.set_ylabel("Manhattan path length")
    ax.set_title("Per-robot path length", fontsize=9)
    ax.legend(fontsize=7)
    for i, b in enumerate(bars):
        ax.text(i, b.get_height() + 0.05, str(int(costs[i])),
                ha="center", fontsize=7)


# ---------------------------------------------------------------------------
# 6. Public entry points
# ---------------------------------------------------------------------------

def visualize_solution(N, M, catalog, robots, order, placement,
                       title="", congestion_mode="endpoint",
                       highlight=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title or "Solution", fontsize=12)

    _grid_border(axes[0, 0], N, M)
    _grid_labels(axes[0, 0], N, M, placement, order)
    axes[0, 0].set_title("Grid  (one letter per module, colour = module)",
                         fontsize=9)

    _draw_paths(axes[0, 1], N, M, robots, order, placement,
                highlight=highlight)
    order_str = " -> ".join(str(m) for m in order)
    axes[0, 1].set_title(f"Robot paths   (order: {order_str})", fontsize=9)

    im = _draw_congestion(axes[1, 0], N, M, robots, order, placement,
                          mode=congestion_mode)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    _draw_costs(axes[1, 1], robots, order, placement)

    plt.tight_layout()
    return fig


def visualize_comparison(N, M, catalog, robots, order,
                         placement_before, placement_after,
                         titles=("Before", "After")):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    order_str = " -> ".join(str(m) for m in order)

    for j, (pl, ttl) in enumerate(
            [(placement_before, titles[0]),
             (placement_after,  titles[1])]):
        _draw_paths(axes[0, j], N, M, robots, order, pl)
        mean = np.mean([robot_cost(r, order, pl) for r in robots])
        axes[0, j].set_title(f"{ttl}: paths (mean = {mean:.2f})", fontsize=9)

        _grid_border(axes[1, j], N, M)
        _grid_labels(axes[1, j], N, M, pl, order)
        axes[1, j].set_title(f"{ttl}: placement", fontsize=9)

    fig.suptitle(f"Before / after   (order: {order_str})", fontsize=12)
    plt.tight_layout()
    return fig


def animate_paths(N, M, catalog, robots, order, placement,
                  filename="paths.gif"):
    fig, ax = plt.subplots(figsize=(7, 7))
    _grid_border(ax, N, M)
    _grid_labels(ax, N, M, placement, order)
    colors = _robot_colors(len(robots))
    paths = [robot_path(r, order, placement) for r in robots]

    lines = [ax.plot([], [], "-o", color=colors[i], alpha=0.9,
                     ms=5, lw=1.5)[0] for i in range(len(robots))]

    def frame(step):
        for ridx, seq in enumerate(paths):
            s = seq[:step + 1]
            if not s:
                continue
            lines[ridx].set_data([p[1] for p in s], [p[0] for p in s])
        ax.set_title(f"Module {order[step]}  (step {step + 1}/{len(order)})")
        return lines

    ani = animation.FuncAnimation(fig, frame, frames=len(order),
                                  interval=900, blit=False, repeat=False)
    ani.save(filename)
    return ani