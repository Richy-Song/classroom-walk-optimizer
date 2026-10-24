"""
visualize.py — visualization for the classroom-movement solver.

Public entry points:
    visualize_solution(...)    -> 2x2 figure of one solution
    visualize_comparison(...)  -> 2x2 figure of two placements side by side
    animate_paths(...)         -> MP4 / GIF of robots stepping through the order

Assumes placement: dict  label=(category, identifier) -> (row, col)
         robots:    list of dict  category -> label
         order:     tuple of categories
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

# ---- 1. Consistent styling ------------------------------------------------

CATEGORY_COLORS = {
    1: "#e6194B", 2: "#3cb44b", 3: "#4363d8",
    4: "#f58231", 5: "#911eb4", 6: "#f032e6",
}
ROBOT_CMAP = plt.cm.Dark2

def _robot_colors(n):
    return [ROBOT_CMAP(i % 8) for i in range(n)]


# ---- 2. Grid with labels --------------------------------------------------

def _draw_grid(ax, N, M, placement, order, show_labels=True):
    ax.set_xlim(-0.5, M - 0.5)
    ax.set_ylim(-0.5, N - 0.5)
    ax.set_aspect("equal")
    ax.invert_yaxis()                     # (0,0) at top-left
    ax.set_xticks(range(M)); ax.set_yticks(range(N))
    ax.tick_params(labelsize=7)

    for r in range(N):
        for c in range(M):
            ax.add_patch(patches.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor="lightgray", lw=0.8))

    if not show_labels:
        return

    pos_to_labels = {}
    for l, p in placement.items():
        pos_to_labels.setdefault(p, []).append(l)

    for (r, c), ls in pos_to_labels.items():
        ls_sorted = sorted(ls, key=lambda l: order.index(l[0]))
        n = len(ls_sorted)
        for i, (cat, ident) in enumerate(ls_sorted):
            y_off = (i - (n - 1) / 2.0) * min(0.28, 0.9 / max(n, 1))
            ax.text(c, r + y_off, f"{cat}{ident}",
                    ha="center", va="center", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.12",
                              facecolor=CATEGORY_COLORS[cat],
                              alpha=0.85, edgecolor="none"))


# ---- 3. Robot paths over the grid ----------------------------------------

def _draw_paths(ax, N, M, robots, order, placement, highlight=None):
    _draw_grid(ax, N, M, placement, order, show_labels=False)
    colors = _robot_colors(len(robots))

    for ridx, r in enumerate(robots):
        seq = [placement[r[c]] for c in order]
        rows = [p[0] for p in seq]
        cols = [p[1] for p in seq]
        col = colors[ridx]
        active = (highlight is None or highlight == ridx)
        alpha = 1.0 if active else 0.12
        lw = 2.2 if active else 1.0

        ax.plot(cols, rows, "-", color=col, alpha=alpha, lw=lw, zorder=3)
        ax.scatter(cols, rows, color=col, s=42, alpha=alpha,
                   zorder=4, edgecolor="white", linewidth=0.5,
                   label=f"robot {ridx}")

        if active:
            for i, (rr, cc) in enumerate(zip(rows, cols)):
                ax.text(cc, rr, str(i + 1),
                        color="white", fontsize=6, ha="center", va="center",
                        fontweight="bold", zorder=5)

    ax.legend(fontsize=6, loc="center left",
              bbox_to_anchor=(1.02, 0.5), framealpha=0.85)


# ---- 4. Congestion heatmaps ----------------------------------------------

def _cell_visits(robots, order, placement):
    visits = {}
    for r in robots:
        for c in order:
            p = placement[r[c]]
            visits[p] = visits.get(p, 0) + 1
    return visits

def _cell_traversals(robots, order, placement, N, M):
    """Approximate: one canonical Manhattan path per pair of consecutive stops."""
    grid = np.zeros((N, M))
    for r in robots:
        visited = set()
        seq = [placement[r[c]] for c in order]
        for i in range(len(seq) - 1):
            (x0, y0), (x1, y1) = seq[i], seq[i + 1]
            sx = 0 if x0 == x1 else (1 if x1 > x0 else -1)
            sy = 0 if y0 == y1 else (1 if y1 > y0 else -1)
            x, y = x0, y0
            visited.add((x, y))
            while (x, y) != (x1, y1):
                if x != x1: x += sx
                elif y != y1: y += sy
                visited.add((x, y))
        for (x, y) in visited:
            grid[x, y] += 1
    return grid

def _draw_congestion(ax, N, M, robots, order, placement, mode="endpoint"):
    if mode == "endpoint":
        visits = _cell_visits(robots, order, placement)
        grid = np.zeros((N, M))
        for (r, c), v in visits.items():
            grid[r, c] = v
        title = "Endpoint congestion\n(stops per cell)"
    else:
        grid = _cell_traversals(robots, order, placement, N, M)
        title = "Traversal congestion\n(approx. Manhattan paths)"

    im = ax.imshow(grid, cmap="Reds", alpha=0.9, vmin=0)
    for r in range(N):
        for c in range(M):
            v = int(grid[r, c])
            if v > 0:
                ax.text(c, r, str(v), ha="center", va="center",
                        fontsize=7, color="black")
    ax.set_xticks(range(M)); ax.set_yticks(range(N))
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    return im


# ---- 5. Per-robot cost bars ----------------------------------------------

def _robot_costs(robots, order, placement):
    costs = []
    for r in robots:
        seq = [placement[r[c]] for c in order]
        cost = sum(
            abs(seq[i][0] - seq[i + 1][0]) + abs(seq[i][1] - seq[i + 1][1])
            for i in range(len(seq) - 1)
        )
        costs.append(cost)
    return costs

def _draw_costs(ax, robots, order, placement):
    costs = _robot_costs(robots, order, placement)
    colors = _robot_colors(len(robots))
    bars = ax.bar(range(len(robots)), costs, color=colors, alpha=0.85)
    ax.axhline(np.mean(costs), color="black", ls="--", lw=1,
               label=f"mean = {np.mean(costs):.2f}")
    ax.set_xlabel("robot index")
    ax.set_ylabel("Manhattan path length")
    ax.set_title("Per-robot path length", fontsize=9)
    ax.legend(fontsize=7)
    for i, b in enumerate(bars):
        ax.text(i, b.get_height() + 0.05, str(costs[i]),
                ha="center", fontsize=7)


# ---- 6. Public entry points ----------------------------------------------

def visualize_solution(N, M, labels, robots, order, placement,
                       title="", congestion_mode="endpoint",
                       highlight=None):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title or "Solution", fontsize=12)

    _draw_grid(axes[0, 0], N, M, placement, order, show_labels=True)
    axes[0, 0].set_title("Grid + label placement", fontsize=9)

    order_str = " → ".join(str(c) for c in order)
    _draw_paths(axes[0, 1], N, M, robots, order, placement,
                highlight=highlight)
    axes[0, 1].set_title(f"Robot paths   (order: {order_str})", fontsize=9)

    im = _draw_congestion(axes[1, 0], N, M, robots, order, placement,
                          mode=congestion_mode)
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    _draw_costs(axes[1, 1], robots, order, placement)

    plt.tight_layout()
    return fig


def visualize_comparison(N, M, labels, robots, order,
                         placement_before, placement_after,
                         titles=("Before", "After")):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    order_str = " → ".join(str(c) for c in order)

    for j, (placement, ttl) in enumerate(
            [(placement_before, titles[0]),
             (placement_after,  titles[1])]):
        _draw_paths(axes[0, j], N, M, robots, order, placement)
        costs = _robot_costs(robots, order, placement)
        axes[0, j].set_title(
            f"{ttl}: paths (mean = {np.mean(costs):.2f})", fontsize=9)

        _draw_grid(axes[1, j], N, M, placement, order, show_labels=True)
        axes[1, j].set_title(f"{ttl}: placement", fontsize=9)

    fig.suptitle(f"Before / after comparison   (order: {order_str})",
                 fontsize=12)
    plt.tight_layout()
    return fig


# ---- 7. Optional: animation of the category order ------------------------

def animate_paths(N, M, robots, order, placement, filename="paths.gif"):
    """Animates the six category steps: one frame per step."""
    fig, ax = plt.subplots(figsize=(7, 7))
    _draw_grid(ax, N, M, placement, order, show_labels=True)
    colors = _robot_colors(len(robots))

    lines = [ax.plot([], [], "-o", color=colors[i], alpha=0.85,
                     ms=5, lw=1.5)[0] for i in range(len(robots))]

    def frame(step):
        for ridx, r in enumerate(robots):
            seq = [placement[r[c]] for c in order][:step + 1]
            rows = [p[0] for p in seq]; cols = [p[1] for p in seq]
            lines[ridx].set_data(cols, rows)
        ax.set_title(f"Category order step {step + 1}: "
                     f"category {order[step]}")
        return lines

    ani = animation.FuncAnimation(fig, frame, frames=len(order),
                                  interval=900, blit=False, repeat=False)
    ani.save(filename)
    return ani