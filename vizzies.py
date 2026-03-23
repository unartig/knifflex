
from __future__ import annotations

from collections import defaultdict
from typing import TypedDict

import networkx as nx
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from genome import LEAF_STRATEGY_NAMES, N_LEAVES, DAGGenome
from rule_primitives import N_STATE_PRIMITIVES, BinaryOp, StatePrimitive

# ------------------------------------------------------------------ #
#  Constants / colour palette                                         #
# ------------------------------------------------------------------ #

# Soft dark-mode-friendly palette
_C = {
    "root_fill": "#E65100",
    "root_border": "#BF360C",
    "active_fill": "#FFF8E1",
    "active_border": "#F9A825",
    "dead_fill": "#FAFAFA",
    "dead_border": "#CFD8DC",
    "leaf_fill": "#E8F5E9",
    "leaf_border": "#2E7D32",
    "leaf_dead_fill": "#F5F5F5",
    "leaf_dead_border": "#B0BEC5",
    "edge_active": "#546E7A",
    "edge_dead": "#ECEFF1",
    "true_label": "#C62828",
    "false_label": "#1565C0",
    "bg": "#FFFFFF",
}

# ------------------------------------------------------------------ #
#  Label helpers                                                      #
# ------------------------------------------------------------------ #

# Bug fix: GT was missing
_BINARY_OP_SYM: dict[BinaryOp, str] = {
    BinaryOp.ADD: "½(a+b)",
    BinaryOp.SUB: "a-b+.5",
    BinaryOp.MUL: "a·b",
    BinaryOp.MAX: "max",
    BinaryOp.MIN: "min",
    BinaryOp.GT: "a>b",
}


def _prim_name(idx: int) -> str:
    try:
        return StatePrimitive(idx).name
    except ValueError:
        return f"P{idx}"


def _op_sym(idx: int) -> str:
    try:
        return _BINARY_OP_SYM[BinaryOp(idx)]
    except (ValueError, KeyError):
        return f"op{idx}"


def _node_label(genome: DAGGenome, i: int) -> str:
    """Improved label: signal names on separate lines, op in the middle."""
    rl = _prim_name(int(genome.rules_left[i]))
    rr = _prim_name(int(genome.rules_right[i]))
    op = _op_sym(int(genome.binary_ops[i]))
    thr = float(genome.thresholds[i])
    return (
        f"<b>{rl}</b>"
        f"<br><span style='color:#607D8B'>{op}</span>"
        f"<br><b>{rr}</b>"
        f"<br><span style='font-size:9px'>≥ {thr:.3f}</span>"
    )


def _leaf_label(strategy_id: int) -> str:
    """Bug fix: use LEAF_STRATEGY_NAMES instead of the removed leaf_score_cat field."""
    name = LEAF_STRATEGY_NAMES[strategy_id] if strategy_id < N_LEAVES else f"strat{strategy_id}"
    return f"<b>{name}</b><br><i style='font-size:9px'>oracle</i>"


def _child_node_id(child: int) -> str:
    return f"N{child}" if child >= 0 else f"L{-child - 1}"


# ------------------------------------------------------------------ #
#  Reachability — BFS over forward edges only                        #
# ------------------------------------------------------------------ #


def _compute_reachability(genome: DAGGenome) -> tuple[set[int], set[int]]:
    """
    BFS from node 0, forward edges only (child > parent).
    Returns (reachable_node_indices, reachable_leaf_strategy_ids).
    """
    n = int(genome.rules_left.shape[0])
    left = [int(genome.left[i]) for i in range(n)]
    right = [int(genome.right[i]) for i in range(n)]

    reachable_nodes: set[int] = set()
    reachable_leaves: set[int] = set()
    queue = [0]

    while queue:
        i = queue.pop()
        if i in reachable_nodes:
            continue
        reachable_nodes.add(i)
        for child in (left[i], right[i]):
            if child < 0:
                reachable_leaves.add(-child - 1)
            elif child > i:
                queue.append(child)

    return reachable_nodes, reachable_leaves


# ------------------------------------------------------------------ #
#  Graph construction                                                 #
# ------------------------------------------------------------------ #


def _build_graph(
    genome: DAGGenome,
    reachable_nodes: set[int],
    reachable_leaves: set[int],
) -> nx.DiGraph:
    G = nx.DiGraph()
    n = int(genome.rules_left.shape[0])

    for i in range(n):
        active = i in reachable_nodes
        G.add_node(
            f"N{i}",
            label=_node_label(genome, i),
            kind="root" if i == 0 else "node",
            active=active,
        )
        for child, side in [(int(genome.left[i]), "F"), (int(genome.right[i]), "T")]:
            if 0 <= child <= i:  # skip back-edges
                continue
            G.add_edge(f"N{i}", _child_node_id(child), label=side, active=active)

    # Bug fix: iterate over strategy IDs 0..N_LEAVES-1 (not 0..12)
    for leaf_idx in range(N_LEAVES):
        G.add_node(
            f"L{leaf_idx}",
            label=_leaf_label(leaf_idx),
            kind="leaf",
            active=leaf_idx in reachable_leaves,
        )

    return G


# ------------------------------------------------------------------ #
#  Layout                                                             #
# ------------------------------------------------------------------ #


def _assign_layers(G: nx.DiGraph) -> dict[str, int]:
    layers: dict[str, int] = {}
    for node in nx.topological_sort(G):
        layers.setdefault(node, 0)
        for _, nbr in G.out_edges(node):
            layers[nbr] = max(layers.get(nbr, 0), layers[node] + 1)
    return layers


def _compute_layout(G: nx.DiGraph) -> dict[str, tuple[float, float]]:
    layers = _assign_layers(G)
    layer_groups: dict[int, list[str]] = defaultdict(list)
    for node, layer in layers.items():
        layer_groups[layer].append(node)

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in layer_groups.items():
        count = len(nodes)
        # Wider horizontal spread (4.5 units) and taller layer gaps (3.2 units)
        for j, node in enumerate(sorted(nodes)):
            pos[node] = ((j - (count - 1) / 2) * 4.5, -layer * 3.2)
    return pos


# ------------------------------------------------------------------ #
#  Styles                                                             #
# ------------------------------------------------------------------ #

# (fill, border, symbol, opacity)
_STYLE: dict[tuple[str, bool], tuple[str, str, str, float]] = {
    ("root", True): (_C["root_fill"], _C["root_border"], "square", 1.0),
    ("node", True): (_C["active_fill"], _C["active_border"], "square", 1.0),
    ("node", False): (_C["dead_fill"], _C["dead_border"], "square", 0.35),
    ("leaf", True): (_C["leaf_fill"], _C["leaf_border"], "circle", 1.0),
    ("leaf", False): (_C["leaf_dead_fill"], _C["leaf_dead_border"], "circle", 0.35),
}

_LEGEND_LABELS: dict[tuple[str, bool], str] = {
    ("root", True): "Root node",
    ("node", True): "Active decision",
    ("node", False): "Inactive decision",
    ("leaf", True): "Reachable leaf",
    ("leaf", False): "Unreachable leaf",
}


# ------------------------------------------------------------------ #
#  Traces                                                             #
# ------------------------------------------------------------------ #


def _edge_annotations(G: nx.DiGraph, pos: dict) -> list[dict]:
    """Arrow annotations with True/False branch labels in distinct colours."""
    annotations = []
    for u, v, data in G.edges(data=True):
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        active = data.get("active", True)
        color = _C["edge_active"] if active else _C["edge_dead"]
        lw = 1.8 if active else 0.7
        side = data["label"]  # "T" or "F"

        annotations.append(
            {
                "x": x1,
                "y": y1,
                "ax": x0,
                "ay": y0,
                "xref": "x",
                "yref": "y",
                "axref": "x",
                "ayref": "y",
                "showarrow": True,
                "arrowhead": 3,
                "arrowsize": 1.1,
                "arrowwidth": lw,
                "arrowcolor": color,
            }
        )
        label_color = (
            _C["true_label"]
            if (active and side == "T")
            else _C["false_label"]
            if (active and side == "F")
            else "#CFD8DC"
        )
        annotations.append(
            {
                "x": (x0 + x1) / 2,
                "y": (y0 + y1) / 2,
                "text": f"<b>{side}</b>",
                "showarrow": False,
                "font": {"size": 11, "color": label_color, "family": "monospace"},
                "bgcolor": "white",
                "borderpad": 3,
            }
        )
    return annotations


def _node_traces(G: nx.DiGraph, pos: dict) -> list[go.Scatter]:
    traces = []
    legend_seen: set[str] = set()

    for (kind, active), (fill, border, symbol, opacity) in _STYLE.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == kind and d.get("active") == active and n in pos]
        if not nodes:
            continue

        legend_key = _LEGEND_LABELS[(kind, active)]
        show = legend_key not in legend_seen
        legend_seen.add(legend_key)

        labels = [G.nodes[n]["label"] for n in nodes]
        text_color = "white" if kind == "root" else "#212121"
        # Slightly larger marker for root, normal for others
        msize = 60 if kind == "root" else 52

        traces.append(
            go.Scatter(
                x=[pos[n][0] for n in nodes],
                y=[pos[n][1] for n in nodes],
                mode="markers+text",
                marker={
                    "size": msize,
                    "color": fill,
                    "symbol": symbol,
                    "line": {"width": 2.5, "color": border},
                    "opacity": opacity,
                },
                text=labels,
                textfont={"size": 9, "color": text_color, "family": "Arial"},
                textposition="middle center",
                hovertext=labels,
                hoverinfo="text",
                name=legend_key,
                showlegend=show,
            )
        )
    return traces


# ------------------------------------------------------------------ #
#  Figure builder                                                     #
# ------------------------------------------------------------------ #


def _build_figure(genome: DAGGenome, title: str) -> go.Figure:
    reachable_nodes, reachable_leaves = _compute_reachability(genome)
    G = _build_graph(genome, reachable_nodes, reachable_leaves)
    pos = _compute_layout(G)
    layers = _assign_layers(G)

    annotations = _edge_annotations(G, pos)
    node_traces = _node_traces(G, pos)

    max_layer = max(layers.values()) if layers else 0
    # Wider canvas + more vertical room per layer
    width = max(1500, 300 + max(len(g) for g in defaultdict(list, {l: [] for l in layers.values()}).values()) * 200)
    height = 260 + (max_layer + 1) * 210

    return go.Figure(
        data=node_traces,
        layout=go.Layout(
            title={"text": title, "font": {"size": 16, "family": "Arial"}, "x": 0.5},
            showlegend=True,
            legend={"orientation": "h", "y": -0.04},
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            annotations=annotations,
            plot_bgcolor=_C["bg"],
            paper_bgcolor=_C["bg"],
            width=1500,
            height=height,
            margin={"l": 20, "r": 20, "t": 60, "b": 60},
        ),
    )


# ------------------------------------------------------------------ #
#  Public API — DAG plots                                             #
# ------------------------------------------------------------------ #


def plot_dag(genome: DAGGenome, title: str = "Kniffel DAGGenome") -> None:
    """Visualise interactively. Call jax.device_get() first."""
    _build_figure(genome, title).show()


def dag_to_image(genome: DAGGenome, title: str = "Kniffel DAGGenome") -> np.ndarray:
    """
    Render to uint8 RGB array (H, W, 3) for TensorBoard.
    Transpose to (C, H, W) before writer.add_image().
    Requires: pip install kaleido pillow
    """
    import io

    from PIL import Image

    buf = io.BytesIO(_build_figure(genome, title).to_image(format="png"))
    return np.array(Image.open(buf).convert("RGB"), dtype=np.uint8)


def plot_best(pop, fitnesses, island_size: int, title: str = "Best Individual") -> None:
    """Pick the globally best genome and plot it interactively."""
    import jax.tree_util as jtu

    flat = fitnesses.reshape(-1)
    idx = int(np.argmax(flat))
    island = idx // island_size
    local = idx % island_size
    best = jtu.tree_map(lambda x: x[island, local], pop)
    plot_dag(best, title=f"{title}  (island {island}, slot {local}, fit={flat[idx]:.1f})")


# ------------------------------------------------------------------ #
#  New: Fitness history chart                                         #
# ------------------------------------------------------------------ #


class FitnessRecord(TypedDict):
    epoch: int
    avg: float
    best: float
    wc: float  # wildcard best


def plot_fitness_history(
    history: list[FitnessRecord],
    title: str = "Fitness over Training",
) -> None:
    """
    Line chart of average, best, and wildcard-best fitness per epoch.

    Parameters
    ----------
    history : list of FitnessRecord dicts, one per epoch.
              Each must have keys: epoch, avg, best, wc.

    Example — build history inside your training loop::

        history: list[FitnessRecord] = []
        for epoch in range(EPOCHS):
            ...
            history.append({"epoch": epoch, "avg": avg_fit,
                             "best": max_fit, "wc": wc_best_fit})
        plot_fitness_history(history)
    """
    epochs = [r["epoch"] for r in history]
    avg = [r["avg"] for r in history]
    best = [r["best"] for r in history]
    wc = [r["wc"] for r in history]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=avg,
            mode="lines",
            name="Island avg",
            line={"color": "#78909C", "width": 1.5, "dash": "dot"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=wc,
            mode="lines",
            name="Wildcard best",
            line={"color": "#AB47BC", "width": 1.8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=best,
            mode="lines",
            name="Global best",
            line={"color": "#2E7D32", "width": 2.5},
        )
    )

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 15}},
        xaxis={"title": "Epoch", "gridcolor": "#ECEFF1"},
        yaxis={"title": "Mean reward (episodes)", "gridcolor": "#ECEFF1"},
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        legend={"orientation": "h", "y": -0.15},
        hovermode="x unified",
        width=900,
        height=420,
        margin={"l": 60, "r": 20, "t": 60, "b": 80},
    )
    fig.show()


# ------------------------------------------------------------------ #
#  New: Signal usage heatmap                                          #
# ------------------------------------------------------------------ #


def plot_signal_heatmap(
    genome: DAGGenome,
    title: str = "Signal Usage in Active Nodes",
    active_only: bool = True,
) -> None:
    """
    Horizontal bar chart showing how many times each of the 36 state
    signals is referenced by rules_left or rules_right across active nodes.

    Parameters
    ----------
    genome      : a CPU-side DAGGenome (call jax.device_get first).
    active_only : if True (default), count only reachable nodes.
    """
    reachable_nodes, _ = _compute_reachability(genome)
    n = int(genome.rules_left.shape[0])

    counts = np.zeros(N_STATE_PRIMITIVES, dtype=np.int32)
    for i in range(n):
        if active_only and i not in reachable_nodes:
            continue
        counts[int(genome.rules_left[i])] += 1
        counts[int(genome.rules_right[i])] += 1

    signal_names = [_prim_name(i) for i in range(N_STATE_PRIMITIVES)]
    order = np.argsort(counts)  # ascending so most-used appears at top

    # Colour by signal group
    group_colors = []
    for i in range(N_STATE_PRIMITIVES):
        if i < 10:
            group_colors.append("#5C6BC0")  # scalar features — indigo
        elif i < 23:
            group_colors.append("#26A69A")  # SCORE_CAT — teal
        else:
            group_colors.append("#EF6C00")  # EV_CAT — orange

    fig = go.Figure(
        go.Bar(
            x=counts[order],
            y=[signal_names[i] for i in order],
            orientation="h",
            marker_color=[group_colors[i] for i in order],
            hovertemplate="%{y}: %{x} references<extra></extra>",
        )
    )

    # Group legend annotations
    for color, label in [
        ("#5C6BC0", "Scalar features (0–9)"),
        ("#26A69A", "SCORE_CAT (10–22)"),
        ("#EF6C00", "EV_CAT (23–35)"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": "square"},
                name=label,
                showlegend=True,
            )
        )

    suffix = "active nodes only" if active_only else "all nodes"
    fig.update_layout(
        title={"text": f"{title}  ({suffix})", "x": 0.5, "font": {"size": 15}},
        xaxis={"title": "# references", "gridcolor": "#ECEFF1"},
        yaxis={"title": "", "tickfont": {"size": 10, "family": "monospace"}},
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        legend={"orientation": "h", "y": -0.12},
        height=max(400, N_STATE_PRIMITIVES * 18 + 140),
        width=700,
        margin={"l": 160, "r": 20, "t": 60, "b": 80},
    )
    fig.show()


# ------------------------------------------------------------------ #
#  New: Strategy leaf frequency                                       #
# ------------------------------------------------------------------ #


def plot_leaf_frequency(
    genome: DAGGenome,
    title: str = "Strategy Leaf Reachability",
) -> None:
    """
    Bar chart showing which strategy leaves are reachable from the root
    and how many distinct parent nodes point to each one.

    Parameters
    ----------
    genome : a CPU-side DAGGenome (call jax.device_get first).
    """
    n = int(genome.rules_left.shape[0])
    left = [int(genome.left[i]) for i in range(n)]
    right = [int(genome.right[i]) for i in range(n)]

    reachable_nodes, reachable_leaves = _compute_reachability(genome)

    # Count how many active nodes point to each leaf
    leaf_counts = np.zeros(N_LEAVES, dtype=np.int32)
    for i in reachable_nodes:
        for child in (left[i], right[i]):
            if child < 0:
                leaf_counts[-child - 1] += 1

    colors = ["#2E7D32" if lid in reachable_leaves else "#B0BEC5" for lid in range(N_LEAVES)]

    fig = go.Figure(
        go.Bar(
            x=LEAF_STRATEGY_NAMES,
            y=leaf_counts,
            marker_color=colors,
            hovertemplate="%{x}<br>Incoming paths: %{y}<extra></extra>",
            text=leaf_counts,
            textposition="outside",
        )
    )

    # Dummy traces for legend
    for color, label in [
        ("#2E7D32", "Reachable"),
        ("#B0BEC5", "Unreachable"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 10, "color": color, "symbol": "square"},
                name=label,
                showlegend=True,
            )
        )

    fig.update_layout(
        title={"text": title, "x": 0.5, "font": {"size": 15}},
        xaxis={"title": "Strategy", "tickangle": -30, "tickfont": {"size": 10, "family": "monospace"}},
        yaxis={"title": "Incoming paths from active nodes", "gridcolor": "#ECEFF1"},
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="#FFFFFF",
        showlegend=True,
        legend={"orientation": "h", "y": -0.28},
        width=750,
        height=420,
        margin={"l": 60, "r": 20, "t": 60, "b": 120},
    )
    fig.show()
