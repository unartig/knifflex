from collections import defaultdict

import networkx as nx
import plotly.graph_objects as go

from gunky import Category, DAGGenome
from rules import DecisionRule, RerollRule

# --- Graph construction ---

def _build_graph(genome: DAGGenome) -> nx.DiGraph:
    G = nx.DiGraph()

    for i in range(len(genome.rules)):
        rule_name = DecisionRule(int(genome.rules[i]) + 1 ).name
        thresh = float(genome.thresholds[i])
        G.add_node(f"N{i}", label=f"{rule_name}<br>≥ {thresh:.2f}", kind="node")

        for child, side in [(genome.left[i], "F"), (genome.right[i], "T")]:
            target = _resolve_child(genome, child)
            G.add_edge(f"N{i}", target, label=side)

    for i in range(len(genome.leaf_values)):
        val = int(genome.leaf_values[i])
        if genome.leaf_is_reroll[i]:
            G.add_node(f"R{i}", label=f"↺ {RerollRule(val + 1).name}", kind="reroll")
        else:
            G.add_node(f"C{i}", label=f"✓ {Category(val).name}", kind="score")

    return G


def _resolve_child(genome: DAGGenome, child: int) -> str:
    if child >= 0:
        return f"N{int(child)}"
    leaf_idx = -int(child) - 1
    prefix = "R" if genome.leaf_is_reroll[leaf_idx] else "C"
    return f"{prefix}{leaf_idx}"


# --- Layout ---

def _compute_layout(G: nx.DiGraph) -> dict[str, tuple[float, float]]:
    layers = _assign_layers(G)
    layer_groups: dict[int, list[str]] = defaultdict(list)
    for node, layer in layers.items():
        layer_groups[layer].append(node)

    pos: dict[str, tuple[float, float]] = {}
    for layer, nodes in layer_groups.items():
        n = len(nodes)
        for j, node in enumerate(nodes):
            pos[node] = ((j - (n - 1) / 2) * 2.5, -layer * 2.0)
    return pos


def _assign_layers(G: nx.DiGraph) -> dict[str, int]:
    layers: dict[str, int] = {}
    for node in nx.topological_sort(G):
        layers.setdefault(node, 0)
        for _, nbr in G.out_edges(node):
            layers[nbr] = max(layers.get(nbr, 0), layers[node] + 1)
    return layers


# --- Plotly traces ---

_KIND_STYLES = {
    "node":   dict(color="#FFF9C4", border="#F9A825", symbol="square",  legend="Decision"),
    "reroll": dict(color="#BBDEFB", border="#1565C0", symbol="circle",  legend="Reroll leaf"),
    "score":  dict(color="#C8E6C9", border="#2E7D32", symbol="circle",  legend="Score leaf"),
}


def _edge_traces(G: nx.DiGraph, pos: dict) -> tuple[list[go.Scatter], list[dict]]:
    traces, annotations = [], []
    for u, v, data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        traces.append(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode="lines",
            line={"width": 1.5, "color": "#888"},
            hoverinfo="none",
            showlegend=False,
        ))
        annotations.append({
            "x": (x0 + x1) / 2, "y": (y0 + y1) / 2,
            "text": f"<b>{data['label']}</b>",
            "showarrow": False,
            "font": {"size": 11, "color": "crimson"},
            "bgcolor": "white",
            "borderpad": 2,
        })
    return traces, annotations


def _node_traces(G: nx.DiGraph, pos: dict) -> list[go.Scatter]:
    traces = []
    for kind, style in _KIND_STYLES.items():
        nodes = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        if not nodes:
            continue
        labels = [G.nodes[n]["label"] for n in nodes]
        traces.append(go.Scatter(
            x=[pos[n][0] for n in nodes],
            y=[pos[n][1] for n in nodes],
            mode="markers+text",
            marker={"size": 40, "color": style["color"], "symbol": style["symbol"],
                    "line": {"width": 2, "color": style["border"]}},
            text=labels,
            textfont={"size": 9},
            textposition="middle center",
            hovertext=labels,
            hoverinfo="text",
            name=style["legend"],
        ))
    return traces


# --- Entry point ---

def plot_dag(genome: DAGGenome) -> None:
    G = _build_graph(genome)
    pos = _compute_layout(G)
    layers = _assign_layers(G)

    edge_traces, annotations = _edge_traces(G, pos)
    node_traces = _node_traces(G, pos)

    fig = go.Figure(
        data=edge_traces + node_traces,
        layout=go.Layout(
            title="Kniffel DAGGenome",
            showlegend=True,
            hovermode="closest",
            xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
            annotations=annotations,
            plot_bgcolor="white",
            width=1200,
            height=200 + (max(layers.values()) + 1) * 160,
        ),
    )
    fig.show()
