"""Render the directly-follows graph as Graphviz DOT (displayed with st.graphviz_chart).

Nodes are shaded by the mean waiting time in front of the activity; edge width follows
frequency; the bottleneck gets a red outline. Rare edges can be hidden with ``min_share``.
"""

from __future__ import annotations

import math

import pandas as pd

from .formatting import duration, num
from .mining import END, START

NODE_LIGHT, NODE_DARK = "#ECFEFF", "#0E7490"
EDGE_SHORT, EDGE_LONG = "#5FA8B8", "#B45309"
BOTTLENECK = "#DC2626"


def _mix(a: str, b: str, t: float) -> str:
    t = 0.0 if math.isnan(t) else max(0.0, min(1.0, t))
    ca = [int(a[i : i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i : i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02X}" for x, y in zip(ca, cb, strict=True))


def _q(*lines: str) -> str:
    """A quoted DOT string; several lines are joined with DOT's newline escape."""
    escaped = [str(line).replace("\\", "\\\\").replace('"', '\\"') for line in lines]
    return '"' + "\\n".join(escaped) + '"'


def filter_edges(edges: pd.DataFrame, min_share: float) -> pd.DataFrame:
    """Keep edges followed by at least ``min_share`` of cases (start and end edges included)."""
    return edges[edges["share"] >= min_share]


def to_dot(
    edges: pd.DataFrame,
    activities: pd.DataFrame,
    *,
    min_share: float = 0.0,
    label: str = "frequency",
    bottleneck: str | None = None,
    rankdir: str = "TB",
) -> str:
    kept = filter_edges(edges, min_share)
    stats = activities.set_index("activity")
    nodes = list(dict.fromkeys([*kept["source"], *kept["target"]]))
    max_wait = stats["mean_wait_h"].max(skipna=True) if len(stats) else 0.0
    max_wait = max_wait if pd.notna(max_wait) and max_wait > 0 else 1.0
    inner = kept[kept["mean_wait_h"].notna()]
    max_edge_wait = inner["mean_wait_h"].max() if len(inner) else 1.0
    max_count = kept["count"].max() if len(kept) else 1

    out = [
        "digraph process {",
        f"  rankdir={rankdir}; bgcolor=\"transparent\"; nodesep=0.3; ranksep=0.34; pad=0.15;",
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11, color="#164E63", '
        'penwidth=1.1, margin="0.20,0.07"];',
        '  edge [fontname="Helvetica", fontsize=9, fontcolor="#334155", arrowsize=0.55];',
    ]
    for node in nodes:
        if node == START:
            out.append(f'  {_q(node)} [shape=circle, label="", width=0.22, fillcolor="#0891B2", color="#0891B2"];')
        elif node == END:
            out.append(f'  {_q(node)} [shape=doublecircle, label="", width=0.16, fillcolor="#164E63", '
                       'color="#164E63"];')
        else:
            row = stats.loc[node]
            wait = row["mean_wait_h"]
            t = (wait if pd.notna(wait) else 0.0) / max_wait
            fill = _mix(NODE_LIGHT, NODE_DARK, t)
            font = "#FFFFFF" if t > 0.5 else "#0B2530"
            sub = f"{num(row['events'])} runs" + (f" · wait {duration(wait)}" if pd.notna(wait) else "")
            extra = f', color="{BOTTLENECK}", penwidth=2.8' if node == bottleneck else ""
            out.append(f"  {_q(node)} [label={_q(node, sub)}, fillcolor=\"{fill}\", fontcolor=\"{font}\"{extra}];")
    for _, e in kept.iterrows():
        width = 0.7 + 4.3 * (e["count"] / max_count)
        if label == "wait":
            text = duration(e["mean_wait_h"]) if pd.notna(e["mean_wait_h"]) else ""
            color = _mix(EDGE_SHORT, EDGE_LONG, (e["mean_wait_h"] / max_edge_wait) if pd.notna(e["mean_wait_h"]) else 0)
        else:
            text, color = num(e["count"]), EDGE_SHORT
        out.append(f"  {_q(e['source'])} -> {_q(e['target'])} [label={_q(' ' + text + ' ')}, "
                   f"penwidth={width:.2f}, color=\"{color}\"];")
    out.append("}")
    return "\n".join(out)
