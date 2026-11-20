"""Small HTML/CSS helpers for a polished Streamlit look (works in light and dark themes)."""

from __future__ import annotations

import html
import re

import streamlit as st

ACCENT = "#0891B2"
ACCENT_DARK = "#164E63"

_CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1240px; }}
.om-hero {{
  border-radius: 22px; padding: 30px 34px; margin-bottom: 1.2rem; color: #fff;
  background:
    radial-gradient(circle at 88% 18%, rgba(255,255,255,0.20), transparent 38%),
    radial-gradient(circle at 10% 110%, rgba(255,255,255,0.10), transparent 40%),
    linear-gradient(135deg, {ACCENT} 0%, {ACCENT_DARK} 100%);
  box-shadow: 0 18px 40px -22px rgba(22,78,99,0.65);
}}
.om-hero .om-title {{ font-size: 2.15rem; font-weight: 750; letter-spacing: -0.02em; line-height: 1.15; }}
.om-hero .om-sub {{ margin-top: 8px; font-size: 1.03rem; opacity: 0.92; max-width: 800px; }}
.om-chip {{
  display: inline-block; margin: 14px 8px 0 0; padding: 4px 12px; border-radius: 999px;
  background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.22); font-size: 0.8rem;
}}
div[data-testid="stMetric"] {{
  border: 1px solid rgba(8,145,178,0.22); border-radius: 16px; padding: 14px 16px;
  background: linear-gradient(180deg, rgba(8,145,178,0.07), rgba(8,145,178,0.015));
}}
div[data-testid="stMetricLabel"] p {{ font-size: 0.82rem; opacity: 0.8; }}
div[data-testid="stMetricValue"] {{ font-size: 1.6rem; font-weight: 650; }}
.om-card {{
  border: 1px solid rgba(148,163,184,0.35); border-radius: 16px; padding: 14px 18px; height: 100%;
}}
.om-card h4 {{ margin: 0 0 6px 0; font-size: 0.95rem; }}
.om-card ul {{ margin: 0; padding-left: 1.1rem; }}
.om-card li {{ margin-bottom: 0.35rem; }}
.om-pill {{
  display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600; font-size: 0.9rem;
  background: rgba(8,145,178,0.12); color: {ACCENT_DARK}; border: 1px solid rgba(8,145,178,0.28);
}}
.om-context {{ font-size: 0.9rem; opacity: 0.75; margin: -0.4rem 0 0.8rem 0; }}
.om-tag {{
  display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.75rem; font-weight: 650;
  margin-left: 6px; vertical-align: middle;
}}
.om-tag-synthetic {{ background: rgba(217,119,6,0.13); color: #B45309; border: 1px solid rgba(217,119,6,0.3); }}
.om-table {{ width: 100%; border-collapse: collapse; font-size: 0.92rem; margin: 0.2rem 0 1rem 0; }}
.om-table th {{
  text-align: left; font-weight: 600; opacity: 0.7; padding: 8px 10px;
  border-bottom: 1px solid rgba(148,163,184,0.45);
}}
.om-table td {{ padding: 9px 10px; border-bottom: 1px solid rgba(148,163,184,0.2); vertical-align: top; }}
.om-table tr:hover td {{ background: rgba(8,145,178,0.035); }}
.om-num {{ font-weight: 650; font-variant-numeric: tabular-nums; white-space: nowrap; }}
.om-muted {{ opacity: 0.55; }}
.om-sev-high, .om-bad {{ color: #DC2626; font-weight: 700; }}
.om-sev-medium, .om-warn {{ color: #D97706; font-weight: 700; }}
.om-sev-low, .om-good {{ color: #059669; font-weight: 700; }}
.om-bar {{
  display: inline-block; width: 64px; height: 8px; border-radius: 99px; overflow: hidden;
  background: rgba(8,145,178,0.13); vertical-align: middle; margin-right: 8px;
}}
.om-bar > div {{ height: 100%; border-radius: 99px; background: {ACCENT}; }}
.om-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 0.3rem 0 1.1rem 0; }}
@media (max-width: 900px) {{ .om-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }} }}
.om-tile {{
  border: 1px solid rgba(148,163,184,0.35); border-radius: 16px; padding: 12px 14px;
  background: rgba(148,163,184,0.04);
}}
.om-tile.on {{ border-color: rgba(8,145,178,0.45); background: linear-gradient(180deg, rgba(8,145,178,0.09),
  rgba(8,145,178,0.02)); }}
.om-tile .t {{ font-weight: 650; font-size: 0.92rem; }}
.om-tile .f {{ margin-top: 4px; font-size: 0.86rem; opacity: 0.85; }}
.om-tile .s {{ margin-top: 6px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
  text-transform: uppercase; }}
.om-tile.on .s {{ color: {ACCENT}; }}
.om-tile.off .s {{ opacity: 0.5; }}
</style>
"""

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"\*(.+?)\*")


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def md_inline(text: str) -> str:
    """Escape text, then render **bold** and *italic* as HTML (for use inside HTML blocks)."""
    escaped = html.escape(text)
    return _ITALIC.sub(r"<i>\1</i>", _BOLD.sub(r"<b>\1</b>", escaped))


def hero(title: str, subtitle: str, chips: list[str]) -> None:
    chip_html = "".join(f'<span class="om-chip">{html.escape(c)}</span>' for c in chips)
    st.markdown(
        f'<div class="om-hero"><div class="om-title">{html.escape(title)}</div>'
        f'<div class="om-sub">{html.escape(subtitle)}</div>{chip_html}</div>',
        unsafe_allow_html=True,
    )


def context_line(text: str, synthetic: bool) -> None:
    tag = '<span class="om-tag om-tag-synthetic">Synthetic data</span>' if synthetic else ""
    st.markdown(f'<div class="om-context">{html.escape(text)}{tag}</div>', unsafe_allow_html=True)


def pill(text: str) -> None:
    st.markdown(f'<span class="om-pill">{md_inline(text)}</span>', unsafe_allow_html=True)


def table(headers: list[str], rows: list[list], html_cols: tuple[int, ...] = ()) -> None:
    """Render a wrapping HTML table. Cells are escaped unless their column is in `html_cols`."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>"
        + "".join(
            f"<td>{cell if i in html_cols else html.escape(str(cell))}</td>" for i, cell in enumerate(row)
        )
        + "</tr>"
        for row in rows
    )
    st.markdown(f'<table class="om-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>',
                unsafe_allow_html=True)


def num_cell(text: str) -> str:
    return f'<span class="om-num">{html.escape(text)}</span>'


def share_bar(value: float, label: str | None = None) -> str:
    width = max(0.0, min(100.0, value * 100))
    text = label if label is not None else f"{value * 100:.1f}%"
    return (f'<span style="white-space:nowrap"><span class="om-bar"><div style="width:{width:.0f}%"></div></span>'
            f'<span class="om-num">{html.escape(text)}</span></span>')


def status(text: str, kind: str) -> str:
    """kind: good, warn or bad."""
    kind = kind if kind in ("good", "warn", "bad") else "warn"
    return f'<span class="om-{kind}">{html.escape(text)}</span>'


def trend_cell(direction: str) -> str:
    kind = {"worsening": "bad", "improving": "good"}.get(direction)
    return status(direction, kind) if kind else f'<span class="om-muted">{html.escape(direction)}</span>'


def severity(level: str) -> str:
    level = level if level in ("low", "medium", "high") else "medium"
    return f'<span class="om-sev-{level}">{level.title()}</span>'


def card(title: str, items: list[str], markdown: bool = False) -> None:
    render = md_inline if markdown else html.escape
    body = "".join(f"<li>{render(i)}</li>" for i in items) or "<li>—</li>"
    st.markdown(f'<div class="om-card"><h4>{html.escape(title)}</h4><ul>{body}</ul></div>', unsafe_allow_html=True)


def waste_grid(tiles: list[tuple[str, str, str, bool]]) -> None:
    """tiles: (icon, waste, headline figure, detected)."""
    cells = "".join(
        f'<div class="om-tile {"on" if on else "off"}"><div class="t">{icon} {html.escape(name)}</div>'
        f'<div class="f">{html.escape(figure)}</div><div class="s">{"Detected" if on else "Not detected"}</div></div>'
        for icon, name, figure, on in tiles
    )
    st.markdown(f'<div class="om-grid">{cells}</div>', unsafe_allow_html=True)
