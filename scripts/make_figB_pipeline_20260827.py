"""Render Fig B, the as-shipped pipeline flowchart, from the node/edge list.

CC dispatch 2026-08-27, item 4a. Two things the 2026-08-25 check asked for:

  1. **A feature-engineering node.** Feature engineering appeared nowhere on the
     node/edge list except inside node R1's note, while the deck criterion names
     it explicitly alongside model architecture and uncertainty quantification.
     Node **F1** is added, carrying the ten direction-arm features by name.
     See `docs/figB_pipeline_node_edge_list.md`, anchor
     `feature-engineering-node-20260827`.

  2. **The PNG in the repo.** `figB_pipeline_20260729.png` existed only in
     project knowledge, never in the repository, so the figure had no generation
     provenance at all. Rather than re-committing an opaque binary, this script
     regenerates the figure from the node/edge list, so the PNG now has a named,
     committed producer like every other artifact (contract §4.2).

**Consistency gate.** Every node id drawn here is asserted to appear in
`docs/figB_pipeline_node_edge_list.md` before anything is rendered. If the list
and the figure ever drift, this script fails rather than drawing a figure that
disagrees with its own source of truth.

**Layout is deliberate, not automatic.** Rows are the forecast arms, columns are
pipeline stages. The d+7 arm is WRAPPED onto two lines rather than spanning the
full width: that is what frees the diagonal corridor the d+1 arm needs to reach
assembly without an edge passing through a box. No graph-layout library is used,
so the figure is reproducible without a Graphviz install.

Deterministic: fixed coordinates, fixed text, no stochastic step, so no seed
applies. Rendered at 150 dpi to match the other report figures
(`explainability/shap_summary_*.png`, `bidding_sim/results_2019/*.png`).

Reads : docs/figB_pipeline_node_edge_list.md  (consistency gate only)
Writes: docs/figures/figB_pipeline_20260827.png
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

REPO_ROOT = Path(__file__).resolve().parents[1]
NODE_LIST = REPO_ROOT / "docs" / "figB_pipeline_node_edge_list.md"
OUT_PNG = REPO_ROOT / "docs" / "figures" / "figB_pipeline_20260827.png"

FIG_W_IN, FIG_H_IN, DPI = 16.0, 9.0, 150

# ── Palette ─────────────────────────────────────────────────────────────
C_DATA = "#dfe3e8"
C_FEAT = "#ffd9a0"      # the added node, deliberately the warmest colour
C_SPEED = "#cfe2f3"
C_DIR = "#d6ead3"
C_ASM = "#e2d6ee"
C_EDGE = "#4a4a4a"
C_TEXT = "#141414"

# ── Nodes: id -> (x, y, w, h, colour, title, subtitle) ──────────────────
# Coordinate space is 0-100 in both axes.
NODES: dict[str, tuple] = {
    # Data sources (left column). Ordered so each source sits beside the arm it
    # feeds, which is what keeps every data edge short and crossing-free.
    "D1":  (8, 90, 11.0, 6.4, C_DATA, "D1  ERA5 init",
            "WeatherBench2 zarr\n0.25 deg global"),
    "D4":  (8, 78, 11.0, 6.4, C_DATA, "D4  Pangu weights",
            "pangu_weather_24.onnx\nfrozen, SHA-verified"),
    "D2":  (8, 66, 11.0, 6.4, C_DATA, "D2  HRES forecasts",
            "coarse 45x57\n= 2,565 cells"),
    "D3":  (8, 54, 11.0, 6.4, C_DATA, "D3  AROME truth",
            "fine 43,715 cells\n2016-2020, 125 m"),

    # SPEED d+1 (kit path, untouched)
    "S1a": (26, 93, 9.6, 6.6, C_SPEED, "S1a  Kit MOS centre", "LightGBM u/v\nFROZEN"),
    "S1b": (37, 93, 9.6, 6.6, C_SPEED, "S1b  Quantile MOS", "width as ratios\nFROZEN"),
    "S1c": (48, 93, 9.6, 6.6, C_SPEED, "S1c  Conformal", "symmetric margin\nFROZEN"),
    "S1d": (59, 93, 9.6, 6.6, C_SPEED, "S1d  Downscaler", "coarse -> fine\nFRESH each run"),

    # SPEED d+7 (Pangu replacement, MOS bypassed) - wrapped onto two lines
    "S7a": (26, 80, 9.6, 6.6, C_SPEED, "S7a  Pangu-24", "7x24 h chain\n4 inits/day"),
    "S7b": (37, 80, 9.6, 6.6, C_FEAT,  "S7b  Extract + couple", "u/v at level ->\ncoarse grid"),
    "S7c": (48, 80, 9.6, 6.6, C_SPEED, "S7c  Block-excl dwn", "2020 minus eval\nblock, FROZEN"),
    "S7d": (59, 80, 9.6, 6.6, C_SPEED, "S7d  Direct centre", "no MOS at all\ncentre = Pangu ws"),
    "S7e": (26, 69, 9.6, 6.6, C_SPEED, "S7e  Fixed ratio", "0.55/1.60 x k=1.846\nUNCERTAINTY"),
    "S7f": (37, 69, 9.6, 6.6, C_SPEED, "S7f  Bias correct", "per cell x season\nshrunk k=30"),
    "S7g": (48, 69, 9.6, 6.6, C_SPEED, "S7g  f=1.25 width", "widths only,\nq50 untouched"),
    "S7h": (59, 69, 9.6, 6.6, C_SPEED, "S7h  Splice", "1,398,880 of\n4,196,640 rows"),

    # SPEED d+14
    "S14a": (26, 57, 9.6, 6.6, C_SPEED, "S14a  Climatology",
             "q05/q50/q95 per cell\nx season x hour, no ML"),

    # FEATURE ENGINEERING (the node this revision adds)
    "F1":  (30, 38, 19.0, 14.0, C_FEAT, "F1  FEATURE ENGINEERING", ""),

    # DIRECTION arms
    "R1":  (52, 45, 9.6, 6.6, C_DIR, "R1  Dir residual d1", "LightGBM quantile\nMAE 15.16 deg"),
    "R7":  (52, 33, 9.6, 6.6, C_DIR, "R7  Dir residual d7", "same arch, separate\nMAE 55.87 deg"),
    "R14": (52, 18, 9.6, 6.6, C_DIR, "R14  Dir d14", "kit constant floor\n+/- 139.3 deg"),

    # ASSEMBLY
    "A1":  (82, 56, 12.0, 6.6, C_ASM, "A1  Assemble CSV", "4,196,640 rows\n6 scored columns"),
    "A2":  (82, 44, 12.0, 6.6, C_ASM, "A2  360.000 wrap fix", "kit bug, reported\nto organisers"),
    "A3":  (82, 32, 12.0, 6.6, C_ASM, "A3  Validation gate", "rows, format,\ndir in [0,360)"),
    "A4":  (82, 20, 12.0, 6.6, C_ASM, "A4  Archive + SHA", "Matteo uploads;\nCC never submits"),
}

# The ten direction-arm features, rendered inside F1.
F1_FEATURES = [
    ("circular", "hres_dir_sin, hres_dir_cos"),
    ("magnitude", "hres_speed"),
    ("diurnal", "hour_sin, hour_cos"),
    ("seasonal", "season_code, month_sin, month_cos"),
    ("spatial", "lat, lon"),
]

EDGES = [
    ("D2", "S1a"), ("D1", "S7a"), ("D4", "S7a"), ("D3", "S14a"),
    ("D2", "F1"), ("D3", "F1"),
    ("S1a", "S1b"), ("S1b", "S1c"), ("S1c", "S1d"),
    ("S7a", "S7b"), ("S7b", "S7c"), ("S7c", "S7d"),
    ("S7d", "S7e"),                      # the wrap, right-to-left
    ("S7e", "S7f"), ("S7f", "S7g"), ("S7g", "S7h"),
    ("F1", "R1"), ("F1", "R7"),
    ("S1d", "A1"), ("S7h", "A1"), ("S14a", "A1"),
    ("R1", "A1"), ("R7", "A1"), ("R14", "A1"),
    ("A1", "A2"), ("A2", "A3"), ("A3", "A4"),
]

# Explicit anchor sides / curvature where the automatic choice reads badly.
EDGE_STYLE = {
    ("D2", "S1a"):  ("r", "l", 0.0),
    ("S7d", "S7e"): ("b", "t", 0.0),     # the wrap arrow, through the row gap
    ("S1d", "A1"):  ("r", "l", 0.0),
    ("S14a", "A1"): ("r", "l", -0.06),
    ("R14", "A1"):  ("r", "l", -0.10),
}

ROW_LABELS = [
    (16.5, 98.5, "SPEED  d+1", "kit path, untouched"),
    (16.5, 87.0, "SPEED  d+7", "Pangu replacement, MOS bypassed  (wraps below)"),
    (16.5, 62.5, "SPEED  d+14", "climatological replacement"),
    (16.5, 48.5, "DIRECTION  d+1 / d+7", "unwrapped residual"),
    (16.5, 23.0, "DIRECTION  d+14", "kit passthrough"),
]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_against_node_list() -> None:
    """Every drawn node id must appear in the node/edge list markdown."""
    if not NODE_LIST.exists():
        raise SystemExit(f"node/edge list not found: {NODE_LIST}")
    text = NODE_LIST.read_text(encoding="utf-8")
    missing = [nid for nid in NODES if f"| {nid} |" not in text]
    if missing:
        raise SystemExit(
            f"node id(s) {missing} are drawn but absent from "
            f"{NODE_LIST.relative_to(REPO_ROOT)}. The figure and its source of "
            f"truth have drifted; refusing to render."
        )
    print(f"consistency gate: all {len(NODES)} node ids present in "
          f"{NODE_LIST.name}")


def anchor(nid: str, side: str) -> tuple[float, float]:
    x, y, w, h, *_ = NODES[nid]
    if side == "r":
        return x + w / 2, y
    if side == "l":
        return x - w / 2, y
    if side == "t":
        return x, y + h / 2
    return x, y - h / 2


def pick_sides(a: str, b: str) -> tuple[str, str, float]:
    if (a, b) in EDGE_STYLE:
        return EDGE_STYLE[(a, b)]
    ax_, ay = NODES[a][0], NODES[a][1]
    bx, by = NODES[b][0], NODES[b][1]
    if abs(bx - ax_) >= abs(by - ay):
        sides = ("r", "l") if bx > ax_ else ("l", "r")
    else:
        sides = ("t", "b") if by > ay else ("b", "t")
    rad = 0.10 if (sides[0] in ("r", "l") and abs(ay - by) > 3) else 0.0
    return sides[0], sides[1], rad


def main() -> None:
    gate_against_node_list()

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    ax.set_xlim(0, 100)
    ax.set_ylim(11, 102)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Row band labels
    for x, y, title, sub in ROW_LABELS:
        ax.text(x, y + 1.4, title, fontsize=8.5, fontweight="bold",
                color="#5a5a5a", ha="left", va="center")
        ax.text(x, y - 1.0, sub, fontsize=6.6, color="#8a8a8a",
                ha="left", va="center", style="italic")

    # Edges first, so boxes paint over the arrow tails
    for a, b in EDGES:
        sa, sb, rad = pick_sides(a, b)
        p0, p1 = anchor(a, sa), anchor(b, sb)
        ax.add_patch(FancyArrowPatch(
            p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle="-|>",
            mutation_scale=9, linewidth=0.9, color=C_EDGE, alpha=0.75,
            shrinkA=1.5, shrinkB=2.5, zorder=1))

    # Nodes
    for nid, (x, y, w, h, colour, title, sub) in NODES.items():
        ax.add_patch(FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.28,rounding_size=0.9",
            facecolor=colour, edgecolor="#3d3d3d",
            linewidth=1.6 if nid == "F1" else 0.8, zorder=2))
        if nid == "F1":
            ax.text(x, y + h / 2 - 1.6, title, fontsize=8.2,
                    fontweight="bold", ha="center", va="center",
                    color=C_TEXT, zorder=3)
            ax.text(x, y + h / 2 - 3.3,
                    "10 features - HRES + static only, no AROME term",
                    fontsize=5.9, ha="center", va="center",
                    color="#5a4a2a", style="italic", zorder=3)
            yy = y + h / 2 - 5.3   # first feature row
            for kind, names in F1_FEATURES:
                ax.text(x - w / 2 + 1.1, yy, kind, fontsize=5.8,
                        ha="left", va="center", color="#7a4f0a",
                        fontweight="bold", zorder=3)
                ax.text(x - w / 2 + 6.0, yy, names, fontsize=5.8,
                        ha="left", va="center", color=C_TEXT,
                        family="monospace", zorder=3)
                yy -= 1.50
            ax.text(x, y - h / 2 + 1.2,
                    "dir_residual_train_models.py:61",
                    fontsize=5.3, ha="center", va="center",
                    color="#8a7a5a", family="monospace", zorder=3)
        else:
            ax.text(x, y + 1.35, title, fontsize=7.1, fontweight="bold",
                    ha="center", va="center", color=C_TEXT, zorder=3)
            ax.text(x, y - 1.15, sub, fontsize=6.1, ha="center", va="center",
                    color="#3f3f3f", linespacing=1.35, zorder=3)

    # Column headers
    ax.text(8, 96.5, "DATA SOURCES", fontsize=7.6, fontweight="bold",
            ha="center", color="#5a5a5a")
    ax.text(82, 63.5, "ASSEMBLY + VALIDATION", fontsize=7.6,
            fontweight="bold", ha="center", color="#5a5a5a")

    # Title and caption
    fig.text(0.012, 0.972, "Fig B  —  As-shipped Phase 2 forecast pipeline",
             fontsize=13, fontweight="bold", color=C_TEXT)
    fig.text(0.012, 0.947,
             "Six arms, one submission. Feature engineering (F1, amber) is the "
             "direction arm's load-bearing design choice: direction is an angle, "
             "so it enters as sin/cos pairs and the target is the unwrapped "
             "residual. The d+7 speed arm bypasses the kit's MOS stack entirely.",
             fontsize=7.6, color="#4a4a4a")
    fig.text(0.012, 0.018,
             "Generated by scripts/make_figB_pipeline_20260827.py from "
             "docs/figB_pipeline_node_edge_list.md (node ids gated against it).  "
             "Amber = derived-feature construction.  FROZEN / FRESH marks the "
             "swap-year runbook status.",
             fontsize=6.2, color="#8a8a8a")

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=DPI, facecolor="white",
                bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)

    from PIL import Image
    with Image.open(OUT_PNG) as im:
        px_w, px_h = im.size
    print(f"wrote {OUT_PNG.relative_to(REPO_ROOT)}")
    print(f"  {px_w} x {px_h} px at {DPI} dpi "
          f"({px_w/DPI:.2f} x {px_h/DPI:.2f} in at native scale)")
    print(f"  sha256 {sha256_of(OUT_PNG)}")
    print(f"  intended print width: 6.5 in (US Letter text width at 1 in "
          f"margins) -> effective {px_w/6.5:.0f} dpi")


if __name__ == "__main__":
    main()
