#!/usr/bin/env python3
"""WS d7 Tier-1 feature experiments — orchestrator + results collation.

Opus-authorized dispatch, 2026-07-16 (Part A). Runs each arm of
ws_d7_feature_experiments.py in its own subprocess (clean module state /
lru_cache per arm), applies the dispatch's conditional logic, and writes the
results table.

Sequencing (per the approved design):
  1. A0 (seeded baseline — the comparator), A2 (static), A3 (residual-both).
  2. A4 = combination of {A2, A3} components that INDIVIDUALLY beat A0 on the
     primary metric (fine d7 Winkler, bias + alpha=0.90). If only one beats A0,
     A4 == that arm and is skipped (logged). If neither beats, A4 skipped.
  3. A3b (det-only residual decomposition) — run ONLY if A3 wins meaningfully,
     i.e. relative improvement on the primary metric >= A3B_THRESHOLD.

Primary metric: fine d7 Winkler (bias-corrected, alpha=0.90). Lower is better.
Board 27.14 is context, not the baseline — every delta is vs seeded A0.

Outputs: scripts/artifacts/ws_d7_feature_experiments_results.csv and a printed
markdown table. Nothing is submitted; board decisions are made in review.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ARM_SCRIPT = HERE / "ws_d7_feature_experiments.py"
A3B_THRESHOLD = 0.005          # 0.5% relative improvement on fine d7 => "meaningful"


def run_arm(arm: str) -> dict:
    print(f"\n===== RUN {arm} =====", flush=True)
    subprocess.run([sys.executable, str(ARM_SCRIPT), "--arm", arm], check=True)
    return load_arm(arm)


def load_arm(arm: str) -> dict:
    with open(ARTIFACTS / f"ws_d7_featexp_{arm}.json", encoding="utf-8") as f:
        return json.load(f)


def d7_primary(res: dict) -> float:
    return res["fine"]["d7_primary_bias_alpha090"]["winkler"]


def collate(results: dict):
    a0 = results["A0"]
    base = d7_primary(a0)

    def row(res):
        arm = res["arm"]
        prov = res["provenance"]
        c1 = res["coarse"]["d1"]
        c7 = res["coarse"]["d7"]
        f7 = res["fine"]["d7_primary_bias_alpha090"]
        f1 = res["fine"]["d1_native"]
        f14 = res["fine"]["d14_native"]
        d7w = f7["winkler"]
        return {
            "arm": arm,
            "features": "+".join(prov["extra_features"]) or "(base)",
            "target": prov["target"],
            "coarse_d1": c1["winkler"] if c1 else None,
            "coarse_d7": c7["winkler"] if c7 else None,
            "fine_d7_primary": d7w,
            "fine_d7_delta_vs_A0": d7w - base,
            "fine_d7_pct_vs_A0": 100.0 * (d7w - base) / base,
            "fine_d7_coverage": f7["coverage"],
            "fine_d1": f1["winkler"] if f1 else None,
            "fine_d14": f14["winkler"] if f14 else None,
        }

    order = ["A0", "A2", "A3", "A3b", "A4"]
    rows = [row(results[a]) for a in order if a in results]

    # CSV
    import csv
    csv_path = ARTIFACTS / "ws_d7_feature_experiments_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {csv_path}")

    # markdown table
    hdr = ["arm", "features", "target", "coarse d7", "fine d7 (bias,a0.90)",
           "Δ vs A0", "% vs A0", "cov d7", "coarse d1", "fine d1", "fine d14"]
    print("\n| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        def fmt(x, p="{:.3f}"):
            return p.format(x) if isinstance(x, (int, float)) else "—"
        print("| " + " | ".join([
            r["arm"], r["features"], r["target"],
            fmt(r["coarse_d7"]), fmt(r["fine_d7_primary"]),
            fmt(r["fine_d7_delta_vs_A0"], "{:+.3f}"),
            fmt(r["fine_d7_pct_vs_A0"], "{:+.2f}%"),
            fmt(r["fine_d7_coverage"], "{:.3f}"),
            fmt(r["coarse_d1"]), fmt(r["fine_d1"]), fmt(r["fine_d14"]),
        ]) + " |")
    return rows


def main():
    results = {}
    for arm in ("A0", "A2", "A3"):
        results[arm] = run_arm(arm)

    base = d7_primary(results["A0"])
    beats = {a: d7_primary(results[a]) < base for a in ("A2", "A3")}
    print(f"\nbeats A0 (fine d7): " +
          ", ".join(f"{a}={'Y' if beats[a] else 'N'} "
                    f"({d7_primary(results[a]):.3f} vs {base:.3f})"
                    for a in ("A2", "A3")))

    # A4: combination of components individually beating A0
    winners = [a for a in ("A2", "A3") if beats[a]]
    if len(winners) >= 2:
        print(f"\nA4 = combine {winners} (both beat A0) -> running A4")
        results["A4"] = run_arm("A4")
    else:
        print(f"\nA4 skipped: fewer than 2 components beat A0 (winners={winners}). "
              f"A4 would duplicate a single arm.")

    # A3b: det-only decomposition if A3 wins meaningfully
    a3_pct = (base - d7_primary(results["A3"])) / base
    if a3_pct >= A3B_THRESHOLD:
        print(f"\nA3 wins meaningfully ({a3_pct*100:.2f}% >= "
              f"{A3B_THRESHOLD*100:.2f}%) -> running A3b (det-only)")
        results["A3b"] = run_arm("A3b")
    else:
        print(f"\nA3b skipped: A3 improvement {a3_pct*100:.2f}% < "
              f"{A3B_THRESHOLD*100:.2f}% threshold.")

    collate(results)


if __name__ == "__main__":
    main()
