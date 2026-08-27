"""
Audit v2, Tier B anchors 1, 2 and 6 (audit_design_v2 section 4.3): extract-only.

Exports three files into docs/audit/extracts/ for the strategist's independent
recomputation of the bonus bidding simulation (Stage 4/5, commit bfbe284).

EXPORT ONLY. This script:
  - runs no part of the pipeline and refits nothing,
  - writes nothing outside docs/audit/extracts/,
  - reads its four sources read-only and asserts every one by SHA-256 first
    (contract A1, and contract v2.1 section 5a for the pinned Stage-4 inputs).

Anchor 1  revenue_by_strategy_1460h.csv
          Byte copy of the Stage 4/5 per-hour output.

Anchor 2  bonus_revenue_inputs_2019-03.csv
          One month of the INPUTS to the revenue calculation, at the native
          granularity of each source: settlement prices per ISP as published,
          day-ahead per hour, forecast quantiles and realized production per
          delivery hour.

          DELIBERATELY NOT INCLUDED: the 4-ISP hourly price mean that
          stage4_bidding_2019.py forms before settling, alpha*, the bids, and
          the revenues. Those are the steps the anchor tests. Handing them over
          pre-answers the recomputation (the anchor-5 precedent, 2026-08-24).

          March is the chosen month, and the choice is disclosed rather than
          arbitrary: it carries the largest positive monthly EVIU
          (+144,557 EUR, monthly_breakdown.csv) AND the spring DST transition
          of 2019-03-31, a 23-hour local day of 92 ISPs. The ISP-to-hour join
          is the part of the pipeline most likely to be silently wrong, so the
          month that exercises it is the month worth anchoring.

Anchor 6  tennet_penalty_alpha_star_derivation.csv
          The penalty extracts behind alpha* = 0.6280. Long format, one
          `block` column:
            population  the two candidate estimation populations, which is the
                        alpha* derivation table proper, including the FORCED
                        population switch (the delivery-instant population
                        fails the positivity gate);
            isp_of_day  96 rows, penalty means per within-day ISP ordinal;
            hour_of_day 24 rows, penalty means per UTC hour, which is the
                        evidence behind the "negative in 11 of 24 hours"
                        statement in stage4_summary.json.

Sign convention, both penalties in EUR/MWh, matching
stage4_bidding_2019.py:478-482:
    psi_hat_plus  = mean(DA - Surplus)    penalty for over-delivery
    psi_hat_minus = mean(Shortage - DA)   penalty for under-delivery
    alpha*        = psi_hat_minus / (psi_hat_plus + psi_hat_minus)

Deterministic. No stochastic step, so no seed applies.

Run:  conda run -n swnd python scripts/audit_export_tierB_anchors_126_20260825.py
"""

import hashlib
import json
import os
import shutil

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(REPO, "docs", "audit", "extracts")

MONTH = 3
MONTH_TAG = "2019-03"

# ---------------------------------------------------------------------------
# Sources, each asserted by SHA-256 before it is read.
#
# The three parquet SHAs are the manifested values:
#   production  bidding_sim/production_2019/MANIFEST.md
#   day-ahead   bidding_sim/market_data_2019/MANIFEST.md
#   TenneT      bidding_sim/market_data_2019/MANIFEST.md, 2026-08-21 addition
# and they are the same literals stage4_bidding_2019.py:76-85 gates on.
#
# The CSV SHA is the output hash stage4_bidding_2019.py recorded for itself in
# results_2019/stage4_summary.json ("output_sha256"."per_hour_csv"), so a byte
# copy that passes this gate is provably the bfbe284 artifact.
# ---------------------------------------------------------------------------
SOURCES = {
    "revenue_csv": (
        os.path.join(REPO, "bidding_sim", "results_2019", "revenue_by_strategy_1460h.csv"),
        "e31c07792a41e8e29f889a95f6854cf52442f3e7671f062323135542fb004b7f",
    ),
    "production_parquet": (
        os.path.join(REPO, "bidding_sim", "production_2019", "production_quantiles_1460h.parquet"),
        "a8010fa8c2cbfe4ad5ebc0265b02dcd04ede71a1e64ede5a0aa7bcb176ee9eca",
    ),
    "dayahead_parquet": (
        os.path.join(REPO, "bidding_sim", "market_data_2019", "day_ahead_prices_2019.parquet"),
        "b900a7f5db0e17b91aec5a5bd56d1ae0c63dd86983e9ee6b0e9374e1d0bf685c",
    ),
    "tennet_parquet": (
        os.path.join(REPO, "bidding_sim", "market_data_2019", "tennet_settlement_prices_2019.parquet"),
        "8627dd1c6d56bee63564f735df425fd3b8d35c5574f2ecaacef8f7b7f7ab8af4",
    ),
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gated_path(key):
    path, expected = SOURCES[key]
    got = sha256_file(path)
    assert got == expected, "%s SHA MISMATCH\n  expected %s\n  got      %s\n  %s" % (
        key, expected, got, path)
    print("  %-20s SHA OK  %s..." % (key, got[:16]))
    return path


def penalties(frame):
    """The pipeline's estimator, stage4_bidding_2019.py:478-482, verbatim."""
    p_plus = float((frame["da_eur_mwh"] - frame["surplus_eur_mwh"]).mean())
    p_minus = float((frame["shortage_eur_mwh"] - frame["da_eur_mwh"]).mean())
    return p_plus, p_minus


def alpha_of(p_plus, p_minus):
    tot = p_plus + p_minus
    return float(p_minus / tot) if tot else float("nan")


def main():
    print("audit v2 Tier B anchors 1, 2, 6 - export only")
    print("\ninput SHA gate (contract A1)")
    revenue_csv = gated_path("revenue_csv")
    prod_parquet = gated_path("production_parquet")
    da_parquet = gated_path("dayahead_parquet")
    tn_parquet = gated_path("tennet_parquet")

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)
    written = []

    # -----------------------------------------------------------------------
    # ANCHOR 1: byte copy.
    # -----------------------------------------------------------------------
    a1 = os.path.join(OUTDIR, "revenue_by_strategy_1460h.csv")
    shutil.copyfile(revenue_csv, a1)
    assert sha256_file(a1) == SOURCES["revenue_csv"][1], "anchor 1 copy is not byte identical"
    written.append(a1)
    print("\nanchor 1: byte copy verified identical to source")

    # -----------------------------------------------------------------------
    # Load the three sources exactly as stage4_bidding_2019.py loads them.
    # -----------------------------------------------------------------------
    prod = pd.read_parquet(prod_parquet)
    prod["utc_time"] = pd.to_datetime(prod["utc_time"], utc=True)
    prod = prod.sort_values("utc_time").reset_index(drop=True)
    assert len(prod) == 1460, "expected 1460 delivery hours, got %d" % len(prod)

    da = pd.read_parquet(da_parquet)
    if not isinstance(da.index, pd.DatetimeIndex):
        da = da.set_index(da.columns[0])
    da.index = pd.to_datetime(da.index, utc=True)
    da = da.rename(columns={da.columns[0]: "da_eur_mwh"})[["da_eur_mwh"]]
    assert len(da) == 8760, "expected 8760 day-ahead hours, got %d" % len(da)

    tn = pd.read_parquet(tn_parquet)
    tn["utc_time"] = pd.to_datetime(tn["utc_time"], utc=True)
    assert len(tn) == 35040, "expected 35040 ISPs, got %d" % len(tn)
    tn["hour"] = tn["utc_time"].dt.floor("h")

    # -----------------------------------------------------------------------
    # ANCHOR 2: one month of inputs, per ISP.
    # -----------------------------------------------------------------------
    prod_m = prod[prod["utc_time"].dt.month == MONTH].copy()
    assert len(prod_m) == 124, "expected 124 March delivery hours, got %d" % len(prod_m)

    hours_m = set(prod_m["utc_time"])
    isp_m = tn[tn["hour"].isin(hours_m)].copy()
    assert len(isp_m) == 4 * len(prod_m), (
        "March delivery-instant join is %d, expected %d" % (len(isp_m), 4 * len(prod_m)))

    slice_df = isp_m.merge(da, left_on="hour", right_index=True, how="inner")
    assert len(slice_df) == 4 * len(prod_m), "day-ahead join lost March rows"

    prod_cols = ["utc_time", "p_q05_mw", "p_q50_mw", "p_q95_mw", "p_real_mw",
                 "p_dang05_mw", "p_dang50_mw", "p_dang95_mw", "cutout_affected"]
    slice_df = slice_df.merge(prod_m[prod_cols], left_on="hour", right_on="utc_time",
                              how="inner", suffixes=("_isp", "_hour"))
    assert len(slice_df) == 4 * len(prod_m), "production join lost March rows"

    slice_df = slice_df.rename(columns={
        "utc_time_isp": "isp_utc_time",
        "hour": "delivery_hour_utc",
        "shortage_eur_mwh": "isp_shortage_eur_mwh",
        "surplus_eur_mwh": "isp_surplus_eur_mwh",
    }).drop(columns=["utc_time_hour"])

    keep2 = (["delivery_hour_utc", "isp_utc_time", "local_stamp", "isp",
              "regulation_state", "isp_shortage_eur_mwh", "isp_surplus_eur_mwh",
              "da_eur_mwh"]
             + [c for c in prod_cols if c != "utc_time"])
    slice_df = slice_df[keep2].sort_values(["delivery_hour_utc", "isp"]).reset_index(drop=True)

    # The DST day must be present: 2019-03-31 is a 23-hour local day, and its
    # 00/06/12/18 UTC delivery hours are what make this month worth anchoring.
    dst_rows = slice_df[slice_df["isp_utc_time"].dt.strftime("%Y-%m-%d") == "2019-03-31"]
    assert len(dst_rows) > 0, "the 2019-03-31 DST day is missing from the slice"

    a2 = os.path.join(OUTDIR, "bonus_revenue_inputs_%s.csv" % MONTH_TAG)
    slice_df.to_csv(a2, index=False)
    written.append(a2)
    print("anchor 2: %d rows (%d delivery hours x 4 ISPs), 2019-03-31 DST day present (%d rows)"
          % (len(slice_df), len(prod_m), len(dst_rows)))

    # -----------------------------------------------------------------------
    # ANCHOR 6: the penalty extracts behind alpha*.
    # -----------------------------------------------------------------------
    isp_all = tn.merge(da, left_on="hour", right_index=True, how="inner")
    delivery_hours = set(prod["utc_time"])
    isp_del = isp_all[isp_all["hour"].isin(delivery_hours)]
    assert len(isp_del) == 4 * 1460, "delivery-instant join is %d, expected 5840" % len(isp_del)

    rows = []

    def emit(block, key, note, frame):
        p_plus, p_minus = penalties(frame)
        rows.append({
            "block": block,
            "group_key": key,
            "population": note,
            "n_isp": int(len(frame)),
            "psi_hat_plus_eur_mwh": p_plus,
            "psi_hat_minus_eur_mwh": p_minus,
            "psi_sum_eur_mwh": p_plus + p_minus,
            "alpha_star_implied": alpha_of(p_plus, p_minus),
            "positivity_gate": "PASS" if (p_plus > 0 and p_minus > 0) else "FAIL",
        })

    # The derivation table proper: the two candidate populations.
    emit("population", "full_year",
         "all ISPs joinable to day-ahead; THE BASIS alpha* IS BUILT FROM", isp_all)
    emit("population", "delivery_instants",
         "the 4 ISPs of each of the 1,460 delivery hours; SPECIFIED FIRST, then "
         "FORCED OUT because it fails the positivity gate", isp_del)

    for i in range(1, 97):
        sub = isp_all[isp_all["isp"] == i]
        if len(sub):
            emit("isp_of_day", str(i), "full year, ISP ordinal %d of the local day" % i, sub)

    for h in range(24):
        sub = isp_all[isp_all["hour"].dt.hour == h]
        if len(sub):
            emit("hour_of_day", "%02d" % h, "full year, UTC hour %02d" % h, sub)

    alpha = pd.DataFrame(rows)

    # Reproduce the two headline numbers stage4_summary.json recorded: this
    # export is worthless as an anchor if it does not.
    with open(os.path.join(REPO, "bidding_sim", "results_2019", "stage4_summary.json"),
              encoding="utf-8") as fh:
        summ = json.load(fh)
    fy = alpha[(alpha["block"] == "population") & (alpha["group_key"] == "full_year")].iloc[0]
    dl = alpha[(alpha["block"] == "population") & (alpha["group_key"] == "delivery_instants")].iloc[0]
    checks = [
        ("psi_hat_plus_fullyear", fy["psi_hat_plus_eur_mwh"]),
        ("psi_hat_minus_fullyear", fy["psi_hat_minus_eur_mwh"]),
        ("psi_hat_plus_delivery", dl["psi_hat_plus_eur_mwh"]),
        ("psi_hat_minus_delivery", dl["psi_hat_minus_eur_mwh"]),
        ("alpha_star_raw", fy["alpha_star_implied"]),
    ]
    for key, got in checks:
        assert np.isclose(got, summ[key], rtol=0, atol=1e-9), (
            "anchor 6 does not reproduce stage4_summary.json[%s]: %.10f != %.10f"
            % (key, got, summ[key]))
    print("anchor 6: reproduces all 5 stage4_summary.json penalty/alpha* values exactly")
    assert dl["positivity_gate"] == "FAIL", "the delivery-instant gate no longer FAILs"
    assert fy["positivity_gate"] == "PASS", "the annual gate no longer PASSes"

    a6 = os.path.join(OUTDIR, "tennet_penalty_alpha_star_derivation.csv")
    alpha.to_csv(a6, index=False)
    written.append(a6)

    # -----------------------------------------------------------------------
    # README: source SHAs for every export.
    # -----------------------------------------------------------------------
    readme = os.path.join(OUTDIR, "README.md")
    L = []
    L.append("# Audit v2, Tier B extracts - anchors 1, 2 and 6")
    L.append("")
    L.append("Generated 2026-08-25 by `scripts/audit_export_tierB_anchors_126_20260825.py`.")
    L.append("Export only: no pipeline was run, nothing was refit, no frozen file was touched.")
    L.append("Lineage: bonus bidding simulation Stage 4/5, commit `bfbe284`.")
    L.append("")
    L.append("## Anchor 1 - `revenue_by_strategy_1460h.csv`")
    L.append("")
    L.append("Byte copy of `bidding_sim/results_2019/revenue_by_strategy_1460h.csv`, "
             "SHA-256 `%s` (the value `results_2019/stage4_summary.json` recorded for its "
             "own output), 1,460 delivery hours." % SOURCES["revenue_csv"][1])
    L.append("")
    L.append("## Anchor 2 - `bonus_revenue_inputs_%s.csv`" % MONTH_TAG)
    L.append("")
    L.append("Source parquets: production quantiles "
             "`bidding_sim/production_2019/production_quantiles_1460h.parquet` SHA-256 `%s`; "
             "day-ahead `bidding_sim/market_data_2019/day_ahead_prices_2019.parquet` SHA-256 `%s`; "
             "TenneT settlement `bidding_sim/market_data_2019/tennet_settlement_prices_2019.parquet` "
             "SHA-256 `%s`."
             % (SOURCES["production_parquet"][1], SOURCES["dayahead_parquet"][1],
                SOURCES["tennet_parquet"][1]))
    L.append("")
    L.append("March 2019, %d rows = %d delivery hours (00/06/12/18 UTC) x 4 ISPs. Each row "
             "carries the settlement prices at their published 15-minute granularity and the "
             "hourly day-ahead price, forecast quantiles and realized production repeated "
             "across the hour's four ISPs." % (len(slice_df), len(prod_m)))
    L.append("")
    L.append("Month chosen for two reasons, both disclosed: March carries the largest positive "
             "monthly EVIU (+144,557 EUR) and it contains 2019-03-31, the 23-hour local DST day "
             "of 92 ISPs that exercises the ISP-to-hour join.")
    L.append("")
    L.append("**What is deliberately absent.** The 4-ISP hourly price mean that "
             "`stage4_bidding_2019.py` forms before settling, `alpha*`, the four bids and the "
             "four revenues are NOT in this file. Those are the steps the anchor tests. Supplying "
             "them would pre-answer the recomputation.")
    L.append("")
    L.append("Note on naming: these are TenneT *settlement* prices (shortage / surplus), not the "
             "ENTSO-E `Long`/`Short` imbalance columns. The ENTSO-E pair was pulled first and "
             "then rejected - neither reading of it yields a valid two-price scheme. See the "
             "market-data manifest.")
    L.append("")
    L.append("## Anchor 6 - `tennet_penalty_alpha_star_derivation.csv`")
    L.append("")
    L.append("Source parquets: TenneT settlement SHA-256 `%s`; day-ahead SHA-256 `%s`."
             % (SOURCES["tennet_parquet"][1], SOURCES["dayahead_parquet"][1]))
    L.append("")
    L.append("Long format, %d rows. `block` = `population` (2 rows, the alpha* derivation table "
             "proper), `isp_of_day` (96 rows), `hour_of_day` (24 rows)." % len(alpha))
    L.append("")
    L.append("Sign convention, EUR/MWh: `psi_hat_plus = mean(DA - Surplus)`, the over-delivery "
             "penalty; `psi_hat_minus = mean(Shortage - DA)`, the under-delivery penalty; "
             "`alpha_star_implied = psi_hat_minus / (psi_hat_plus + psi_hat_minus)`.")
    L.append("")
    L.append("| Population | n ISP | psi_hat_plus | psi_hat_minus | alpha* | Gate |")
    L.append("|---|---:|---:|---:|---:|---|")
    for _, r in alpha[alpha["block"] == "population"].iterrows():
        L.append("| %s | %d | %+.4f | %+.4f | %.6f | %s |"
                 % (r["group_key"], r["n_isp"], r["psi_hat_plus_eur_mwh"],
                    r["psi_hat_minus_eur_mwh"], r["alpha_star_implied"], r["positivity_gate"]))
    L.append("")
    L.append("The `full_year` row is the basis actually used: alpha* = 0.627955. The "
             "`delivery_instants` row is the population the plan specified FIRST; it fails the "
             "positivity gate, which is why the switch was made. The switch is disclosed in "
             "`stage4_summary.json` and reproduced here so the reason is inspectable rather than "
             "asserted. The script re-derives all five figures from the parquets and asserts them "
             "against `stage4_summary.json` before writing.")
    L.append("")
    L.append("## Exported file hashes")
    L.append("")
    L.append("Raw-byte SHA-256 of the file as written. The three CSVs are CRLF - two as pandas "
             "`to_csv` writes them on Windows, and anchor 1 as a byte copy of a CRLF working "
             "copy, which is what makes its hash equal to the one `stage4_summary.json` recorded "
             "for itself. `core.autocrlf` is `true` in this checkout, so all three are listed "
             "`-text` in `.gitattributes`: without that, git would store LF blobs and a Linux "
             "clone would hold different bytes from the ones hashed here. The hashes below are "
             "therefore properties of the content, not of the machine that wrote them.")
    L.append("")
    L.append("| File | Bytes | SHA-256 |")
    L.append("|---|---:|---|")
    for p in written:
        L.append("| `%s` | %d | `%s` |"
                 % (os.path.basename(p), os.path.getsize(p), sha256_file(p)))
    L.append("")
    with open(readme, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(L))

    print("\nwritten")
    for p in written + [readme]:
        print("  %-46s %9d B  %s"
              % (os.path.relpath(p, REPO).replace("\\", "/"), os.path.getsize(p),
                 sha256_file(p)))


if __name__ == "__main__":
    main()
