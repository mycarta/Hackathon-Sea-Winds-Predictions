# Audit v2, Tier B extracts - anchors 1, 2 and 6

Generated 2026-08-25 by `scripts/audit_export_tierB_anchors_126_20260825.py`.
Export only: no pipeline was run, nothing was refit, no frozen file was touched.
Lineage: bonus bidding simulation Stage 4/5, commit `bfbe284`.

## Anchor 1 - `revenue_by_strategy_1460h.csv`

Byte copy of `bidding_sim/results_2019/revenue_by_strategy_1460h.csv`, SHA-256 `e31c07792a41e8e29f889a95f6854cf52442f3e7671f062323135542fb004b7f` (the value `results_2019/stage4_summary.json` recorded for its own output), 1,460 delivery hours.

## Anchor 2 - `bonus_revenue_inputs_2019-03.csv`

Source parquets: production quantiles `bidding_sim/production_2019/production_quantiles_1460h.parquet` SHA-256 `a8010fa8c2cbfe4ad5ebc0265b02dcd04ede71a1e64ede5a0aa7bcb176ee9eca`; day-ahead `bidding_sim/market_data_2019/day_ahead_prices_2019.parquet` SHA-256 `b900a7f5db0e17b91aec5a5bd56d1ae0c63dd86983e9ee6b0e9374e1d0bf685c`; TenneT settlement `bidding_sim/market_data_2019/tennet_settlement_prices_2019.parquet` SHA-256 `8627dd1c6d56bee63564f735df425fd3b8d35c5574f2ecaacef8f7b7f7ab8af4`.

March 2019, 496 rows = 124 delivery hours (00/06/12/18 UTC) x 4 ISPs. Each row carries the settlement prices at their published 15-minute granularity and the hourly day-ahead price, forecast quantiles and realized production repeated across the hour's four ISPs.

Month chosen for two reasons, both disclosed: March carries the largest positive monthly EVIU (+144,557 EUR) and it contains 2019-03-31, the 23-hour local DST day of 92 ISPs that exercises the ISP-to-hour join.

**What is deliberately absent.** The 4-ISP hourly price mean that `stage4_bidding_2019.py` forms before settling, `alpha*`, the four bids and the four revenues are NOT in this file. Those are the steps the anchor tests. Supplying them would pre-answer the recomputation.

Note on naming: these are TenneT *settlement* prices (shortage / surplus), not the ENTSO-E `Long`/`Short` imbalance columns. The ENTSO-E pair was pulled first and then rejected - neither reading of it yields a valid two-price scheme. See the market-data manifest.

## Anchor 6 - `tennet_penalty_alpha_star_derivation.csv`

Source parquets: TenneT settlement SHA-256 `8627dd1c6d56bee63564f735df425fd3b8d35c5574f2ecaacef8f7b7f7ab8af4`; day-ahead SHA-256 `b900a7f5db0e17b91aec5a5bd56d1ae0c63dd86983e9ee6b0e9374e1d0bf685c`.

Long format, 122 rows. `block` = `population` (2 rows, the alpha* derivation table proper), `isp_of_day` (96 rows), `hour_of_day` (24 rows).

Sign convention, EUR/MWh: `psi_hat_plus = mean(DA - Surplus)`, the over-delivery penalty; `psi_hat_minus = mean(Shortage - DA)`, the under-delivery penalty; `alpha_star_implied = psi_hat_minus / (psi_hat_plus + psi_hat_minus)`.

| Population | n ISP | psi_hat_plus | psi_hat_minus | alpha* | Gate |
|---|---:|---:|---:|---:|---|
| full_year | 35040 | +0.8047 | +1.3583 | 0.627955 | PASS |
| delivery_instants | 5840 | +1.6369 | -0.0211 | -0.013042 | FAIL |

The `full_year` row is the basis actually used: alpha* = 0.627955. The `delivery_instants` row is the population the plan specified FIRST; it fails the positivity gate, which is why the switch was made. The switch is disclosed in `stage4_summary.json` and reproduced here so the reason is inspectable rather than asserted. The script re-derives all five figures from the parquets and asserts them against `stage4_summary.json` before writing.

## Exported file hashes

Raw-byte SHA-256 of the file as written. The three CSVs are CRLF - two as pandas `to_csv` writes them on Windows, and anchor 1 as a byte copy of a CRLF working copy, which is what makes its hash equal to the one `stage4_summary.json` recorded for itself. `core.autocrlf` is `true` in this checkout, so all three are listed `-text` in `.gitattributes`: without that, git would store LF blobs and a Linux clone would hold different bytes from the ones hashed here. The hashes below are therefore properties of the content, not of the machine that wrote them.

| File | Bytes | SHA-256 |
|---|---:|---|
| `revenue_by_strategy_1460h.csv` | 430844 | `e31c07792a41e8e29f889a95f6854cf52442f3e7671f062323135542fb004b7f` |
| `bonus_revenue_inputs_2019-03.csv` | 96613 | `1bf63c875a5612ba84c39854dfb97d16c0680bf5d3bb5c05d0a3d418dad5d8ab` |
| `tennet_penalty_alpha_star_derivation.csv` | 17399 | `3dc60658b59ea1284fa0e1c8b9c276432178ce7581211811110d63aaafd4089a` |
