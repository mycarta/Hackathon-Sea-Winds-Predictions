"""
Audit v2, anchor 6: independent recompute of ONE S3 cell in a clean
environment (dispatch of 2026-08-25, item 6; run authorised after cell 4 was
dropped on 08-24).

Target: case1_grid_baseline_centre, the plain 7D grid at the organizer
baseline centre 53.5N 1.5E.

Expected, from reports/three_case_scorer_20260818.json:
    cf_net             0.4709482933331519   (47.0948 percent)
    wake_loss_fraction 0.09699240714124902  (9.6992 percent)
Tolerance: 0.1 percentage point on net CF, per the dispatch.

WHAT THIS IS TESTING, and what it is not. This re-runs the S3 physics in a
freshly created environment built from the kit's requirements file, with no
cached artifacts, and checks that the number comes back. It tests
environment-independence and re-runnability. It does NOT test the number
against the spec independently: the dispatch permits reading repo code for
the unstated defaults, and this script takes all of them from
scripts/task2_scorer_replica.py, which is the same module the 2026-08-19 run
used. A from-spec-only reconstruction was the job of anchor 5 cell 4, and
that was dropped on 08-24; rung 2's independence now rests on the kit
simulate_year cross-check of 2026-07-13.

So: agreement here is evidence the pipeline is reproducible. It is not
evidence the configuration is correct. Those are different claims and the
post-statement keeps them apart.

Deterministic. No stochastic step anywhere in this path (the AROME series is
real data, the layout is analytic, PyWake is deterministic), so no seed
applies.

Run:  conda run -n s3anchor20260825 python \\
          scripts/audit_anchor6_s3_case1_recompute_20260825.py
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KIT_PHASE2 = REPO / "phase_2" / "kit" / "phase_2"

# The dataset root moved after the S3 run: phase_2/build/phase2_dataset is
# gone and the train tree now lives under inference_2022/. target_loader reads
# PHASE2_DATA_ROOT, so it is set here rather than left to the stale fallback.
# Same resolution the anchor-5 export used on 08-24.
DATA_ROOT = Path(os.environ.get(
    "PHASE2_DATA_ROOT",
    REPO / "phase_2" / "inference_2022" / "phase2_dataset_ship"))
os.environ["PHASE2_DATA_ROOT"] = str(DATA_ROOT)

sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(KIT_PHASE2))
sys.path.insert(0, str(KIT_PHASE2 / "part0_dataset_setup"))

BASELINE_CENTRE = (53.5, 1.5)
N_TURBINES = 55

EXPECTED = {
    "cf_net": 0.4709482933331519,
    "wake_loss_fraction": 0.09699240714124902,
    "aep_gwh": 4991.863530014077,
    "mean_ws_hub": 10.46489499321267,
    "n_steps": 14608,
}
CF_TOL_PP = 0.1

OUT = REPO / "reports" / "audit_anchor6_s3_case1_20260825.json"


def env_fingerprint():
    """pip freeze, its SHA-256, and the resolved PyWake version."""
    p = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                       capture_output=True, text=True)
    freeze = p.stdout
    sha = hashlib.sha256(freeze.encode("utf-8")).hexdigest()
    pw = [l for l in freeze.splitlines() if l.lower().startswith("py-wake")
          or l.lower().startswith("py_wake")]
    return freeze, sha, (pw[0] if pw else "py_wake NOT FOUND in pip freeze")


def main():
    t_start = time.time()
    print("=" * 74)
    print("anchor 6: S3 case1_grid_baseline_centre, clean-environment recompute")
    print("=" * 74)
    print("python     : %s" % sys.version.split()[0])
    print("executable : %s" % sys.executable)
    print("data root  : %s" % DATA_ROOT)
    if not DATA_ROOT.exists():
        raise FileNotFoundError("dataset root not found: %s" % DATA_ROOT)

    freeze, freeze_sha, pywake_line = env_fingerprint()
    print("pip freeze SHA-256 : %s" % freeze_sha)
    print("PyWake resolved    : %s" % pywake_line)
    freeze_path = REPO / "reports" / "audit_anchor6_pipfreeze_20260825.txt"
    freeze_path.write_text(freeze, encoding="utf-8", newline="\n")

    import numpy as np
    import task2_scorer_replica as rep

    import py_wake
    pywake_version = getattr(py_wake, "__version__", "unknown")
    print("py_wake.__version__: %s" % pywake_version)

    # ---- configuration actually used, read from the replica ---------------
    cfg = {
        "diameter_m": rep.DIAMETER_M,
        "hub_height_m": rep.HUB_HEIGHT_M,
        "spacing_d": rep.SPACING_D,
        "shear_alpha": rep.SHEAR_ALPHA,
        "ref_height_m": rep.REF_HEIGHT_M,
        "n_ti_sectors": rep.N_TI_SECTORS,
        "n_turbines": N_TURBINES,
        "deficit_model": "BastankhahGaussianDeficit (library defaults)",
        "wind_farm_model": "PropagateDownwind",
        "superposition_model": "LinearSum",
        "turbulence_model": None,
        "blockage_model": None,
        "rotor_avg_model": None,
    }
    print("\nconfiguration taken from scripts/task2_scorer_replica.py")
    for k, v in cfg.items():
        print("  %-22s %s" % (k, v))

    # ---- layout ------------------------------------------------------------
    turbine = rep.build_turbine()
    x_m, y_m = rep.grid_layout()
    assert x_m.size == N_TURBINES, "grid_layout returned %d turbines" % x_m.size
    spacing_m = rep.SPACING_D * rep.DIAMETER_M
    print("\nlayout: %d turbines, %.0fD x %.0f m = %.0f m spacing"
          % (x_m.size, rep.SPACING_D, rep.DIAMETER_M, spacing_m))

    # ---- wind --------------------------------------------------------------
    print("\nloading AROME at the baseline centre %s" % (BASELINE_CENTRE,))
    t_load = time.time()
    times, ws_hub, wd = rep.load_arome_series(*BASELINE_CENTRE)
    load_s = time.time() - t_load
    print("  load wall time: %.1f s" % load_s)

    # Gate the input before spending the physics: if the series is not the one
    # the 08-19 run consumed, a matching CF would be a coincidence and a
    # mismatching CF would be misdiagnosed as a physics difference.
    assert ws_hub.size == EXPECTED["n_steps"], (
        "step count %d != the %d the S3 run recorded; the wind input is not "
        "the same series" % (ws_hub.size, EXPECTED["n_steps"]))
    d_ws = abs(float(np.mean(ws_hub)) - EXPECTED["mean_ws_hub"])
    assert d_ws < 1e-6, (
        "mean hub wind speed differs by %.3e from the S3 record" % d_ws)
    print("  input gate PASS: %d steps, mean ws %.6f m/s matches the S3 record"
          % (ws_hub.size, float(np.mean(ws_hub))))

    # ---- physics -----------------------------------------------------------
    print("\nrunning the wake model")
    t_phys = time.time()
    res = rep.run_case("case1_grid_baseline_centre", times, ws_hub, wd,
                       turbine, x_m, y_m)
    phys_s = time.time() - t_phys

    cf = res["capacity_factor"]
    wake = res["wake_loss_fraction"]
    d_cf_pp = (cf - EXPECTED["cf_net"]) * 100.0
    d_wake_pp = (wake - EXPECTED["wake_loss_fraction"]) * 100.0
    passed = abs(d_cf_pp) <= CF_TOL_PP

    total_s = time.time() - t_start
    print("\n" + "=" * 74)
    print("RESULT")
    print("=" * 74)
    print("  %-22s %12s %12s %12s" % ("quantity", "recomputed", "S3 record", "delta pp"))
    print("  %-22s %12.6f %12.6f %12.6f"
          % ("net capacity factor", cf, EXPECTED["cf_net"], d_cf_pp))
    print("  %-22s %12.6f %12.6f %12.6f"
          % ("wake loss fraction", wake, EXPECTED["wake_loss_fraction"], d_wake_pp))
    print("  %-22s %12.3f %12.3f %12.3f"
          % ("AEP GWh", res["aep_gwh"], EXPECTED["aep_gwh"],
             res["aep_gwh"] - EXPECTED["aep_gwh"]))
    print()
    print("  tolerance: %.2f pp on net CF" % CF_TOL_PP)
    print("  ANCHOR 6: %s" % ("PASS" if passed else "FAIL"))
    print()
    print("  wall time: load %.1f s, physics %.1f s, total %.1f s"
          % (load_s, phys_s, total_s))

    OUT.write_text(json.dumps({
        "anchor": "audit v2 anchor 6, S3 case1_grid_baseline_centre",
        "date": "2026-08-25",
        "generated_by": "scripts/audit_anchor6_s3_case1_recompute_20260825.py",
        "environment": {
            "conda_env": "s3anchor20260825",
            "python": sys.version.split()[0],
            "pip_freeze_sha256": freeze_sha,
            "pip_freeze_file": "reports/audit_anchor6_pipfreeze_20260825.txt",
            "py_wake_version": pywake_version,
            "py_wake_freeze_line": pywake_line,
        },
        "data_root": str(DATA_ROOT),
        "configuration": cfg,
        "input_gate": {"n_steps": int(ws_hub.size),
                       "mean_ws_hub": float(np.mean(ws_hub)), "passed": True},
        "recomputed": {"cf_net": cf, "wake_loss_fraction": wake,
                       "aep_gwh": res["aep_gwh"],
                       "mean_ws_hub": res["mean_ws_hub"],
                       "rated_capacity_mw": res["rated_capacity_mw"]},
        "s3_record": EXPECTED,
        "delta": {"cf_net_pp": d_cf_pp, "wake_pp": d_wake_pp,
                  "aep_gwh": res["aep_gwh"] - EXPECTED["aep_gwh"]},
        "tolerance_cf_pp": CF_TOL_PP,
        "verdict": "PASS" if passed else "FAIL",
        "wall_time_s": {"load": load_s, "physics": phys_s, "total": total_s},
    }, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("  written: %s" % OUT.relative_to(REPO).as_posix())
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
