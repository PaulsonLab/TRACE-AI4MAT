"""Verify a hand-edited suite has Deco Hop and JNK3 wired up correctly.

Checks every place the two oracles have to be registered, and scores two
molecules whose values are fixed by the definitions. Run before launching:

    python check_new_oracles.py            # or set SUITE and run from anywhere
"""
import json, os, shutil, sys
from pathlib import Path

SUITE = Path(os.environ.get("SUITE") or os.environ.get("TRACE_REINVENT_SUITE_ROOT") or Path(__file__).resolve().parent)
os.environ["TRACE_REINVENT_SUITE_ROOT"] = str(SUITE)
sys.path.insert(0, str(SUITE))
fails = []

def check(ok, label, fix=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    if not ok:
        fails.append(label)
        if fix:
            print(f"        -> {fix}")

print(f"\nSuite: {SUITE}\n\n[1] files that must exist")
for rel, fix in [
    ("trace_reinvent/rf_oracle.py", "new file - numpy random forest"),
    ("assets/jnk3_forest.npz", "new file - 1.1 MB, must be uploaded, cannot be typed"),
    ("initial_data/deco_hop/manifest.json", "new folder - 5 seed CSVs + manifest"),
    ("initial_data/jnk3/manifest.json", "new folder - 5 seed CSVs + manifest"),
]:
    check((SUITE / rel).exists(), rel, fix)
for oracle in ("deco_hop", "jnk3"):
    missing = [s for s in (11, 22, 33, 44, 55) if not (SUITE / "initial_data" / oracle / f"seed_{s}.csv").exists()]
    check(not missing, f"initial_data/{oracle}/seed_*.csv (5 files)", f"missing seeds: {missing}")

print("\n[2] stale bytecode")
caches = list(SUITE.rglob("__pycache__"))
for c in caches:
    shutil.rmtree(c, ignore_errors=True)
print(f"  removed {len(caches)} __pycache__ dir(s) so edits are actually picked up")

print("\n[3] trace_reinvent/oracles.py")
try:
    from trace_reinvent.oracles import ORACLE_ORDER, get_oracle_spec
    check("deco_hop" in ORACLE_ORDER, "ORACLE_ORDER contains deco_hop", "edit the ORACLE_ORDER list near line 23")
    check("jnk3" in ORACLE_ORDER, "ORACLE_ORDER contains jnk3", "edit the ORACLE_ORDER list near line 23")
    TARGET = "CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"
    for name, expected in (("deco_hop", 0.5), ("jnk3", 0.3)):
        try:
            got = get_oracle_spec(name).score_smiles_value(TARGET)
            check(abs(got - expected) < 1e-9, f"{name} scores the target molecule at {expected} (got {got})",
                  "score differs -> the scoring function body was mistyped")
        except Exception as exc:
            check(False, f"{name} resolves via get_oracle_spec", f"{type(exc).__name__}: {exc}")
except Exception as exc:
    check(False, "import trace_reinvent.oracles", f"{type(exc).__name__}: {exc}")

print("\n[4] run_reinvent_suite.py")
try:
    from run_reinvent_suite import ORACLE_CONFIGS, normalize_oracles
    for name in ("deco_hop", "jnk3"):
        check(name in ORACLE_CONFIGS, f"ORACLE_CONFIGS has {name}", "add the constraints line near line 23")
    try:
        got = normalize_oracles(["deco_hop", "jnk3"])
        check(got == ["deco_hop", "jnk3"], f"--oracles deco_hop jnk3 resolves (got {got})")
    except Exception as exc:
        check(False, "--oracles deco_hop jnk3 resolves", f"add them to the aliases dict in normalize_oracles: {exc}")
except Exception as exc:
    check(False, "import run_reinvent_suite", f"{type(exc).__name__}: {exc}")

print("\n[5] configs/oracle_suite.json")
try:
    cfg = json.load(open(SUITE / "configs" / "oracle_suite.json"))
    for name in ("deco_hop", "jnk3"):
        check(name in cfg.get("oracles", []), f"oracles list has {name}")
        check(name in cfg.get("oracle_constraints", {}), f"oracle_constraints has {name}")
except Exception as exc:
    check(False, "configs/oracle_suite.json parses", f"{type(exc).__name__}: {exc} (trailing comma?)")

print("\n[6] the REINVENT scoring component itself")
try:
    from reinvent_plugins.components.comp_trace_oracles import Parameters, TraceOracleReward
    for name in ("deco_hop", "jnk3"):
        c = TraceOracleReward(Parameters(oracle_name=[name], reward_low=[0.0], reward_high=[1.0],
                                         max_sa_score=[6.5], max_smiles_length=[100], max_heavy_atoms=[50]))
        res = c(["CCCOc1cc2ncnc(Nc3ccc4ncsc4c3)c2cc1S(=O)(=O)C(C)(C)C"])
        check(res is not None, f"component builds and scores for {name}")
except ImportError as exc:
    print(f"  SKIP  component check (REINVENT not importable here): {exc}")
except Exception as exc:
    check(False, "component builds", f"{type(exc).__name__}: {exc}")

print("\n" + "=" * 60)
print(f"{len(fails)} check(s) FAILED: {fails}" if fails else "All checks passed - safe to launch the suite.")
sys.exit(1 if fails else 0)
