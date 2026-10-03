# -*- coding: utf-8 -*-
"""analyze_v6.py — summarize round-2 blocks ("profiles", "fine13_m4") in
v5_seeds.json; append "analysis.task1_profiles" / "analysis.task2_fine13_m4";
print tables."""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "results", "v5_seeds.json")
R = json.load(open(P, encoding="utf-8"))
SEEDS = ["42", "43", "44", "45", "46"]

def ms(vals):
    return {"mean": round(st.mean(vals), 4), "std": round(st.stdev(vals), 4), "runs": vals}

def fmt(d):
    return f"{d['mean']:.4f}+/-{d['std']:.4f}"

PR = R["profiles"]
print("=" * 84)
print("TASK 1a. attack profiles p=0.5 — macroF1 (fraudF1) per seed")
print("=" * 84)
print(f"{'seed':5s} {'clean-M4':>9s} {'struct-M4':>10s} {'eq?':>5s} | {'struct-M1':>10s} | "
      f"{'destr-M1':>9s} {'destr-M4':>9s}")
for s in SEEDS:
    r = PR[s]
    eq = r["M4_struct_eq_clean"]
    print(f"{s:5s} {r['clean']['M4']['macroF1']:9.4f} {r['structural']['M4']['macroF1']:10.4f} "
          f"{str(eq['rounded4_equal']):>5s} | {r['structural']['M1']['macroF1']:10.4f} | "
          f"{r['destructive']['M1']['macroF1']:9.4f} {r['destructive']['M4']['macroF1']:9.4f}"
          f"   exact_diff={eq['exact_diff']:+.8f}")

t1 = {"per_seed_eq": {s: PR[s]["M4_struct_eq_clean"] for s in SEEDS},
      "n_exact_equal": sum(1 for s in SEEDS if PR[s]["M4_struct_eq_clean"]["rounded4_equal"]),
      "summary": {}}
for prof in ["structural", "destructive"]:
    for m in ["M1", "M4"]:
        t1["summary"][f"{m}_{prof}"] = {
            "macroF1": ms([PR[s][prof][m]["macroF1"] for s in SEEDS]),
            "fraudF1": ms([PR[s][prof][m]["fraudF1"] for s in SEEDS])}
print("\nmean +/- std over 5 seeds:")
for k, v in t1["summary"].items():
    print(f"  {k:16s} macroF1 {fmt(v['macroF1'])}   fraudF1 {fmt(v['fraudF1'])}")

print()
print("=" * 84)
print("TASK 1b. M0 keyword filter — fraud precision / recall (paper claim: recall 0.70 -> 0.43)")
print("=" * 84)
m0 = {"clean": {"fraudP": ms([PR[s]["M0_clean"]["fraudP"] for s in SEEDS]),
                "fraudR": ms([PR[s]["M0_clean"]["fraudR"] for s in SEEDS])},
      "p0.7":  {"fraudP": ms([PR[s]["M0_p0.7"]["fraudP"] for s in SEEDS]),
                "fraudR": ms([PR[s]["M0_p0.7"]["fraudR"] for s in SEEDS])}}
t1["M0"] = m0
print(f"  clean : P {fmt(m0['clean']['fraudP'])}  R {fmt(m0['clean']['fraudR'])}  runs R={m0['clean']['fraudR']['runs']}")
print(f"  p=0.7 : P {fmt(m0['p0.7']['fraudP'])}  R {fmt(m0['p0.7']['fraudR'])}  runs R={m0['p0.7']['fraudR']['runs']}")

print()
print("=" * 84)
print("TASK 2. 13-class M4 (OvR LR, max_iter=2000, n_jobs=-1) vs M1 — macroF1 mean +/- std")
print("=" * 84)
FM = R["fine13_m4"]; F1 = R["fine13"]
t2 = {"config": FM["_meta"], "calibration_seed42": FM["42"]["clean_vs_archive"],
      "convergence_warnings": {s: FM[s]["convergence_warning"] for s in SEEDS},
      "train_s": {s: FM[s]["train_s"] for s in SEEDS}}
print(f"  calibration seed42 clean: {FM['42']['clean']['macroF1']} vs archive 0.9506 -> "
      f"diff {FM['42']['clean_vs_archive']['diff']:+.4f} (within tol: {FM['42']['clean_vs_archive']['within_0.003']})")
print(f"  convergence warnings: {t2['convergence_warnings']}  train_s: {t2['train_s']}")
for cond in ["clean", "p0.3"]:
    m4v = [FM[s][cond]["macroF1"] for s in SEEDS]
    m1v = [F1[s][cond]["M1"]["macroF1"] for s in SEEDS]
    diffs = [round(a - b, 4) for a, b in zip(m4v, m1v)]
    t2[cond] = {"M4": ms(m4v), "M1": ms(m1v), "paired_diff_M4_minus_M1": diffs,
                "paired_diff_mean": round(st.mean(diffs), 4)}
    print(f"  {cond:6s} M4 {fmt(t2[cond]['M4'])}   M1 {fmt(t2[cond]['M1'])}   "
          f"paired diff {t2[cond]['paired_diff_mean']:+.4f}  runs {diffs}")

R.setdefault("analysis", {})["task1_profiles"] = t1
R["analysis"]["task2_fine13_m4"] = t2
tmp = P + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
os.replace(tmp, P)
print(f"\nsummary appended -> {P}")
