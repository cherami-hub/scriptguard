# -*- coding: utf-8 -*-
"""analyze_v5.py — read results/v5_seeds.json, verify A-reproduction,
build B comparison, C mean+/-std, D summary, timing stats; append back as
R['analysis'] and print human-readable tables."""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "results", "v5_seeds.json")
R = json.load(open(P, encoding="utf-8"))

SEEDS = ["42", "43", "44", "45", "46"]
CONDS = ["clean", "p0.3", "p0.5", "p0.7"]
MODELS = ["M1", "M2", "M3", "M4", "M4b"]
TOL = 0.002

# paper reference values (Table II clean macroF1; Table III p=0.7 fraudF1)
REF_CLEAN_MACROF1 = {"M1": 0.9854, "M2": 0.9903, "M3": 0.4704, "M4": 0.9884}
REF_P07_FRAUDF1   = {"M1": 0.9797, "M2": 0.9847, "M3": 0.3701, "M4": 0.9806}
# original results.json (same pipeline, archived Aug 2) for extra context
REF_ORIG = json.load(open(os.path.join(HERE, "..", "results", "results.json"), encoding="utf-8"))

B = R["binary"]
A = {"tolerance": TOL, "checks": []}
print("=" * 78)
print("A. REPRODUCTION CHECK (seed 42)   diff = rerun - reference")
print("=" * 78)
print(f"{'check':28s} {'rerun':>8s} {'ref':>8s} {'diff':>9s}  verdict")
ok_all = True
for m in ["M1", "M2", "M3", "M4"]:
    v = B["42"]["clean"][m]["macroF1"]; ref = REF_CLEAN_MACROF1[m]
    ok = abs(v - ref) <= TOL; ok_all &= ok
    A["checks"].append({"what": f"clean macroF1 {m}", "rerun": v, "ref": ref,
                        "diff": round(v - ref, 4), "ok": ok})
    print(f"clean macroF1 {m:14s} {v:8.4f} {ref:8.4f} {v-ref:+9.4f}  {'OK' if ok else 'DIFF'}")
for m in ["M1", "M2", "M3", "M4"]:
    v = B["42"]["p0.7"][m]["fraudF1"]; ref = REF_P07_FRAUDF1[m]
    ok = abs(v - ref) <= TOL; ok_all &= ok
    A["checks"].append({"what": f"p0.7 fraudF1 {m}", "rerun": v, "ref": ref,
                        "diff": round(v - ref, 4), "ok": ok})
    print(f"p0.7  fraudF1 {m:14s} {v:8.4f} {ref:8.4f} {v-ref:+9.4f}  {'OK' if ok else 'DIFF'}")
A["consistent"] = ok_all
print(f"overall: {'CONSISTENT' if ok_all else 'PARTIAL - see DIFF rows'}")
print("note: original archived results.json T3/p0.7 for reference:",
      {m: REF_ORIG["T3_robust"]["0.7"][m]["posF1"] for m in ["M1", "M2", "M3", "M4"]})

print()
print("=" * 78)
print("B. M4b vs M2 vs M4 (seed 42)")
print("=" * 78)
print(f"{'cond':7s} {'M2.macroF1':>10s} {'M4.macroF1':>10s} {'M4b.macroF1':>11s} | "
      f"{'M2.fraudF1':>10s} {'M4.fraudF1':>10s} {'M4b.fraudF1':>11s} | M4b-M2 macroF1")
Btab = {}
for c in CONDS:
    r = {m: B["42"][c][m] for m in ["M2", "M4", "M4b"]}
    Btab[c] = r
    d = r["M4b"]["macroF1"] - r["M2"]["macroF1"]
    print(f"{c:7s} {r['M2']['macroF1']:10.4f} {r['M4']['macroF1']:10.4f} {r['M4b']['macroF1']:11.4f} | "
          f"{r['M2']['fraudF1']:10.4f} {r['M4']['fraudF1']:10.4f} {r['M4b']['fraudF1']:11.4f} | {d:+.4f}")

print()
print("=" * 78)
print("C. 5-SEED VARIANCE (mean +/- std, n=5)  macroF1 / fraudF1")
print("=" * 78)
Csum = {}
for m in MODELS:
    Csum[m] = {}
    for c in CONDS:
        mf = [B[s][c][m]["macroF1"] for s in SEEDS]
        ff = [B[s][c][m]["fraudF1"] for s in SEEDS]
        Csum[m][c] = {"macroF1_mean": round(st.mean(mf), 4), "macroF1_std": round(st.stdev(mf), 4),
                      "fraudF1_mean": round(st.mean(ff), 4), "fraudF1_std": round(st.stdev(ff), 4),
                      "macroF1_runs": mf, "fraudF1_runs": ff}
        print(f"{m:4s} {c:6s} macroF1 {st.mean(mf):.4f}+/-{st.stdev(mf):.4f}   "
              f"fraudF1 {st.mean(ff):.4f}+/-{st.stdev(ff):.4f}")

print()
print("=" * 78)
print("D. 13-CLASS (fine) 5-SEED  macroF1 mean +/- std")
print("=" * 78)
Dsum = {}
F = R.get("fine13", {})
for m in ["M1", "M4"]:
    Dsum[m] = {}
    for c in ["clean", "p0.3"]:
        vals = [F[s][c][m]["macroF1"] for s in SEEDS if s in F and c in F[s] and m in F[s][c]]
        if len(vals) == 5:
            Dsum[m][c] = {"mean": round(st.mean(vals), 4), "std": round(st.stdev(vals), 4), "runs": vals, "n": 5}
            print(f"{m:4s} {c:6s} {st.mean(vals):.4f} +/- {st.stdev(vals):.4f}   runs={vals}")
        else:
            Dsum[m][c] = {"n": len(vals), "runs": vals,
                          "note": "incomplete - M4 fine LR fit exceeds 300s sandbox limit" if m == "M4" else "incomplete"}
            print(f"{m:4s} {c:6s} incomplete (n={len(vals)})")
Dsum["M4_reference_from_results_json"] = {
    "clean_macroF1": REF_ORIG["T2_fine"]["M4"]["clean"]["macroF1"],
    "p0.3_macroF1": REF_ORIG["T2_fine"]["M4"]["p0.3"]["macroF1"],
    "note": "seed-42 values archived in results.json (Aug 2 original run); not re-run in v5"}
print(f"M4   reference (results.json, seed42): clean {REF_ORIG['T2_fine']['M4']['clean']['macroF1']}, "
      f"p0.3 {REF_ORIG['T2_fine']['M4']['p0.3']['macroF1']}")

print()
print("=" * 78)
print("TIMING (mean over seeds)  train_s / pred_ms_per_msg(clean)")
print("=" * 78)
Tsum = {}
for m in MODELS:
    tr = [B[s]["train_s"][m] for s in SEEDS]
    pr = [B[s]["clean"][m]["pred_ms_per_msg"] for s in SEEDS]
    p7 = [B[s]["p0.7"][m]["pred_ms_per_msg"] for s in SEEDS]
    Tsum[m] = {"train_s_mean": round(st.mean(tr), 2),
               "pred_ms_per_msg_clean": round(st.mean(pr), 3),
               "pred_ms_per_msg_p0.7": round(st.mean(p7), 3)}
    print(f"{m:4s} train {st.mean(tr):6.2f}s | infer clean {st.mean(pr):.3f} ms/msg | "
          f"p0.7 {st.mean(p7):.3f} ms/msg  (~{2792/st.mean([B[s]['clean'][m]['pred_s'] for s in SEEDS]):,.0f} msg/s)")
m1f = [F[s]["train_s"]["M1"] for s in SEEDS]
print(f"M1-fine13 train {st.mean(m1f):.2f}s ; M4-fine13 train >300s (sandbox limit, not completed)")

R["analysis"] = {"A_reproduction": A, "B_M4b_vs_M2_M4": Btab,
                 "C_summary_5seed": Csum, "D_fine13": Dsum, "timing": Tsum}
tmp = P + ".tmp"
with open(tmp, "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
os.replace(tmp, P)
print(f"\nanalysis appended -> {P}")
