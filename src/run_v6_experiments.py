# -*- coding: utf-8 -*-
"""
run_v6_experiments.py
ScriptGuard supplementary experiments, round 2.

Task 1 (--task profiles): attack profiles at p=0.5 (structural-only /
destructive-only) for M1/M4 on seeds 42-46, per-seed check whether
M4(structural) macroF1 == M4(clean) macroF1 exactly, plus the M0 keyword
filter (blacklist_predict from run_experiments.py) on clean and p=0.7.

Task 2 (--task fineM4): 13-class M4 with a configurable LR solver
(--max-iter, --ovr) so a single fit fits inside the 300 s sandbox limit.
One seed per invocation; calibration mode (--conds clean) compares against
the archived seed-42 value 0.9506 from results.json (max_iter=2000 run).

All results are appended incrementally to ../results/v5_seeds.json under
new keys: "profiles" (task 1) and "fine13_m4" (task 2). Nothing existing
is overwritten.
"""
import os, json, sys, time, warnings, argparse

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants, rules
from run_v5_experiments import (load, bin_label, metrics, build_model,
                                perturbed, make_tfidf, rule_matrix)

import numpy as np
from scipy.sparse import hstack
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.exceptions import ConvergenceWarning

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_JSON = os.path.join(HERE, "..", "results", "v5_seeds.json")

# task-1 profiles (user-specified composition)
STRUCT = ["symbol_insert", "traditional", "lookalike", "fullwidth", "emoji_noise"]
DESTRUCT = ["homophone", "pinyin_init", "deletion"]

# original run_experiments.py T5 口径: structural = 4 types, lookalike -> destructive
STRUCT4 = ["symbol_insert", "traditional", "fullwidth", "emoji_noise"]

ARCHIVE_M4_FINE42_CLEAN = 0.9506   # results.json T2_fine M4 clean macroF1 (max_iter=2000)

def load_results():
    with open(OUT_JSON, encoding="utf-8") as f:
        return json.load(f)

def save_results(R):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)

def fraud_pr(y_true, y_pred):
    P, R_, F, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label="fraud", zero_division=0)
    return {"fraudP": round(P, 4), "fraudR": round(R_, 4), "fraudF1": round(F, 4)}

def setup_seed(X, ytop, seed):
    """Replicates the v5/paper protocol: stratified split, train-only
    homophone index, rule-library keyword set. Returns everything needed."""
    ybin = [bin_label(t) for t in ytop]
    idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2,
                                      random_state=seed, stratify=ytop)
    variants.build_homophone_index(set("".join(X[i] for i in idx_tr)))
    kw = set()
    for plist in rules.SCRIPT_RULES.values():
        kw.update(p for p, _ in plist)
    kw.update(["微信", "客服", "转账", "汇款", "密码", "验证码", "点击", "登陆", "下载"])
    variants.set_keywords(kw)
    Xtr = [X[i] for i in idx_tr]; Xte = [X[i] for i in idx_te]
    ytr_bin = [ybin[i] for i in idx_tr]; yte_bin = [ybin[i] for i in idx_te]
    return kw, Xtr, Xte, ytr_bin, yte_bin, idx_tr, idx_te

# ---------------- task 1 ----------------
def run_profiles(seeds):
    X, ytop, _ = load()
    R = load_results()
    blk = R.setdefault("profiles", {})
    blk["_meta"] = {"structural_types": STRUCT, "destructive_types": DESTRUCT,
                    "p": 0.5, "M0": "keyword blacklist (any hit -> fraud), clean + mixed p=0.7",
                    "attack_seed": "same as split seed"}
    for seed in seeds:
        kw, Xtr, Xte, ytr, yte, _, _ = setup_seed(X, ytop, seed)
        print(f"\n=== profiles seed {seed} ===", flush=True)
        m1 = build_model("M1"); t0 = time.time(); m1.fit(Xtr, ytr)
        t1s = time.time() - t0
        m4 = build_model("M4"); t0 = time.time(); m4.fit(Xtr, ytr)
        t4s = time.time() - t0
        print(f"  trained M1 {t1s:.1f}s, M4 {t4s:.1f}s", flush=True)

        row = {"train_s": {"M1": round(t1s, 3), "M4": round(t4s, 3)}}
        # clean reference (full precision kept for exact-equality check)
        pred4_clean = m4.predict(Xte)
        pred1_clean = m1.predict(Xte)
        m4c = metrics(yte, pred4_clean, pos_label="fraud")
        m1c = metrics(yte, pred1_clean, pos_label="fraud")
        row["clean"] = {"M1": m1c, "M4": m4c}
        # cross-check against v5 binary block
        stored = R["binary"][str(seed)]["clean"]
        row["clean_matches_v5"] = {
            "M1": stored["M1"]["macroF1"] == m1c["macroF1"],
            "M4": stored["M4"]["macroF1"] == m4c["macroF1"]}

        for pname, types in [("structural", STRUCT), ("destructive", DESTRUCT)]:
            Xev = perturbed(Xte, 0.5, seed, types=types)
            pr = {}
            for key, m in [("M1", m1), ("M4", m4)]:
                t0 = time.time(); pred = m.predict(Xev); ps = time.time() - t0
                mt = metrics(yte, pred, pos_label="fraud")
                mt["pred_s"] = round(ps, 3)
                pr[key] = mt
            row[pname] = pr
            print(f"  {pname:12s} M1 macroF1={pr['M1']['macroF1']}  "
                  f"M4 macroF1={pr['M4']['macroF1']}", flush=True)

        # exact-equality claim: M4 structural == M4 clean (per seed)
        eq_rounded = row["structural"]["M4"]["macroF1"] == m4c["macroF1"]
        from sklearn.metrics import precision_recall_fscore_support as _prf
        f_exact_clean = _prf(yte, pred4_clean, average="macro", zero_division=0)[2]
        f_exact_struct = _prf(yte, m4.predict(perturbed(Xte, 0.5, seed, types=STRUCT)),
                              average="macro", zero_division=0)[2]
        row["M4_struct_eq_clean"] = {"rounded4_equal": eq_rounded,
                                     "clean_macroF1": m4c["macroF1"],
                                     "struct_macroF1": row["structural"]["M4"]["macroF1"],
                                     "exact_diff": round(float(f_exact_struct - f_exact_clean), 8)}
        print(f"  M4 struct==clean(4dp)? {eq_rounded}  exact diff {f_exact_struct - f_exact_clean:+.6f}",
              flush=True)

        # M0 keyword filter
        kw_list = list(kw)
        def m0(Xev):
            return ["fraud" if any(k in t for k in kw_list) else "ad" for t in Xev]
        row["M0_clean"] = fraud_pr(yte, m0(Xte))
        row["M0_p0.7"] = fraud_pr(yte, m0(perturbed(Xte, 0.7, seed)))
        print(f"  M0 clean R={row['M0_clean']['fraudR']}  p0.7 R={row['M0_p0.7']['fraudR']}",
              flush=True)

        blk[str(seed)] = row
        save_results(R)
        print(f"  seed {seed} saved", flush=True)
    return R

# ---------------- task 1 rerun: original 4-type structural profile ----------------
def run_struct4(seeds):
    """Structural-only profile with the ORIGINAL run_experiments.py T5
    composition (4 types, no lookalike) at p=0.5 -> key "profiles_struct4"."""
    from sklearn.metrics import precision_recall_fscore_support as _prf
    X, ytop, _ = load()
    R = load_results()
    blk = R.setdefault("profiles_struct4", {})
    blk["_meta"] = {"structural_types": STRUCT4,
                    "note": "original T5 composition; lookalike belongs to destructive",
                    "p": 0.5, "attack_seed": "same as split seed"}
    for seed in seeds:
        kw, Xtr, Xte, ytr, yte, _, _ = setup_seed(X, ytop, seed)
        print(f"\n=== struct4 seed {seed} ===", flush=True)
        m1 = build_model("M1"); t0 = time.time(); m1.fit(Xtr, ytr); t1s = time.time() - t0
        m4 = build_model("M4"); t0 = time.time(); m4.fit(Xtr, ytr); t4s = time.time() - t0
        print(f"  trained M1 {t1s:.1f}s, M4 {t4s:.1f}s", flush=True)
        row = {"train_s": {"M1": round(t1s, 3), "M4": round(t4s, 3)}}
        pred1_clean = m1.predict(Xte); pred4_clean = m4.predict(Xte)
        row["clean"] = {"M1": metrics(yte, pred1_clean, pos_label="fraud"),
                        "M4": metrics(yte, pred4_clean, pos_label="fraud")}
        Xev = perturbed(Xte, 0.5, seed, types=STRUCT4)
        pr = {}
        for key, m in [("M1", m1), ("M4", m4)]:
            t0 = time.time(); pred = m.predict(Xev); ps = time.time() - t0
            mt = metrics(yte, pred, pos_label="fraud"); mt["pred_s"] = round(ps, 3)
            pr[key] = mt
        row["structural"] = pr
        f_clean = _prf(yte, pred4_clean, average="macro", zero_division=0)[2]
        f_struct = _prf(yte, m4.predict(Xev), average="macro", zero_division=0)[2]
        row["M4_struct_eq_clean"] = {
            "rounded4_equal": pr["M4"]["macroF1"] == row["clean"]["M4"]["macroF1"],
            "clean_macroF1": row["clean"]["M4"]["macroF1"],
            "struct_macroF1": pr["M4"]["macroF1"],
            "exact_diff": round(float(f_struct - f_clean), 8)}
        print(f"  structural M1={pr['M1']['macroF1']}  M4={pr['M4']['macroF1']}  "
              f"M4==clean? {row['M4_struct_eq_clean']['rounded4_equal']} "
              f"(exact diff {f_struct - f_clean:+.8f})", flush=True)
        blk[str(seed)] = row
        save_results(R)
        print(f"  seed {seed} saved", flush=True)
    return R

# ---------------- task 2 ----------------
class M4Fine:
    """M4 (normalize + TF-IDF + 13 rule features) with configurable LR."""
    def __init__(self, max_iter=500, ovr=False):
        self.vec = make_tfidf()
        base = LogisticRegression(C=4.0, max_iter=max_iter)
        self.clf = OneVsRestClassifier(base, n_jobs=-1) if ovr else base
        self.ovr = ovr
        self.max_iter = max_iter
        self.conv_warn = False
        self.n_iter_ = None
        self.name = f"NormTFIDF+Rules+LR(mi={max_iter}{',ovr' if ovr else ''})"
    def _prep(self, X):
        return [variants.normalize(t) for t in X]
    def fit(self, X, y):
        Xp = self._prep(X)
        S = hstack([self.vec.fit_transform(Xp), rule_matrix(Xp)]).tocsr()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            self.clf.fit(S, y)
        self.conv_warn = any(issubclass(x.category, ConvergenceWarning) for x in w)
        if self.ovr:
            self.n_iter_ = [int(e.n_iter_[0]) for e in self.clf.estimators_]
        else:
            self.n_iter_ = [int(v) for v in np.atleast_1d(self.clf.n_iter_)]
        return self
    def predict(self, X):
        Xp = self._prep(X)
        S = hstack([self.vec.transform(Xp), rule_matrix(Xp)]).tocsr()
        return self.clf.predict(S)

def run_fine_m4(seeds, conds, max_iter, ovr):
    X, ytop, yfine = load()
    R = load_results()
    blk = R.setdefault("fine13_m4", {})
    blk["_meta"] = {"model": "normalize + TF-IDF(char2-4) + 13 rule feats + LR(C=4.0)",
                    "max_iter": max_iter, "ovr": bool(ovr),
                    "archive_ref_seed42_clean": ARCHIVE_M4_FINE42_CLEAN,
                    "archive_ref_note": "results.json T2_fine M4 (max_iter=2000, original run)"}
    for seed in seeds:
        kw, Xtr, Xte, _, _, idx_tr, idx_te = setup_seed(X, ytop, seed)
        ytr = [yfine[i] for i in idx_tr]; yte = [yfine[i] for i in idx_te]
        print(f"\n=== fineM4 seed {seed} (max_iter={max_iter}, ovr={ovr}) ===", flush=True)
        m = M4Fine(max_iter=max_iter, ovr=ovr)
        t0 = time.time(); m.fit(Xtr, ytr); ts = time.time() - t0
        row = {"train_s": round(ts, 3), "convergence_warning": m.conv_warn,
               "n_iter": m.n_iter_, "config": {"max_iter": max_iter, "ovr": bool(ovr)}}
        print(f"  trained in {ts:.1f}s  conv_warn={m.conv_warn}  n_iter={m.n_iter_}", flush=True)
        for cond in conds:
            Xev = Xte if cond == "clean" else perturbed(Xte, float(cond[1:]), seed)
            t0 = time.time(); pred = m.predict(Xev); ps = time.time() - t0
            mt = metrics(yte, pred); mt["pred_s"] = round(ps, 3)
            row[cond] = mt
            print(f"  {cond:6s} macroF1={mt['macroF1']} (pred {ps:.1f}s)", flush=True)
            if seed == 42 and cond == "clean":
                d = mt["macroF1"] - ARCHIVE_M4_FINE42_CLEAN
                row["clean_vs_archive"] = {"archive": ARCHIVE_M4_FINE42_CLEAN, "diff": round(d, 4),
                                           "within_0.003": abs(d) <= 0.003}
                print(f"  vs archive 0.9506: diff {d:+.4f}", flush=True)
        blk[str(seed)] = row
        save_results(R)
        print(f"  seed {seed} saved", flush=True)
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["profiles", "fineM4", "struct4"], required=True)
    ap.add_argument("--seeds", default="42,43,44,45,46")
    ap.add_argument("--conds", default="")
    ap.add_argument("--max-iter", type=int, default=500)
    ap.add_argument("--ovr", action="store_true")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    t_all = time.time()
    if args.task == "profiles":
        run_profiles(seeds)
    elif args.task == "struct4":
        run_struct4(seeds)
    else:
        conds = [c.strip() for c in args.conds.split(",") if c.strip()] or ["clean", "p0.3"]
        run_fine_m4(seeds, conds, args.max_iter, args.ovr)
    print(f"\ndone in {time.time()-t_all:.1f}s -> {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
