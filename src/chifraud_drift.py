# -*- coding: utf-8 -*-
"""chifraud_drift.py -- reproduces Table 7 of the paper.

Out-of-time evaluation of the whole model family on ChiFraud, a long-term
Chinese fraud-text benchmark. Every model is fitted once on ChiFraud_train and
then applied unchanged to the 2022 and 2023 out-of-time slices, so the only
thing that varies between rows is the model.

  word   word-level TF-IDF (50k)             + LogReg(C=1)   reference row
  M1     char 2-4g TF-IDF (min_df=3, 60k)    + LogReg(C=4)   == run_experiments.py
  M2     char 2-4g TF-IDF                    + LinearSVC(C=1)
  M3     rule library only                   (src/rules.py)
  M4     normalise + char TF-IDF + rule feats + LogReg(C=4)   == ScriptGuard

The benchmark's own terms do not allow redistribution, so the corpus is not
shipped with this repository. Point the script at a local checkout of it:

    export CHIFRAUD_DIR=/path/to/ChiFraud/dataset
    python src/chifraud_drift.py                      # Table 7
    python src/chifraud_drift.py --paired             # + McNemar / bootstrap

Expected directory contents: ChiFraud_train.csv, ChiFraud_t2022.csv,
ChiFraud_t2023.csv, each tab-separated with the header ``Label_id<TAB>Text``
and label 0 = normal, 1..10 = fraud categories.

Runtime is dominated by fitting the three TF-IDF views on the training split
(word, raw char, normalised char): about 12 minutes on a single CPU core, no
GPU needed.

Memory note: only the training split is held in memory. Each out-of-time slice
is read, transformed by the three already-fitted vectorisers and released
before the next one is opened, which keeps the peak comfortably under 2 GB and
makes the script safe to run next to a normal desktop session.
"""
import argparse
import csv
import faulthandler
import json
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, hstack
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.svm import LinearSVC

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rules                     # noqa: E402
import variants                  # noqa: E402

SPLITS = [("train", "ChiFraud_train.csv"),
          ("t2022", "ChiFraud_t2022.csv"),
          ("t2023", "ChiFraud_t2023.csv")]

MODELS = ("word_tfidf_lr", "M1_char_lr", "M2_char_svc", "M3_rule_only",
          "M4_scriptguard")


def log(*a):
    print(*a)
    sys.stdout.flush()


def dataset_dir(cli):
    d = cli or os.environ.get("CHIFRAUD_DIR") or os.path.join(
        HERE, "..", "data", "chifraud")
    d = os.path.abspath(d)
    missing = [f for _, f in SPLITS if not os.path.exists(os.path.join(d, f))]
    if missing:
        sys.exit("ChiFraud CSV files not found in %s\n  missing: %s\n"
                 "Pass --chifraud-dir DIR or set CHIFRAUD_DIR. The corpus is "
                 "not redistributed with this repository; obtain it from the "
                 "ChiFraud release and cite Tang et al. (COLING 2025)."
                 % (d, ", ".join(missing)))
    return d


def load(d, name):
    X, y = [], []
    with open(os.path.join(d, name), encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if len(row) < 2:
                continue
            X.append(row[1])
            y.append(0 if int(float(row[0])) == 0 else 1)
    return X, np.array(y, dtype=np.int8)


def rule_matrix(texts):
    """Dense rule features; values are integers, so float32 is lossless."""
    return csr_matrix(np.array([rules.rule_features(t) for t in texts],
                               dtype=np.float32))


def m3_predict(texts):
    """The symbolic layer alone: 1 when the top rule fires on fraud."""
    out, marg = [], []
    for s in texts:
        top, mm = rules.rule_predict_top(s)
        out.append(1 if top in ("FR", "IL") else 0)
        marg.append(mm)
    return np.array(out), np.array(marg, dtype=float)


def score(y, pred, prob):
    """Fraud is the positive class; both F1 flavours are reported."""
    P, R, F, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                 pos_label=1, zero_division=0)
    return {"acc": round(float(accuracy_score(y, pred)), 4),
            "P": round(float(P), 4), "R": round(float(R), 4),
            "F1_fraud": round(float(F), 4),
            "macroF1": round(float(f1_score(y, pred, average="macro",
                                            zero_division=0)), 4),
            "AUC": round(float(roc_auc_score(y, prob)), 4),
            "n": int(len(y)), "n_fraud": int(y.sum())}


def mcnemar(y, pa, pb):
    """b: pa right and pb wrong; c: the reverse. Continuity-corrected."""
    a_ok, b_ok = (pa == y), (pb == y)
    b = int(np.sum(a_ok & ~b_ok))
    c = int(np.sum(~a_ok & b_ok))
    if b + c == 0:
        return {"b": 0, "c": 0, "chi2": 0.0, "p": 1.0}
    stat = (abs(b - c) - 1) ** 2 / float(b + c)
    return {"b": b, "c": c, "chi2": round(stat, 3),
            "p": round(float(1 - chi2.cdf(stat, 1)), 6)}


def main():
    faulthandler.enable()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--chifraud-dir", default=None,
                    help="directory holding the ChiFraud CSV splits")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results",
                                                  "chifraud_drift.json"))
    ap.add_argument("--paired", action="store_true",
                    help="also run McNemar and a bootstrap CI on the 2023 slice")
    ap.add_argument("--max-iter", type=int, default=2000,
                    help="LogReg iteration budget (default 2000, the paper's "
                         "setting; it does not fully converge on this corpus)")
    cli = ap.parse_args()

    d = dataset_dir(cli.chifraud_dir)
    t0 = time.time()

    results = {name: {} for name in MODELS}
    fires = {}
    preds23 = {}

    # ================= training split ======================================
    raw, ytr = load(d, "ChiFraud_train.csv")
    log("train %d rows (fraud %d)  %.1fs" % (len(raw), ytr.sum(),
                                             time.time() - t0))
    ntr = [variants.normalize(x) for x in raw]
    Rtr = rule_matrix(ntr)

    # M3 reads raw text, so it has to run before the training text is released
    pred, marg = m3_predict(raw)
    results["M3_rule_only"]["train"] = score(ytr, pred, marg)
    fires["train"] = int(np.sum(np.asarray(Rtr.todense())[:, :-1] > 0))
    del pred, marg
    log("M3 rule-only (train): F1=%.4f  rule fires on %d/%d"
        % (results["M3_rule_only"]["train"]["F1_fraud"], fires["train"],
           len(raw)))

    # ---- three feature views, all fitted on the training split only -------
    t = time.time()
    vw = TfidfVectorizer(max_features=50000)
    Aw = vw.fit_transform(raw)
    log("word vectoriser            %s  %.1fs" % (Aw.shape, time.time() - t))

    t = time.time()
    vc = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                         max_features=60000)
    Ac = vc.fit_transform(raw)
    log("char vectoriser (raw)      %s  %.1fs" % (Ac.shape, time.time() - t))

    t = time.time()
    vn = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3,
                         max_features=60000)
    An = vn.fit_transform(ntr)
    log("char vectoriser (norm)     %s  %.1fs" % (An.shape, time.time() - t))

    # every view is fitted, so the training text itself can go
    del raw, ntr

    S_tr = hstack([An, Rtr]).tocsr()
    del An

    # ---- the five models, all fitted on the training split only -----------
    t = time.time()
    m_word = LogisticRegression(C=1.0, max_iter=2000).fit(Aw, ytr)
    m1 = LogisticRegression(C=4.0, max_iter=2000).fit(Ac, ytr)
    m2 = LinearSVC(C=1.0).fit(Ac, ytr)
    m4 = LogisticRegression(C=4.0, max_iter=cli.max_iter).fit(S_tr, ytr)
    log("four fitted models         %.1fs" % (time.time() - t))
    converged = int(m4.n_iter_[0]) < cli.max_iter

    results["word_tfidf_lr"]["train"] = score(
        ytr, m_word.predict(Aw), m_word.predict_proba(Aw)[:, 1])
    results["M1_char_lr"]["train"] = score(
        ytr, m1.predict(Ac), m1.predict_proba(Ac)[:, 1])
    results["M2_char_svc"]["train"] = score(
        ytr, m2.predict(Ac), m2.decision_function(Ac))
    results["M4_scriptguard"]["train"] = score(
        ytr, m4.predict(S_tr), m4.predict_proba(S_tr)[:, 1])
    del Aw, Ac, S_tr

    # ================= out-of-time slices, one at a time ==================
    for tag, fn in SPLITS[1:]:
        t = time.time()
        X, y = load(d, fn)
        Xn = [variants.normalize(x) for x in X]
        Rn = rule_matrix(Xn)

        pred, marg = m3_predict(X)
        results["M3_rule_only"][tag] = score(y, pred, marg)
        fires[tag] = int(np.sum(np.asarray(Rn.todense())[:, :-1] > 0))
        del pred, marg

        A_w = vw.transform(X)
        A_c = vc.transform(X)
        A_n = vn.transform(Xn)
        del X, Xn

        S = hstack([A_n, Rn]).tocsr()
        del A_n, Rn

        results["word_tfidf_lr"][tag] = score(
            y, m_word.predict(A_w), m_word.predict_proba(A_w)[:, 1])
        results["M1_char_lr"][tag] = score(
            y, m1.predict(A_c), m1.predict_proba(A_c)[:, 1])
        results["M2_char_svc"][tag] = score(
            y, m2.predict(A_c), m2.decision_function(A_c))
        results["M4_scriptguard"][tag] = score(
            y, m4.predict(S), m4.predict_proba(S)[:, 1])

        if tag == "t2023":
            # only this slice is needed for the paired tests
            y23 = y
            preds23["M1"] = m1.predict(A_c)
            preds23["M2"] = m2.predict(A_c)
            preds23["M4"] = m4.predict(S)
        del A_w, A_c, S

        log("%-6s %d rows (fraud %d)  %-28s %.1fs"
            % (tag, len(y), y.sum(),
               "F1: " + " ".join("%.3f" % results[n][tag]["F1_fraud"]
                                 for n in MODELS), time.time() - t))

    if not converged:
        log("NOTE: LogReg on the M4 design matrix did not converge within %d "
            "iterations; raise --max-iter for a converged fit." % cli.max_iter)

    # ================= summary ============================================
    summary = {}
    for name, r in results.items():
        base = r["train"]["F1_fraud"]
        summary[name] = {
            "F1_train": base, "F1_t2022": r["t2022"]["F1_fraud"],
            "F1_t2023": r["t2023"]["F1_fraud"],
            "drift_t2023": round(base - r["t2023"]["F1_fraud"], 4),
            "recall_fall_t2023": round(r["train"]["R"] - r["t2023"]["R"], 4)}

    log("\n%-18s %9s %9s %9s %8s" % ("model", "F1_train", "F1_2022",
                                     "F1_2023", "Drift"))
    for name, s in summary.items():
        log("%-18s %9.4f %9.4f %9.4f %8.4f"
            % (name, s["F1_train"], s["F1_t2022"], s["F1_t2023"],
               s["drift_t2023"]))

    out = {"protocol": "fit on ChiFraud_train; out-of-time t2022 / t2023",
           "per_split": results, "drift_summary": summary,
           "rule_fires_rows_M3": fires,
           "M4_converged": converged,
           "max_iter": cli.max_iter,
           "elapsed_s": round(time.time() - t0, 1)}

    if cli.paired:
        out["mcnemar_M4_vs_M1"] = mcnemar(y23, preds23["M4"], preds23["M1"])
        out["mcnemar_M4_vs_M2"] = mcnemar(y23, preds23["M4"], preds23["M2"])
        rng = np.random.default_rng(42)
        idx = np.arange(len(y23))
        boot = {k: [] for k in ("M1", "M2", "M4")}
        for _ in range(500):
            s = rng.choice(idx, size=len(idx), replace=True)
            for k in boot:
                boot[k].append(float(precision_recall_fscore_support(
                    y23[s], preds23[k][s], average="binary",
                    pos_label=1, zero_division=0)[2]))
        out["t2023_bootstrap95"] = {
            k: [round(float(np.percentile(v, 2.5)), 4),
                round(float(np.percentile(v, 97.5)), 4)]
            for k, v in boot.items()}
        log("\nMcNemar M4 vs M1: %s" % out["mcnemar_M4_vs_M1"])
        log("McNemar M4 vs M2: %s" % out["mcnemar_M4_vs_M2"])
        log("bootstrap 95%% CI: %s" % out["t2023_bootstrap95"])

    dst = os.path.abspath(cli.out)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    log("\nwrote %s  (%.0f s)" % (dst, time.time() - t0))


if __name__ == "__main__":
    main()
