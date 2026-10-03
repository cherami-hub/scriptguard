# -*- coding: utf-8 -*-
"""
run_experiments.py
Full experiment suite for the IEEE CRESS 2026 paper.

T1  binary   : fraud diversion (FR+IL) vs non-fraud spam (AD)
T2  fine     : 13-class fraud-script classification (macro-F1)
T3  robust   : accuracy/F1 under adversarial variants p in {0.1,0.3,0.5}
T4  ablation : normalization / rule-feature contribution, per-type sensitivity

Models
  M1 TF-IDF(char 2-4g) + LogReg          (statistical baseline)
  M2 TF-IDF(char 2-4g) + LinearSVC       (statistical baseline)
  M3 Rule-only script library            (symbolic baseline)
  M4 Ours: normalize + TF-IDF + rule-features + LogReg  (neuro-symbolic)
"""
import os, csv, json, random, sys, time, warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants, rules

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.sparse import hstack, csr_matrix
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "fbs_messages.csv")
OUT  = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

SEED = 42
INTENSITIES = [0.3, 0.5, 0.7]   # keyword-targeted attack strength

# ---------------- data ----------------
def load():
    X, ytop, yfine = [], [], []
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(row["raw"]); ytop.append(row["top"]); yfine.append(row["fine"])
    return X, ytop, yfine

def bin_label(t):
    return "fraud" if t in ("FR", "IL") else "ad"

def metrics(y_true, y_pred, pos_label=None):
    acc = accuracy_score(y_true, y_pred)
    P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {"acc": round(acc, 4), "macroP": round(P, 4), "macroR": round(R, 4), "macroF1": round(F, 4)}
    if pos_label is not None:
        P2, R2, F2, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        out["posF1"] = round(F2, 4)
    return out

# ---------------- models ----------------
def make_tfidf():
    return TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3, max_features=60000)

def rule_matrix(texts):
    return csr_matrix(np.array([rules.rule_features(t) for t in texts], dtype=np.float32))

class M_TFIDF:
    def __init__(self, clf="lr"):
        self.vec = make_tfidf()
        self.clf = LogisticRegression(C=4.0, max_iter=2000) if clf == "lr" else LinearSVC(C=1.0)
        self.name = "TFIDF+LR" if clf == "lr" else "TFIDF+SVC"
    def fit(self, X, y):
        self.vec.fit(X); self.clf.fit(self.vec.transform(X), y); return self
    def predict(self, X):
        return self.clf.predict(self.vec.transform(X))

class M_RuleOnly:
    name = "RuleOnly"
    def fit(self, X, y): return self
    def predict(self, X):
        out = []
        for t in X:
            top, _ = rules.rule_predict_top(t)
            out.append(top)
        return out

class M_Ours:
    """normalize -> TF-IDF (+rule dense features) -> LogReg"""
    name = "NormTFIDF+Rules+LR"
    def __init__(self, use_norm=True, use_rules=True):
        self.use_norm, self.use_rules = use_norm, use_rules
        self.vec = make_tfidf()
        self.clf = LogisticRegression(C=4.0, max_iter=2000)
    def _prep(self, X):
        return [variants.normalize(t) for t in X] if self.use_norm else list(X)
    def fit(self, X, y):
        Xp = self._prep(X)
        S = self.vec.fit_transform(Xp)
        if self.use_rules:
            S = hstack([S, rule_matrix(Xp)]).tocsr()
        self.clf.fit(S, y); return self
    def predict(self, X):
        Xp = self._prep(X)
        S = self.vec.transform(Xp)
        if self.use_rules:
            S = hstack([S, rule_matrix(Xp)]).tocsr()
        return self.clf.predict(S)

# ---------------- experiment driver ----------------
def perturbed(X, p, seed, types=None):
    rng = random.Random(seed)
    return [variants.perturb_keywords(t, p, rng, types) for t in X]

def main():
    t0 = time.time()
    X, ytop, yfine = load()
    print(f"loaded {len(X)} messages | top={Counter(ytop)}")
    print(f"avg_len={sum(len(t) for t in X)/len(X):.1f}")

    # homophone index from corpus vocab (train-only chars to avoid leakage)
    idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2, random_state=SEED, stratify=ytop)
    n_py = variants.build_homophone_index(set("".join(X[i] for i in idx_tr)))
    print(f"homophone index: {n_py} pinyin groups")

    # keyword-target set for the evasion attack = rule-library vocabulary
    kw = set()
    for plist in rules.SCRIPT_RULES.values():
        kw.update(p for p, _ in plist)
    kw.update(["微信", "客服", "转账", "汇款", "密码", "验证码", "点击", "登陆", "下载"])
    n_kw = variants.set_keywords(kw)
    print(f"attack target keywords: {n_kw}")

    ybin = [bin_label(t) for t in ytop]
    Xtr = [X[i] for i in idx_tr]; Xte = [X[i] for i in idx_te]
    ytr_bin = [ybin[i] for i in idx_tr]; yte_bin = [ybin[i] for i in idx_te]
    ytr_top = [ytop[i] for i in idx_tr]; yte_top = [ytop[i] for i in idx_te]
    ytr_fine = [yfine[i] for i in idx_tr]; yte_fine = [yfine[i] for i in idx_te]

    R = {"dataset": {
            "total": len(X), "train": len(Xtr), "test": len(Xte),
            "top_dist": dict(Counter(ytop)),
            "fine_dist": dict(sorted(Counter(yfine).items())),
            "avg_len": round(sum(len(t) for t in X)/len(X), 1)}}

    # ---------- T1 clean binary ----------
    print("\n[T1] clean binary fraud-vs-ad")
    models_bin = {}
    for key, m in [("M1", M_TFIDF("lr")), ("M2", M_TFIDF("svc")),
                   ("M3", M_RuleOnly()), ("M4", M_Ours())]:
        if key == "M3":
            m.fit(Xtr, ytr_top)                      # rule-only predicts top labels
            pred_top = m.predict(Xte)
            pred = [bin_label(t) for t in pred_top]
        else:
            m.fit(Xtr, ytr_bin); pred = m.predict(Xte)
        models_bin[key] = m
        R.setdefault("T1_clean", {})[key] = metrics(yte_bin, pred, pos_label="fraud")
        print(f"  {m.name:22s} {R['T1_clean'][key]}")

    # ---------- T3 robustness (binary) ----------
    R["T3_robust"] = {}
    for p in INTENSITIES:
        print(f"\n[T3] perturbed test p={p}")
        Xte_p = perturbed(Xte, p, SEED)
        R["T3_robust"][str(p)] = {}
        for key in ["M1", "M2", "M4"]:
            pred = models_bin[key].predict(Xte_p)
            R["T3_robust"][str(p)][key] = metrics(yte_bin, pred, pos_label="fraud")
            print(f"  {models_bin[key].name:22s} {R['T3_robust'][str(p)][key]}")
        # rule-only on perturbed
        pred_top = models_bin["M3"].predict(Xte_p)
        pred = [bin_label(t) for t in pred_top]
        R["T3_robust"][str(p)]["M3"] = metrics(yte_bin, pred, pos_label="fraud")
        print(f"  {'RuleOnly':22s} {R['T3_robust'][str(p)]['M3']}")

    # ---------- T2 fine-grained 13-class ----------
    print("\n[T2] fine-grained 13-class (clean / p=0.3)")
    fine_models = {"M1": M_TFIDF("lr"), "M4": M_Ours()}
    R["T2_fine"] = {}
    Xte_p3 = perturbed(Xte, 0.3, SEED)
    for key, m in fine_models.items():
        m.fit(Xtr, ytr_fine)
        for cond, Xev in [("clean", Xte), ("p0.3", Xte_p3)]:
            pred = m.predict(Xev)
            R["T2_fine"].setdefault(key, {})[cond] = metrics(yte_fine, pred)
            print(f"  {m.name:22s} {cond:6s} {R['T2_fine'][key][cond]}")

    # ---------- T4 ablation (binary, p=0.3) ----------
    print("\n[T4] ablation at p=0.3 (binary)")
    R["T4_ablation"] = {}
    abl = {
        "full":        M_Ours(use_norm=True,  use_rules=True),
        "w/o_rules":   M_Ours(use_norm=True,  use_rules=False),
        "w/o_norm":    M_Ours(use_norm=False, use_rules=True),
        "neither(M1)": M_TFIDF("lr"),
    }
    for tag, m in abl.items():
        m.fit(Xtr, ytr_bin)
        for cond, Xev in [("clean", Xte), ("p0.3", Xte_p3)]:
            pred = m.predict(Xev)
            R["T4_ablation"].setdefault(tag, {})[cond] = metrics(yte_bin, pred, pos_label="fraud")
        print(f"  {tag:14s} clean={R['T4_ablation'][tag]['clean']['macroF1']}  p0.3={R['T4_ablation'][tag]['p0.3']['macroF1']}")

    # ---------- per-variant-type sensitivity (binary, p=0.3) ----------
    print("\n[T4b] per-variant-type sensitivity at p=0.3")
    R["T4b_types"] = {}
    for t in variants.PERTURBATION_TYPES:
        Xte_t = perturbed(Xte, 0.3, SEED, types=[t])
        r_m1 = metrics(yte_bin, abl["neither(M1)"].predict(Xte_t), pos_label="fraud")
        r_m4 = metrics(yte_bin, abl["full"].predict(Xte_t), pos_label="fraud")
        R["T4b_types"][t] = {"M1": r_m1, "M4": r_m4}
        print(f"  {t:14s} M1-F1={r_m1['macroF1']}  M4-F1={r_m4['macroF1']}")

    # ---------- T5: attack profiles + deployed keyword-filter collapse ----------
    print("\n[T5] attack profiles at p=0.5 (binary)")
    STRUCT = ["symbol_insert", "traditional", "fullwidth", "emoji_noise"]
    DESTRUCT = ["homophone", "pinyin_init", "deletion", "lookalike"]
    MIXED = variants.PERTURBATION_TYPES
    kw_list = list(kw)
    def blacklist_predict(Xev):
        out = []
        for t in Xev:
            hit = any(k in t for k in kw_list)
            out.append("fraud" if hit else "ad")
        return out
    R["T5_profiles"] = {}
    for pname, ptypes in [("structural", STRUCT), ("mixed", MIXED), ("destructive", DESTRUCT)]:
        Xte_pr = perturbed(Xte, 0.5, SEED, types=ptypes)
        row = {}
        row["M0_keyword_filter"] = metrics(yte_bin, blacklist_predict(Xte_pr), pos_label="fraud")
        row["M1"] = metrics(yte_bin, models_bin["M1"].predict(Xte_pr), pos_label="fraud")
        row["M3"] = metrics(yte_bin, [bin_label(t) for t in models_bin["M3"].predict(Xte_pr)], pos_label="fraud")
        row["M4"] = metrics(yte_bin, models_bin["M4"].predict(Xte_pr), pos_label="fraud")
        R["T5_profiles"][pname] = row
        print(f"  [{pname:11s}] M0-recall={row['M0_keyword_filter']['macroF1']} M1={row['M1']['macroF1']} M4={row['M4']['macroF1']}")
    # clean M0 for reference
    R["M0_clean"] = metrics(yte_bin, blacklist_predict(Xte), pos_label="fraud")
    print(f"  M0 keyword filter on clean text: {R['M0_clean']}")

    # ---------- T6: length-stratified robustness (mixed attack p=0.5) ----------
    print("\n[T6] length-stratified robustness at p=0.5 mixed")
    Xte_p5m = perturbed(Xte, 0.5, SEED, types=MIXED)
    pred_m1 = models_bin["M1"].predict(Xte_p5m)
    pred_m4 = models_bin["M4"].predict(Xte_p5m)
    R["T6_length"] = {}
    for lname, lo, hi in [("short(<=40)", 0, 40), ("mid(41-80)", 41, 80), ("long(>80)", 81, 10**9)]:
        idxs = [i for i, t in enumerate(Xte) if lo <= len(t) <= hi]
        if not idxs:
            continue
        ys = [yte_bin[i] for i in idxs]
        m1s = metrics(ys, [pred_m1[i] for i in idxs], pos_label="fraud")
        m4s = metrics(ys, [pred_m4[i] for i in idxs], pos_label="fraud")
        R["T6_length"][lname] = {"n": len(idxs), "M1": m1s, "M4": m4s}
        print(f"  {lname:14s} n={len(idxs):4d}  M1={m1s['macroF1']}  M4={m4s['macroF1']}")

    # ---------- sample adversarial examples (for paper figure) ----------
    rng = random.Random(SEED)
    samples = []
    fraud_idx = [i for i in range(len(Xte)) if yte_bin[i] == "fraud"][:3]
    for i in fraud_idx:
        v = variants.perturb_keywords(Xte[i], 0.5, rng)
        samples.append({"orig": Xte[i], "adv": v, "norm": variants.normalize(v), "label": yte_fine[i]})
    R["adv_samples"] = samples

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    print(f"\nresults -> {os.path.join(OUT, 'results.json')}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
