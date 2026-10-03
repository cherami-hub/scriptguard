# -*- coding: utf-8 -*-
"""
run_independent_vocab_attack.py
Independent vocabulary attack experiment for ScriptGuard (CRESS 2026).

Key idea: The existing keyword-targeted attack uses K (rule-library keywords).
This script creates K_independent -- a disjoint set of common fraud keywords
NOT in the rule library -- and compares model degradation under:
  (a) K-targeted attack (attacker knows the rule library)
  (b) K_independent-targeted attack (attacker targets general fraud vocab)

Models evaluated:
  M1  TF-IDF+LR (statistical baseline)
  M2  TF-IDF+SVC
  M3  Rule-only
  M4  ScriptGuard (neuro-symbolic)
  M5  BERT (Chinese-RoBERTa-wwm-ext, 3-seed agg from bert_results.json)

Hypothesis: ScriptGuard's structural immunity advantage is more pronounced
under K_independent attack because its rules are not directly targeted, while
BERT's distributed representation degrades similarly under both attacks.
"""
import os, sys, csv, json, random, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants, rules

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy.sparse import hstack, csr_matrix
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "fbs_messages.csv")
OUT  = os.path.join(HERE, "..", "results")
os.makedirs(OUT, exist_ok=True)

SEED = 42
INTENSITIES = [0.3, 0.5, 0.7]

def bin_label(t):
    return "fraud" if t in ("FR", "IL") else "ad"

def metrics(y_true, y_pred, pos_label=None):
    acc = accuracy_score(y_true, y_pred)
    P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    out = {"acc": round(acc, 4), "macroF1": round(F, 4)}
    if pos_label is not None:
        P2, R2, F2, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        out["fraudF1"] = round(F2, 4)
    return out

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

class M_Ours:
    name = "ScriptGuard"
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

# ---------------- Build K_independent ----------------
# Strategy: extract top-N frequent Chinese bigrams/words from fraud messages
# that are NOT in the rule-library keyword set K.
# These represent "general fraud vocabulary" that an independent attacker
# might target without knowing the specific rule library.

def build_independent_keywords(Xtr, ytr_top, rule_kw_set, n_keywords=80):
    """Extract top fraud-discriminative character bigrams NOT in rule_kw_set."""
    fraud_texts = [Xtr[i] for i in range(len(Xtr)) if ytr_top[i] in ("FR", "IL")]
    ad_texts = [Xtr[i] for i in range(len(Xtr)) if ytr_top[i] == "AD"]

    # Get frequent bigrams in fraud vs ad
    fraud_vec = CountVectorizer(analyzer="char", ngram_range=(2, 3), min_df=5)
    X_f = fraud_vec.fit_transform(fraud_texts)
    freq_f = np.array(X_f.sum(axis=0)).flatten()
    feat_f = fraud_vec.get_feature_names_out()

    ad_vec = CountVectorizer(analyzer="char", ngram_range=(2, 3), min_df=2)
    X_a = ad_vec.fit_transform(ad_texts)
    freq_a_dict = dict(zip(ad_vec.get_feature_names_out(), np.array(X_a.sum(axis=0)).flatten()))

    # Score: fraud frequency * (1 - normalized ad frequency) → fraud-discriminative
    scores = []
    for i, fw in enumerate(feat_f):
        if len(fw) < 2:
            continue
        # Skip if any part is in rule_kw_set (avoid overlap)
        if any(k in fw for k in rule_kw_set if len(k) >= 1) or fw in rule_kw_set:
            continue
        ad_freq = freq_a_dict.get(fw, 0)
        discriminative = freq_f[i] / (1 + ad_freq)
        scores.append((fw, discriminative))

    scores.sort(key=lambda x: -x[1])
    selected = [w for w, _ in scores[:n_keywords]]
    print(f"K_independent: {len(selected)} keywords selected from {len(scores)} candidates")
    return selected

# ---------------- Main experiment ----------------
def main():
    t0 = time.time()
    X, ytop, yfine = [], [], []
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(row["raw"]); ytop.append(row["top"]); yfine.append(row["fine"])

    idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2,
                                       random_state=SEED, stratify=ytop)

    # Build homophone index and original keyword set K
    variants.build_homophone_index(set("".join(X[i] for i in idx_tr)))

    kw = set()
    for plist in rules.SCRIPT_RULES.values():
        kw.update(p for p, _ in plist)
    kw.update(["微信", "客服", "转账", "汇款", "密码", "验证码", "点击", "登陆", "下载"])

    ybin = [bin_label(t) for t in ytop]
    Xtr = [X[i] for i in idx_tr]; Xte = [X[i] for i in idx_te]
    ytr_bin = [ybin[i] for i in idx_tr]; yte_bin = [ybin[i] for i in idx_te]
    ytr_top = [ytop[i] for i in idx_tr]

    # Build K_independent
    kw_independent = build_independent_keywords(Xtr, ytr_top, kw, n_keywords=80)
    print(f"K (rule-library): {len(kw)} keywords")
    print(f"K_independent:      {len(kw_independent)} keywords")
    overlap = kw & set(kw_independent)
    print(f"Overlap: {len(overlap)} (should be 0 or near 0)")

    # Train models
    models = {
        "M1": M_TFIDF("lr"),
        "M2": M_TFIDF("svc"),
        "M4": M_Ours(),
    }
    for key, m in models.items():
        m.fit(Xtr, ytr_bin)

    # Load BERT results
    bert_path = os.path.join(OUT, "bert_results.json")
    bert_agg = None
    if os.path.exists(bert_path):
        bert_agg = json.load(open(bert_path, encoding="utf-8"))["agg"]

    R = {"dataset": {"train": len(Xtr), "test": len(Xte)}}

    # ---------- Experiment 1: K-targeted attack (original, for reference) ----------
    print("\n=== K-targeted attack (original, rule-library keywords) ===")
    variants.set_keywords(kw)
    R["K_targeted"] = {}
    for p in INTENSITIES:
        rng = random.Random(SEED)
        Xte_p = [variants.perturb_keywords(t, p, rng) for t in Xte]
        row = {}
        for key, m in models.items():
            pred = m.predict(Xte_p)
            row[key] = metrics(yte_bin, pred, pos_label="fraud")
        R["K_targeted"][str(p)] = row
        print(f"  p={p}: M1={row['M1']['macroF1']} M2={row['M2']['macroF1']} M4={row['M4']['macroF1']}")

    # ---------- Experiment 2: K_independent-targeted attack ----------
    print("\n=== K_independent-targeted attack (independent fraud vocab) ===")
    variants.set_keywords(kw_independent)
    R["K_independent"] = {}
    for p in INTENSITIES:
        rng = random.Random(SEED)
        Xte_p = [variants.perturb_keywords(t, p, rng) for t in Xte]
        row = {}
        for key, m in models.items():
            pred = m.predict(Xte_p)
            row[key] = metrics(yte_bin, pred, pos_label="fraud")
        R["K_independent"][str(p)] = row
        print(f"  p={p}: M1={row['M1']['macroF1']} M2={row['M2']['macroF1']} M4={row['M4']['macroF1']}")

    # ---------- Compute degradation deltas ----------
    print("\n=== Degradation Delta (K_independent - K_targeted) ===")
    R["degradation"] = {}
    for p_str in [str(p) for p in INTENSITIES]:
        row = {}
        for key in ["M1", "M2", "M4"]:
            d_macro = (R["K_independent"][p_str][key]["macroF1"]
                       - R["K_targeted"][p_str][key]["macroF1"])
            d_fraud = (R["K_independent"][p_str][key]["fraudF1"]
                       - R["K_targeted"][p_str][key]["fraudF1"])
            row[key] = {"macroF1_delta": round(d_macro, 4), "fraudF1_delta": round(d_fraud, 4)}
        if bert_agg:
            b_k_p = bert_agg.get(f"p{p_str}", {})
            b_k_macro = b_k_p.get("macroF1_mean", 0)
            b_k_fraud = b_k_p.get("fraudF1_mean", 0)
            row["M5_BERT"] = {
                "macroF1_k_targeted": b_k_macro,
                "fraudF1_k_targeted": b_k_fraud,
                "note": "K_independent BERT to run separately"
            }
        R["degradation"][p_str] = row
        print(f"  p={p_str}: d_macro  M1={row['M1']['macroF1_delta']}  "
              f"M2={row['M2']['macroF1_delta']}  M4={row['M4']['macroF1_delta']}")

    # ---------- Experiment 3: Clean baseline ----------
    print("\n=== Clean baseline ===")
    variants.set_keywords(kw)  # restore
    R["clean"] = {}
    for key, m in models.items():
        pred = m.predict(Xte)
        R["clean"][key] = metrics(yte_bin, pred, pos_label="fraud")
    if bert_agg:
        R["clean"]["M5_BERT"] = {
            "macroF1": bert_agg["clean"]["macroF1_mean"],
            "fraudF1": bert_agg["clean"]["fraudF1_mean"],
        }

    # Save
    out_path = os.path.join(OUT, "independent_vocab_results.json")
    json.dump(R, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nResults -> {out_path} ({time.time()-t0:.1f}s)")
    print("\n=== Summary for Paper ===")
    print("Key finding: ScriptGuard (M4) should show SMALLER degradation under")
    print("K_independent attack because its rule-library features are not directly")
    print("targeted, while statistical models (M1/M2/BERT) degrade similarly under")
    print("both attacks because they depend on all character distributions.")

if __name__ == "__main__":
    main()
