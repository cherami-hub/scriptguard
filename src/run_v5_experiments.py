# -*- coding: utf-8 -*-
"""
run_v5_experiments.py
Supplementary experiments for ScriptGuard (IEEE CRESS 2026 submission).

  A. reproduction check at seed 42 (M1/M2/M3/M4, clean + p in {0.3,0.5,0.7})
  B. new ablation M4b: normalize + TF-IDF + rule-features + LinearSVC(C=1.0)
  C. 5-seed variance study (seeds 42..46, models M1,M2,M3,M4,M4b)
  D. 13-class fine-grained, 5 seeds (M1/M4, clean + p=0.3)

Reuses variants.py / rules.py unchanged. Results are written INCREMENTALLY
(after every seed x condition block) to ../results/v5_seeds.json (atomic
replace) so a timeout never loses completed work.

Usage:
  python run_v5_experiments.py --task bin  --seeds 42 --conds clean,p0.7
  python run_v5_experiments.py --task fine --seeds 42,43 --models M1,M4
"""
import os, csv, json, random, sys, time, warnings, argparse
from collections import Counter

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
OUT_DIR = os.path.join(HERE, "..", "results")
OUT_JSON = os.path.join(OUT_DIR, "v5_seeds.json")
os.makedirs(OUT_DIR, exist_ok=True)

INTENSITIES = [0.3, 0.5, 0.7]
ALL_MODELS = ["M1", "M2", "M3", "M4", "M4b"]

# ---------------- data / metrics (identical to run_experiments.py) ----------------
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
        _, _, F2, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", pos_label=pos_label, zero_division=0)
        out["fraudF1"] = round(F2, 4)
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
        return [rules.rule_predict_top(t)[0] for t in X]

class M_Ours:
    """normalize -> TF-IDF (+rule dense features) -> classifier.
    clf='lr'  -> M4 (paper);  clf='svc' -> M4b (new ablation)."""
    def __init__(self, clf="lr"):
        self.vec = make_tfidf()
        self.clf = LogisticRegression(C=4.0, max_iter=2000) if clf == "lr" else LinearSVC(C=1.0)
        self.name = "NormTFIDF+Rules+LR" if clf == "lr" else "NormTFIDF+Rules+SVC"
    def _prep(self, X):
        return [variants.normalize(t) for t in X]
    def fit(self, X, y):
        Xp = self._prep(X)
        S = self.vec.fit_transform(Xp)
        S = hstack([S, rule_matrix(Xp)]).tocsr()
        self.clf.fit(S, y); return self
    def predict(self, X):
        Xp = self._prep(X)
        S = self.vec.transform(Xp)
        S = hstack([S, rule_matrix(Xp)]).tocsr()
        return self.clf.predict(S)

def build_model(key):
    if key == "M1":  return M_TFIDF("lr")
    if key == "M2":  return M_TFIDF("svc")
    if key == "M3":  return M_RuleOnly()
    if key == "M4":  return M_Ours("lr")
    if key == "M4b": return M_Ours("svc")
    raise ValueError(key)

# ---------------- attack (identical protocol to run_experiments.py) ----------------
def perturbed(X, p, seed, types=None):
    rng = random.Random(seed)
    return [variants.perturb_keywords(t, p, rng, types) for t in X]

# ---------------- incremental result store ----------------
def load_results():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON, encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_results(R):
    tmp = OUT_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    os.replace(tmp, OUT_JSON)

# ---------------- per-seed driver ----------------
def run_seed(R, X, ytop, yfine, seed, conds, model_keys, task):
    ybin = [bin_label(t) for t in ytop]
    idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2, random_state=seed, stratify=ytop)

    # homophone index from TRAIN-only chars (same protocol as run_experiments.py)
    n_py = variants.build_homophone_index(set("".join(X[i] for i in idx_tr)))
    kw = set()
    for plist in rules.SCRIPT_RULES.values():
        kw.update(p for p, _ in plist)
    kw.update(["微信", "客服", "转账", "汇款", "密码", "验证码", "点击", "登陆", "下载"])
    n_kw = variants.set_keywords(kw)

    Xtr = [X[i] for i in idx_tr]; Xte = [X[i] for i in idx_te]
    if task == "bin":
        ytr = [ybin[i] for i in idx_tr]; yte = [ybin[i] for i in idx_te]
        block = R.setdefault("binary", {}).setdefault(str(seed), {})
    else:
        ytr = [yfine[i] for i in idx_tr]; yte = [yfine[i] for i in idx_te]
        block = R.setdefault("fine13", {}).setdefault(str(seed), {})

    block["meta"] = {"n_train": len(Xtr), "n_test": len(Xte),
                     "homophone_groups": n_py, "attack_keywords": n_kw,
                     "split_seed": seed, "attack_seed": seed}

    # ---- train each model exactly once for this split ----
    models = {}
    train_s = block.setdefault("train_s", {})
    for key in model_keys:
        m = build_model(key)
        t0 = time.time()
        m.fit(Xtr, ytr)
        train_s[key] = round(time.time() - t0, 3)
        models[key] = m
        print(f"  [seed {seed}] {key:4s} {m.name:22s} trained in {train_s[key]:8.2f}s", flush=True)
        save_results(R)   # flush after every single model fit

    # ---- evaluate per condition (perturb test set once per p, reused) ----
    for cond in conds:
        t0 = time.time()
        if cond == "clean":
            Xev = Xte
        else:
            p = float(cond[1:])
            Xev = perturbed(Xte, p, seed)
        gen_s = time.time() - t0
        row = block.setdefault(cond, {})
        for key in model_keys:
            if key in row:
                continue
            m = models[key]
            t1 = time.time()
            pred = m.predict(Xev)
            pred_s = time.time() - t1
            if key == "M3" and task == "bin":
                pred = [bin_label(t) for t in pred]
            mt = metrics(yte, pred, pos_label="fraud" if task == "bin" else None)
            mt["pred_s"] = round(pred_s, 3)
            mt["pred_ms_per_msg"] = round(1000.0 * pred_s / len(Xte), 4)
            row[key] = mt
            print(f"  [seed {seed}] {cond:6s} {key:4s} {m.name:22s} "
                  f"macroF1={mt['macroF1']}"
                  + (f" fraudF1={mt['fraudF1']}" if task == "bin" else "")
                  + f"  pred {pred_s:6.2f}s ({mt['pred_ms_per_msg']:.3f} ms/msg)", flush=True)
        row["_gen_s"] = round(gen_s, 3)
        save_results(R)   # incremental flush after every seed x condition
        print(f"  [seed {seed}] {cond:6s} done (attack-gen {gen_s:6.1f}s) -> saved", flush=True)
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["bin", "fine"], default="bin")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--conds", default="")   # default: bin->all four; fine->clean,p0.3
    ap.add_argument("--models", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s]
    if args.conds:
        conds = [c.strip() for c in args.conds.split(",") if c.strip()]
    else:
        conds = ["clean", "p0.7", "p0.3", "p0.5"] if args.task == "bin" else ["clean", "p0.3"]
    model_keys = [m.strip() for m in args.models.split(",") if m.strip()] or \
                 (ALL_MODELS if args.task == "bin" else ["M1", "M4"])

    t_all = time.time()
    X, ytop, yfine = load()
    print(f"loaded {len(X)} messages | top={Counter(ytop)}", flush=True)

    R = load_results()
    R.setdefault("meta", {
        "script": "run_v5_experiments.py",
        "data": os.path.basename(DATA), "n": len(X),
        "protocol": "train_test_split(test_size=0.2, stratify=ytop); "
                    "attack = keyword-targeted perturbation on TEST set only; "
                    "perturbation seed == split seed",
        "models": {"M1": "TFIDF(char2-4)+LR(C=4)", "M2": "TFIDF+LinearSVC(C=1)",
                   "M3": "RuleOnly", "M4": "Norm+TFIDF+RuleFeat+LR(C=4)",
                   "M4b": "Norm+TFIDF+RuleFeat+LinearSVC(C=1)"},
        "top_dist": dict(Counter(ytop)),
    })
    save_results(R)

    for seed in seeds:
        print(f"\n=== seed {seed} | task={args.task} | conds={conds} | models={model_keys} ===", flush=True)
        run_seed(R, X, ytop, yfine, seed, conds, model_keys, args.task)

    print(f"\nall done in {time.time()-t_all:.1f}s -> {OUT_JSON}", flush=True)

if __name__ == "__main__":
    main()
