# -*- coding: utf-8 -*-
"""extra_metrics.py — M0 keyword-filter fraud-class P/R/F1 across attack
strengths and profiles; merged into results.json (no model training)."""
import os, csv, json, random, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants, rules
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "fbs_messages.csv")
RES = os.path.join(HERE, "..", "results", "results.json")
SEED = 42

X, ytop = [], []
with open(DATA, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        X.append(row["raw"]); ytop.append(row["top"])
ybin = ["fraud" if t in ("FR", "IL") else "ad" for t in ytop]
idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2, random_state=SEED, stratify=ytop)
Xte = [X[i] for i in idx_te]; yte = [ybin[i] for i in idx_te]

variants.build_homophone_index(set("".join(X[i] for i in idx_tr)))
kw = set()
for pl in rules.SCRIPT_RULES.values():
    kw.update(p for p, _ in pl)
kw.update(["微信", "客服", "转账", "汇款", "密码", "验证码", "点击", "登陆", "下载"])
variants.set_keywords(kw)
kw_list = list(kw)

def blacklist_predict(Xev):
    return ["fraud" if any(k in t for k in kw_list) else "ad" for t in Xev]

def prf(y_true, y_pred):
    tp = sum(1 for a, b in zip(y_true, y_pred) if a == "fraud" and b == "fraud")
    fp = sum(1 for a, b in zip(y_true, y_pred) if a == "ad" and b == "fraud")
    fn = sum(1 for a, b in zip(y_true, y_pred) if a == "fraud" and b == "ad")
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    return {"P": round(P, 4), "R": round(R, 4), "F1": round(F, 4)}

out = {"M0_vs_p": {}, "M0_profiles": {}}
for p in (0.0, 0.3, 0.5, 0.7):
    rng = random.Random(SEED)
    Xev = Xte if p == 0 else [variants.perturb_keywords(t, p, rng) for t in Xte]
    out["M0_vs_p"][str(p)] = prf(yte, blacklist_predict(Xev))
    print(f"M0 p={p}: {out['M0_vs_p'][str(p)]}")

STRUCT = ["symbol_insert", "traditional", "fullwidth", "emoji_noise"]
DESTRUCT = ["homophone", "pinyin_init", "deletion", "lookalike"]
for pname, ptypes in [("structural", STRUCT), ("destructive", DESTRUCT)]:
    rng = random.Random(SEED)
    Xev = [variants.perturb_keywords(t, 0.5, rng, types=ptypes) for t in Xte]
    out["M0_profiles"][pname] = prf(yte, blacklist_predict(Xev))
    print(f"M0 {pname}@0.5: {out['M0_profiles'][pname]}")

R = json.load(open(RES, encoding="utf-8"))
R.update(out)
json.dump(R, open(RES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("merged -> results.json")
