# -*- coding: utf-8 -*-
"""
run_bert_baseline.py
Chinese-RoBERTa-wwm-ext fine-tuned baseline for the ScriptGuard paper.

Same split / attack protocol as run_experiments.py:
  - stratified 80/20, seed 42 (same indices via sklearn)
  - binary fraud-vs-ad; clean + keyword-targeted attack p in {0.3,0.5,0.7}
  - N seeds (default 3): report mean +/- std of macro-F1 / fraud-F1

Model: hfl/chinese-roberta-wwm-ext (HF; set HF_ENDPOINT=https://hf-mirror.com
if huggingface.co is unreachable).

GPU is preferred; falls back to CPU (then use --seeds 2 --epochs 2).
Output: results/bert_results.json + merged into results.json as T8_bert.
"""
import os, sys, csv, json, random, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants, rules

import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "fbs_messages.csv")
RES  = os.path.join(HERE, "..", "results", "results.json")
OUT  = os.path.join(HERE, "..", "results", "bert_results.json")
SEED = 42
INTENSITIES = [0.3, 0.5, 0.7]

def load_split():
    from sklearn.model_selection import train_test_split
    X, ytop = [], []
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(row["raw"]); ytop.append(row["top"])
    y = [1 if t in ("FR", "IL") else 0 for t in ytop]
    idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2,
                                      random_state=SEED, stratify=ytop)
    # attack setup identical to run_experiments.py
    variants.build_homophone_index(set("".join(X[i] for i in idx_tr)))
    kw = set()
    for pl in rules.SCRIPT_RULES.values():
        kw.update(p for p, _ in pl)
    kw.update(["微信", "客服", "转账", "汇款", "密码", "验证码", "点击", "登陆", "下载"])
    variants.set_keywords(kw)
    return ([X[i] for i in idx_tr], [y[i] for i in idx_tr],
            [X[i] for i in idx_te], [y[i] for i in idx_te])

def attacked(Xte, p):
    rng = random.Random(SEED)
    return [variants.perturb_keywords(t, p, rng) for t in Xte]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hfl/chinese-roberta-wwm-ext")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=96)
    args = ap.parse_args()

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model}")
    if device == "cpu" and args.seeds > 2:
        print("WARNING: CPU detected; consider --seeds 2 --epochs 2")

    Xtr, ytr, Xte, yte = load_split()
    print(f"train={len(Xtr)} test={len(Xte)}")

    tok = AutoTokenizer.from_pretrained(args.model)

    class DS(Dataset):
        def __init__(self, X, y):
            self.X, self.y = X, y
        def __len__(self):
            return len(self.X)
        def __getitem__(self, i):
            enc = tok(self.X[i], truncation=True, max_length=args.maxlen,
                      padding="max_length", return_tensors="pt")
            return {k: v.squeeze(0) for k, v in enc.items()}, self.y[i]

    def collate(batch):
        items, labels = zip(*batch)
        out = {k: torch.stack([it[k] for it in items]) for k in items[0]}
        return out, torch.tensor(labels)

    train_loader = DataLoader(DS(Xtr, ytr), batch_size=args.batch,
                              shuffle=True, collate_fn=collate)

    def train_one(seed):
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model, num_labels=2).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * total_steps), total_steps)
        model.train()
        for ep in range(args.epochs):
            t0 = time.time(); tot = 0
            for batch, labels in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = labels.to(device)
                out = model(**batch, labels=labels)
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
                tot += 1
            print(f"  seed{seed} epoch{ep+1} done ({time.time()-t0:.0f}s, {tot} steps)")
        return model

    @torch.no_grad()
    def evaluate(model, Xev):
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        model.eval()
        preds = []
        loader = DataLoader(DS(Xev, [0]*len(Xev)), batch_size=64,
                            shuffle=False, collate_fn=collate)
        for batch, _ in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            preds.extend(logits.argmax(-1).cpu().tolist())
        P, R, F, _ = precision_recall_fscore_support(yte, preds, average="macro", zero_division=0)
        P2, R2, F2, _ = precision_recall_fscore_support(yte, preds, average="binary", pos_label=1, zero_division=0)
        return {"acc": round(accuracy_score(yte, preds), 4),
                "macroF1": round(F, 4), "fraudP": round(P2, 4),
                "fraudR": round(R2, 4), "fraudF1": round(F2, 4)}

    conds = {"clean": Xte}
    for p in INTENSITIES:
        conds[f"p{p}"] = attacked(Xte, p)

    all_runs = []
    for si in range(args.seeds):
        seed = SEED + si
        print(f"=== seed {seed} ===")
        model = train_one(seed)
        run = {}
        for cname, Xev in conds.items():
            run[cname] = evaluate(model, Xev)
            print(f"  {cname}: {run[cname]}")
        all_runs.append(run)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # aggregate mean +/- std
    agg = {}
    for cname in conds:
        vals = [r[cname]["macroF1"] for r in all_runs]
        fv = [r[cname]["fraudF1"] for r in all_runs]
        agg[cname] = {
            "macroF1_mean": round(float(np.mean(vals)), 4),
            "macroF1_std": round(float(np.std(vals)), 4),
            "fraudF1_mean": round(float(np.mean(fv)), 4),
            "fraudF1_std": round(float(np.std(fv)), 4),
        }
    result = {"model": args.model, "seeds": [SEED + i for i in range(args.seeds)],
              "epochs": args.epochs, "batch": args.batch, "maxlen": args.maxlen,
              "device": device, "runs": all_runs, "agg": agg}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    R = json.load(open(RES, encoding="utf-8"))
    R["T8_bert"] = result
    json.dump(R, open(RES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\n== aggregated ==")
    for cname, m in agg.items():
        print(f"  {cname}: macroF1 {m['macroF1_mean']}±{m['macroF1_std']}  fraudF1 {m['fraudF1_mean']}±{m['fraudF1_std']}")
    print(f"-> {OUT} and merged into results.json (T8_bert)")

if __name__ == "__main__":
    main()
