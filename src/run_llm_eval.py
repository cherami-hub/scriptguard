# -*- coding: utf-8 -*-
"""
run_llm_eval.py
LLM comparison experiment for the ScriptGuard paper.
Supports OpenAI-compatible providers: DeepSeek / Aliyun DashScope (Qwen) /
OpenAI (GPT). API key via env only (never hard-code):

  DeepSeek : export DEEPSEEK_API_KEY=sk-...
  Qwen     : export DASHSCOPE_API_KEY=sk-...
  OpenAI   : export OPENAI_API_KEY=sk-...  (HTTPS_PROXY honored by requests)

Usage:
  python run_llm_eval.py --dry-run
  python run_llm_eval.py --provider deepseek --model deepseek-chat
  python run_llm_eval.py --provider qwen     --model qwen-plus
  python run_llm_eval.py --provider openai   --model gpt-4o-mini

Each run appends to results/llm_results.json (keyed by model) and merges
into results.json under T7_llm_multi[model].
"""
import os, sys, csv, json, random, time, argparse, re
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import variants, rules

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data", "fbs_messages.csv")
RES  = os.path.join(HERE, "..", "results", "results.json")
OUT  = os.path.join(HERE, "..", "results", "llm_results.json")

PROVIDERS = {
    "deepseek": ("https://api.deepseek.com/chat/completions", "DEEPSEEK_API_KEY"),
    "qwen":     ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "DASHSCOPE_API_KEY"),
    "openai":   ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
}
SEED    = 42
N_SUBSET = 300
WORKERS  = 8

# ---------------- data & attacks ----------------
def load_test_subset(n=N_SUBSET):
    from sklearn.model_selection import train_test_split
    X, ytop, yfine = [], [], []
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(row["raw"]); ytop.append(row["top"]); yfine.append(row["fine"])
    idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2,
                                      random_state=SEED, stratify=ytop)
    # attack setup must match run_experiments (train chars + rule keywords)
    variants.build_homophone_index(set("".join(X[i] for i in idx_tr)))
    kw = set()
    for pl in rules.SCRIPT_RULES.values():
        kw.update(p for p, _ in pl)
    kw.update(["微信", "客服", "转账", "汇款", "密码", "验证码", "点击", "登陆", "下载"])
    variants.set_keywords(kw)

    Xte  = [X[i] for i in idx_te]
    yb   = ["fraud" if ytop[i] in ("FR", "IL") else "ad" for i in idx_te]
    yf   = [yfine[i] for i in idx_te]
    fraud_idx = [i for i, y in enumerate(yb) if y == "fraud"]
    ad_idx    = [i for i, y in enumerate(yb) if y == "ad"]
    rng = random.Random(SEED)
    rng.shuffle(fraud_idx); rng.shuffle(ad_idx)
    sel = fraud_idx[: n // 2] + ad_idx[: n // 2]
    rng.shuffle(sel)
    subset = [{"i": i, "text": Xte[i], "y": yb[i], "fine": yf[i]} for i in sel]
    return subset

def attacked(texts, kind):
    rng = random.Random(SEED)
    types = (["symbol_insert", "traditional", "fullwidth", "emoji_noise",
              "homophone", "pinyin_init", "deletion", "lookalike"] if kind == "mixed"
             else ["homophone", "pinyin_init", "deletion", "lookalike"])
    return [variants.perturb_keywords(t, 0.5, rng, types=types) for t in texts]

# ---------------- prompts ----------------
SCRIPT_TYPES = ["FR_Phishing_Bank", "FR_Phishing_Other", "FR_Financial",
                "FR_Other", "IL_Gambling", "IL_Escort_service",
                "IL_Fake_ID_and_invoice"]

ZS_SYS = (
    "你是电信网络诈骗研判助手。给定一条中文短信，判断它属于"
    "【涉诈/违法引流】(fraud：冒充银行钓鱼、金融诈骗、中奖诈骗、赌博、"
    "色情引流、假证假票等) 还是【普通商业广告/正常短信】(ad)。"
    "注意：诈骗短信常使用谐音字、拼音、符号插入、繁体字等变体伪装关键词，"
    "请按语义判断。只输出 JSON：{\"label\":\"fraud\" 或 \"ad\"}"
)

RC_SYS_TMPL = (
    "你是电信网络诈骗研判助手。我们维护了一个涉诈剧本规则库，覆盖以下剧本类型：\n"
    "{scripts}\n"
    "给定一条中文短信，以及规则库在归一化文本上的命中证据（可能为空），"
    "请完成：1) 判断 fraud/ad；2) 若 fraud，给出最可能的剧本类型；"
    "3) 给出判定依据（引用短信中的关键片段）。"
    "规则证据可作为先验，但短信语义为最终依据；"
    "诈骗短信常使用谐音字、拼音、符号插入、繁体字等变体，请按语义还原理解。"
    "只输出 JSON：{\"label\":\"fraud\"或\"ad\",\"script\":\"类型或null\",\"evidence\":\"片段或null\"}"
)

def rule_evidence(text):
    sc = rules.script_scores(variants.normalize(text))
    fired = {k: v for k, v in sc.items() if v > 0}
    ind = rules.inducement_count(variants.normalize(text))
    return fired, ind

def make_prompts(method, text):
    if method == "zs":
        return ZS_SYS, f"短信内容：{text}"
    fired, ind = rule_evidence(text)
    ev = json.dumps(fired, ensure_ascii=False) if fired else "（无命中）"
    sys_p = RC_SYS_TMPL.replace("{scripts}", ", ".join(SCRIPT_TYPES))
    user = (f"规则命中证据：{ev}；诱导动作命中数：{ind}\n"
            f"短信内容：{text}")
    return sys_p, user

# ---------------- OpenAI-compatible API ----------------
class LLMConfig:
    def __init__(self, provider, model):
        self.api_url, self.env_name = PROVIDERS[provider]
        self.model = model
        self.key = os.environ.get(self.env_name, "").strip()

def call_llm(client_session, cfg, sys_p, user_p, max_retry=4):
    import requests
    headers = {"Authorization": f"Bearer {cfg.key}", "Content-Type": "application/json"}
    body = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": user_p},
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "response_format": {"type": "json_object"},
    }
    for attempt in range(max_retry):
        try:
            r = client_session.post(cfg.api_url, headers=headers, json=body, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt * 2)
                continue
            return f"__ERROR__ HTTP {r.status_code}: {r.text[:200]}"
        except Exception:
            time.sleep(2 ** attempt)
    return "__ERROR__ retry exhausted"

def parse_label(content):
    if not content or content.startswith("__ERROR__"):
        return None, content
    try:
        obj = json.loads(content)
        lab = str(obj.get("label", "")).strip().lower()
        if lab in ("fraud", "ad"):
            return lab, obj
    except Exception:
        pass
    m = re.search(r'"(?:label)?"\s*[:：]?\s*"?(fraud|ad)"?', content, re.I)
    if m:
        return m.group(1).lower(), {"raw": content}
    return None, {"raw": content}

# ---------------- metrics ----------------
def prf(y_true, y_pred):
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    P, R, F, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    P2, R2, F2, _ = precision_recall_fscore_support(y_true, y_pred, average="binary",
                                                    pos_label="fraud", zero_division=0)
    return {"acc": round(accuracy_score(y_true, y_pred), 4),
            "macroF1": round(F, 4), "fraudP": round(P2, 4),
            "fraudR": round(R2, 4), "fraudF1": round(F2, 4)}

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n", type=int, default=N_SUBSET)
    ap.add_argument("--provider", default="deepseek", choices=list(PROVIDERS))
    ap.add_argument("--model", default="deepseek-chat")
    args = ap.parse_args()
    cfg = LLMConfig(args.provider, args.model)

    subset = load_test_subset(args.n)
    texts = [s["text"] for s in subset]
    y = [s["y"] for s in subset]
    conds = {
        "clean": texts,
        "mixed_p0.5": attacked(texts, "mixed"),
        "destructive_p0.5": attacked(texts, "destructive"),
    }
    print(f"subset n={len(subset)} (fraud={sum(1 for v in y if v=='fraud')}, "
          f"ad={sum(1 for v in y if v=='ad')})")

    if args.dry_run:
        for cname, cx in conds.items():
            print(f"\n--- condition {cname} sample ---")
            print(cx[0][:80])
        for m in ("zs", "rc"):
            s, u = make_prompts(m, texts[0])
            print(f"\n--- prompt[{m}] SYS ---\n{s[:300]}")
            print(f"--- prompt[{m}] USER ---\n{u[:200]}")
        est = len(subset) * len(conds) * 2
        print(f"\nestimated API calls: {est}")
        return

    if not cfg.key:
        print(f"ERROR: set {cfg.env_name} first"); sys.exit(1)

    import requests
    sess = requests.Session()
    results = {"subset_n": len(subset), "model": cfg.model, "conds": {}}

    for cname, cx in conds.items():
        for method in ("zs", "rc"):
            tag = f"LLM-{method}"
            preds, raw = [None] * len(cx), [None] * len(cx)
            jobs = {}
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                for i, t in enumerate(cx):
                    s, u = make_prompts(method, t)
                    jobs[ex.submit(call_llm, sess, cfg, s, u)] = i
                done = 0
                for fut in as_completed(jobs):
                    i = jobs[fut]
                    content = fut.result()
                    lab, obj = parse_label(content)
                    preds[i] = lab
                    raw[i] = obj
                    done += 1
                    if done % 100 == 0:
                        print(f"  [{cname}/{tag}] {done}/{len(cx)}")
            # unresolved -> count as parse failures
            fails = sum(1 for p in preds if p is None)
            y_eval = [yy for yy, pp in zip(y, preds) if pp is not None]
            p_eval = [pp for pp in preds if pp is not None]
            m = prf(y_eval, p_eval) if p_eval else {}
            m["parse_failures"] = fails
            results["conds"].setdefault(cname, {})[tag] = m
            print(f"  [{cname}/{tag}] {m}")

    # reference: M1/M4 on the same subset
    import numpy as np
    from run_experiments import M_TFIDF, M_Ours
    from sklearn.model_selection import train_test_split
    X, ytop = [], []
    with open(DATA, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            X.append(row["raw"]); ytop.append(row["top"])
    ybin_all = ["fraud" if t in ("FR", "IL") else "ad" for t in ytop]
    idx_tr, _ = train_test_split(range(len(X)), test_size=0.2, random_state=SEED, stratify=ytop)
    Xtr = [X[i] for i in idx_tr]; ytr = [ybin_all[i] for i in idx_tr]
    for tag, mdl in [("M1", M_TFIDF("lr")), ("M4", M_Ours())]:
        mdl.fit(Xtr, ytr)
        for cname, cx in conds.items():
            pred = mdl.predict(cx)
            m = prf(y, list(pred))
            results["conds"].setdefault(cname, {})[tag] = m
            print(f"  [{cname}/{tag}] {m}")

    # append/merge per-model results (migrate legacy flat format)
    allr = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    if "conds" in allr and "model" in allr:        # legacy single-model file
        allr = {allr["model"]: allr}
    allr[cfg.model] = results
    json.dump(allr, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    R = json.load(open(RES, encoding="utf-8"))
    R.setdefault("T7_llm_multi", {})[cfg.model] = results["conds"]
    R["T7_llm_meta"] = {"subset_n": len(subset), "models": sorted(allr.keys())}
    json.dump(R, open(RES, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\n-> {OUT} and merged into results.json (T7_llm_multi[{cfg.model}])")

if __name__ == "__main__":
    main()
