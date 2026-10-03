# Results

Two kinds of file live here.

## 1. Regenerable experiment output

These are produced by the scripts in `src/` and are **not** committed (they are
large, and the case-study strings inside them are corpus text):

| file | produced by |
|---|---|
| `v5_seeds.json` | `run_v5_experiments.py` |
| `results.json` | `run_experiments.py`, then `run_v6_experiments.py` |
| `per_class_13way.npz` | `run_v5_experiments.py` (`cm1`, `cm4`, `labels`) |
| `independent_vocab_results.json` | `run_independent_vocab_attack.py` |
| `llm_results.json` | `run_llm_eval.py` |
| `bert_results.json` | `run_bert_baseline.py` |

`results.json` is committed in a **trimmed** form: every aggregate metric used
by the paper is present, but the `adv_samples` array is reduced to the single
bank-phishing message shown in Figure 5. The other sampled messages are
illegal-content material and are intentionally excluded.

## 2. Vendored external benchmarks

These are small, purely aggregate files (a few hundred bytes each). They are
committed so that Figure 6 can be regenerated without re-running two unrelated
off-device benchmarks.

### `chifraud_baseline.json`

A TF-IDF + logistic-regression model over an independent Chinese SMS corpus
(ChiFraud), trained on 2020-era messages and evaluated out of time on the 2022
and 2023 slices. Keys:

```
"train(训练集自评)"   in-time self-evaluation on the training era
"t2022 时间外"        out-of-time evaluation on the 2022 slice
"t2023 时间外"        out-of-time evaluation on the 2023 slice
```

Each holds `accuracy`, `precision`, `recall`, `f1`, `auc`. Feeds Figure 6(a).

### `lamda_drift.json`

A model anchored on the 2018 slice of the LAMDA Android-app benchmark and
applied year by year through 2025. Keys are the years `"2018"` … `"2025"`; each
holds the same five metrics plus the raw class counts `malicious` and `benign`.
The class counts are what make the collapse interpretable: the 2025 slice
contains 5 malicious applications against 8 928 benign ones, so a macro-F1 of
0 is a statement about the sample, not only about the model. Feeds Figure 6(b).

Both files record **derived** numbers, not raw data; neither is a substitute for
the original benchmark releases, which should be cited from their own papers.
