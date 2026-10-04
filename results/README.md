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

(The two ChiFraud drift files produced by `chifraud_drift.py` are the exception:
they are aggregate-only, and they *are* committed — see Section 3.)

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

## 3. Out-of-time evaluation on ChiFraud

Two committed files back Section 5.9 of the paper. Both are purely aggregate
plus a handful of counts, so neither carries corpus text.

| file | produced by | backs |
|---|---|---|
| `chifraud_drift.json` | `chifraud_drift.py --paired` (default `--max-iter 2000`) | Table 7 |
| `chifraud_drift_converged.json` | `chifraud_drift.py --paired --max-iter 5000` | the McNemar / bootstrap numbers in Section 5.9 |

```sh
export CHIFRAUD_DIR=/path/to/ChiFraud/dataset
python src/chifraud_drift.py --paired
python src/chifraud_drift.py --paired --max-iter 5000 \
       --out results/chifraud_drift_converged.json
```

The corpus itself is **not** redistributed here — ChiFraud's own terms govern
it, so obtain it from the ChiFraud release and cite Tang et al. (COLING 2025).
The script refuses to run, with instructions, if the CSVs are missing.

Layout (identical in both files):

```
per_split        <model> -> {train, t2022, t2023} -> {acc, P, R, F1_fraud, macroF1, AUC, n, n_fraud}
drift_summary    per model: F1 per split, train-to-2023 drift, recall fall
rule_fires_rows_M3   how many messages the symbolic layer alone flagged
M4_converged / max_iter   whether the LogReg fit reached its budget
mcnemar_M4_vs_M1 / _vs_M2   only written with --paired
t2023_bootstrap95           only written with --paired
```

Four things worth knowing when quoting these numbers.

**1. The M4 fit does not converge at the default budget.** `LogisticRegression`
over character 4-grams plus rule features needs more than `max_iter=2000` on
192k messages: the fit stops on the iteration limit and scikit-learn raises a
`ConvergenceWarning`. Table 7 is deliberately the default-budget run, because
that is what a fresh `python src/run_experiments.py` style pipeline produces.
At `--max-iter 5000` the same fit converges and gives train 0.984 / t2022 0.972
/ t2023 0.630, i.e. a drift of 0.354 instead of 0.343. No row ordering changes,
which is why the paper quotes the default-budget row in Table 7 and reports the
converged figures alongside it.

**2. The paired tests in Section 5.9 are computed on the converged fit, so they
live in the second file.** That is stated in the paper, but it is easy to miss
when reproducing: running the tests at the default budget gives different
statistics for the same two comparisons.

| comparison | converged (`--max-iter 5000`) | default (`--max-iter 2000`) |
|---|---|---|
| McNemar, M4 vs M1 | b=2243, c=1125, chi2=370.45, p<1e-3 | b=2238, c=1068, chi2=413.36, p<1e-3 |
| McNemar, M4 vs M2 | b=1450, c=1474, chi2=0.18, p=0.67 | b=1646, c=1618, chi2=0.22, p=0.64 |
| bootstrap 95% CI, M4 | 0.624-0.637 | 0.627-0.640 |
| bootstrap 95% CI, M1 | 0.571-0.584 | 0.571-0.584 |

Both budgets reach the *same two verdicts*: M4 is clearly better than the
monolithic logistic model (M1) and statistically indistinguishable from the
linear SVM (M2). The bootstrap intervals of M1 and M2 are identical across the
two budgets because those fits do converge; only M4's predictions move. The
paper quotes the converged numbers because they come from a model that actually
reached its optimum.

**3. The word-level row is a reference, not M1.** M1-M4 in this repository are
all built on character `n`-grams (`src/run_experiments.py`). The `word_tfidf_lr`
row exists only because it is the configuration that reproduces the decay
originally reported for ChiFraud, and it is far worse out of time (0.378 against
0.633 in 2023) — most of that gap is tokenisation, not modelling.

**4. The script keeps only the training split in memory.** The three TF-IDF
views and the four models are all fitted on `ChiFraud_train` before any test
slice is opened; each out-of-time slice is then read, transformed by the
already-fitted vectorisers, scored and released. This caps the peak well below
2 GB, which matters because the same computation with all three splits resident
can be killed outright on a 16 GB desktop that is also running an office suite.


