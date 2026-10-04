# ScriptGuard — auditable fraud-script detection for Chinese SMS

Reference implementation and experiment code for the manuscript

> **Keyword-targeted evasion of Chinese SMS filters: an adversarial benchmark
> and the cost of auditable detection**
> Runsheng Luan, Ping Jiang, Ming Yang, Xiaogang Wang
> *(submitted to Journal of Information Security and Applications)*

ScriptGuard normalises adversarial text variants, scores a message with a
character *n*-gram TF-IDF model and a hand-built fraud-script rule library in
parallel, fuses the two feature blocks, and emits **matched rules and offending
spans** alongside every decision. The claim defended in the paper is
*auditability at a quantified cost* — we measure what the auditable operating
point gives up in macro-F1 and in latency — rather than a new accuracy ceiling.

---

## 1. Repository layout

```
.
├── src/                              experiment and benchmark code
│   ├── prepare_data.py               FBS-SMS -> unified CSV (labels, splits)
│   ├── variants.py                   the 8-type adversarial variant generator
│   ├── rules.py                      the fraud-script rule library
│   ├── run_experiments.py            first-generation harness (M0-M4, p sweep)
│   ├── run_v5_experiments.py         main harness: 5 seeds, profiles, length
│   ├── run_v6_experiments.py         extended ablations on top of v5 output
│   ├── run_independent_vocab_attack.py  attacker with an unseen vocabulary
│   ├── run_llm_eval.py               three-vendor LLM comparison (0-shot/1-shot)
│   ├── run_bert_baseline.py          fine-tuned RoBERTa accuracy ceiling
│   ├── extra_metrics.py              per-class and threshold-free metrics
│   └── analyze_{v5,v6}.py            turn result JSONs into paper tables
├── figures/
│   └── make_figures.py               regenerates Figures 1-6 as vector PDF
├── results/                          vendored aggregates + how to regenerate
├── data/                             how to obtain FBS-SMS (not redistributed)
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

All scripts resolve their inputs relative to the repository root
(`../data`, `../results`), so no path editing is needed after cloning.

---

## 2. Getting the data

The corpus is **not redistributed here** — it is the FBS-SMS dataset released
with

> Zhang et al. *Characterizing and Detecting Fraudulent Activities on Mobile
> Devices: A Case Study of SMS*. (Tsinghua University)

Download it, then lay it out as

```
data/FBS_SMS_Dataset-master/<category>/messages.txt
```

and convert it:

```bash
python src/prepare_data.py            # -> data/fbs_messages.csv
```

`prepare_data.py` maps the original 13 categories onto the paper's two label
spaces:

| label space | definition |
|---|---|
| binary | `fraud` = `FR` (fraud) ∪ `IL` (illegal-content diversion); `ad` = the five `AD_*` advertising classes |
| fine | the 13 original categories |

The 101 `Other` messages (ambiguous miscellany) are dropped.

---

## 3. Reproducing the experiments

```bash
pip install -r requirements.txt
python src/prepare_data.py

# main results: 5 seeds x {clean, p=0.3, p=0.5, p=0.7} x {M1..M4, M4b}
python src/run_v5_experiments.py --task bin --seeds 42,43,44,45,46

# 13-way fine-grained task
python src/run_v5_experiments.py --task fine --seeds 42,43,44,45,46

# extended ablations (attack profiles, length strata, type-wise robustness)
python src/run_v6_experiments.py

# attacker holding an independent lexicon
python src/run_independent_vocab_attack.py

# accuracy ceiling and LLM comparison
python src/run_bert_baseline.py            # needs torch + transformers
python src/run_llm_eval.py                 # needs API keys, see the file header
```

`run_v5_experiments.py` supports `--task`, `--seeds`, `--conds` and `--models`
so a single seed/condition block can be rerun in isolation. Results are flushed
to `results/v5_seeds.json` after **every** block by atomic replace, so an
interrupted run never loses completed work.

---

## 4. Result files and how they map to the paper

| file | produced by | used for |
|---|---|---|
| `results/v5_seeds.json` | `run_v5_experiments.py` | Tables 2-4, the paired *t*-tests, the 5-seed variance study |
| `results/results.json` | `run_experiments.py` / `run_v6_experiments.py` | attack profiles (Fig. 2), length strata (Fig. 3), type-wise robustness, the case study (Fig. 5) |
| `results/per_class_13way.npz` | `run_v5_experiments.py` | the 13-way confusion matrix (Fig. 4) |
| `results/independent_vocab_results.json` | `run_independent_vocab_attack.py` | the independent-vocabulary attack |
| `results/llm_results.json` | `run_llm_eval.py` | the LLM comparison |
| `results/bert_results.json` | `run_bert_baseline.py` | the RoBERTa ceiling |
| `results/chifraud_baseline.json` | external benchmark (ChiFraud, Chinese SMS) | Fig. 6(a) |
| `results/lamda_drift.json` | external benchmark (LAMDA, Android apps) | Fig. 6(b) |

`results/chifraud_baseline.json` and `results/lamda_drift.json` are small,
purely aggregate files (scores and class counts per year) from the two external
drift benchmarks; they are vendored so that Figure 6 can be regenerated without
re-running those benchmarks. Their provenance is described in `results/README.md`.

Vendored `results/results.json` carries the aggregate metrics that back
Figures 2, 3 and 5, plus a **single** illustrative message (the one shown in
Figure 5). The remaining corpus text is deliberately not included.

---

## 5. Regenerating the figures

```bash
python figures/make_figures.py        # -> figures/fig1.pdf ... fig6.pdf
```

Figures are written as **vector PDF** at their true printed size. The script
audits itself: boxed text is auto-shrunk until its *measured* bounding box fits,
value labels are placed by testing candidate positions against every plotted
curve, and the run exits non-zero if any overflow or box overlap survives.
Set `FIG_DEBUG=1` to see why individual label candidates were rejected.

Figure 5 embeds Chinese text, so a CJK font must be installed. The script
probes for `Microsoft YaHei`, `Source Han Sans SC`, `Noto Sans CJK SC`,
`SimHei`, `SimSun` and `PingFang SC`; override with `SG_CJK_FONT=<name>`.
For redistribution-safe output, install an SIL-licensed face such as
**Source Han Sans** and set `SG_CJK_FONT=Source Han Sans SC`.

---

## 6. Two design decisions worth knowing before you read the code

**The rule library is deliberately incomplete on one class.**
`rules.py` defines no active keywords for `IL_Political_propaganda`; that class
is left entirely to the statistical model. This is intentional (see Section 4.2
of the paper) and is why the rule-only variant M3 under-performs on that class.

**The attack generator is corpus-derived, not hand-written.**
`variants.py` builds its homophone and look-alike maps from the training split
itself (`lru_cache`d pinyin conversion), so an adversary that re-runs this code
on the *test* split is a different, stronger adversary than the one evaluated in
the paper. `run_independent_vocab_attack.py` implements that stronger setting
explicitly.

---

## 7. Licence and citation

Code is released under the MIT licence (`LICENSE`). The FBS-SMS corpus keeps its
own terms — see `data/README.md`.

If you use this code, please cite the manuscript (`CITATION.cff`).
