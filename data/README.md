# Data

## FBS-SMS (the corpus used in the paper) — not redistributed

The experiments run on the **FBS-SMS** dataset of Chinese SMS messages, released
by the authors of the Tsinghua University study that first characterised
fraudulent SMS on mobile devices. It is publicly available from those authors;
please obtain it from the original source and cite their paper.

This repository deliberately does **not** redistribute the corpus: most of it
consists of real unsolicited messages, several categories are illegal-content
material, and the original terms of use are not ours to relax.

### Expected layout

```
data/FBS_SMS_Dataset-master/
├── AD_Loan/messages.txt
├── AD_Network_service/messages.txt
├── AD_Other/messages.txt
├── AD_Real_estate/messages.txt
├── AD_Retail/messages.txt
├── FR_Financial/messages.txt
├── FR_Other/messages.txt
├── FR_Phishing(Bank)/messages.txt
├── FR_Phishing(Other)/messages.txt
├── IL_Escort_service/messages.txt
├── IL_Fake_ID_and_invoice/messages.txt
├── IL_Gambling/messages.txt
└── IL_Political_propaganda/messages.txt
```

Then run

```bash
python src/prepare_data.py     # -> data/fbs_messages.csv
```

`fbs_messages.csv` has four columns: `raw`, `top`, `fine`, `source`. The
`Other` directory, if present, is skipped by design (101 ambiguous messages).

## Why the corpus is handled carefully

The rule library in `src/rules.py` includes **no active keywords** for the
`IL_Political_propaganda` category, and `results/results.json` in this
repository carries only the single innocuous bank-phishing example printed in
Figure 5. If you regenerate `results.json` locally you will produce case-study
strings for other categories; keep those out of any public fork.

## External drift benchmarks

`results/chifraud_baseline.json` (Chinese SMS, time-split) and
`results/lamda_drift.json` (Android applications, year-by-year) are aggregate
score sheets from public benchmarks. See `results/README.md`.
