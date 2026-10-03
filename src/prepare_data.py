# -*- coding: utf-8 -*-
"""
prepare_data.py
Load FBS_SMS_Dataset (CCS'20, Tsinghua) into a unified CSV.

Category mapping:
  AD_*  -> top = AD   (ordinary advertisement spam, non-fraud)
  FR_*  -> top = FR   (fraud)
  IL_*  -> top = IL   (illegal-content diversion)
  Other -> dropped (101 msgs, ambiguous misc)

Binary task : fraud diversion (FR+IL) vs non-fraud spam (AD)
Fine task   : 13 fine-grained script categories
"""
import os, csv, sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "FBS_SMS_Dataset-master")
OUT_CSV  = os.path.join(os.path.dirname(__file__), "..", "data", "fbs_messages.csv")

FILES = [
    ("AD_Loan", "AD"), ("AD_Network_service", "AD"), ("AD_Other", "AD"),
    ("AD_Real_estate", "AD"), ("AD_Retail", "AD"),
    ("FR_Financial", "FR"), ("FR_Other", "FR"),
    ("FR_Phishing(Bank)", "FR"), ("FR_Phishing(Other)", "FR"),
    ("IL_Escort_service", "IL"), ("IL_Fake_ID_and_invoice", "IL"),
    ("IL_Gambling", "IL"), ("IL_Political_propaganda", "IL"),
]

def main():
    rows = []
    for fname, top in FILES:
        path = os.path.join(DATA_DIR, fname)
        fine = fname.replace("FR_Phishing(Bank)", "FR_Phishing_Bank") \
                    .replace("FR_Phishing(Other)", "FR_Phishing_Other")
        with open(path, encoding="utf-8") as f:
            for line in f:
                seg = line.strip()
                if not seg:
                    continue
                raw = seg.replace(" ", "")          # de-segment -> raw text
                if len(raw) < 6:
                    continue
                rows.append((raw, seg, top, fine))
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["raw", "seg", "top", "fine"])
        w.writerows(rows)
    from collections import Counter
    c_top, c_fine = Counter(r[2] for r in rows), Counter(r[3] for r in rows)
    print(f"total={len(rows)}")
    print("top:", dict(c_top))
    print("fine:", dict(sorted(c_fine.items())))
    print("->", OUT_CSV)

if __name__ == "__main__":
    main()
