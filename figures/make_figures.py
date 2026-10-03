# -*- coding: utf-8 -*-
"""
make_journal_figures.py
-----------------------
Regenerate all six figures of the ScriptGuard journal manuscript as **vector PDF**
(Elsevier accepts PDF/EPS vector artwork; a bitmapped line drawing would need
>= 1000 dpi).

Data sources (read directly - no number is hard-coded):
  ../results/results.json        -> Fig.2 profiles, Fig.3 length, Fig.5 case study
  ../results/per_class_13way.npz -> Fig.4 13-way confusion matrix
  ../results/chifraud_baseline.json, ../results/lamda_drift.json -> Fig.6

Layout safety
-------------
* Every figure is drawn at its true printed size.  The manuscript includes the
  artwork at 0.85-0.92\\linewidth and \\linewidth = 5.67 in, so a figure designed
  at ~5.2 in wide prints at its design font size instead of being rescaled.
* All boxed text is auto-shrunk until its *measured* pixel bounding box fits
  inside its parent box; the audit at the end re-measures and reports anything
  that still overflows, plus any pair of overlapping boxes.
* Fig.5 highlights are derived from the data with difflib (not hard-coded) and
  the text is wrapped by measured width, so highlight spans can never drift.
"""
import difflib
import json
import os
import sys
import unicodedata

import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
RES = os.path.join(ROOT, "results")
BASE = RES            # external-benchmark aggregates are vendored in results/
OUT = os.path.join(ROOT, "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.linewidth": 0.7,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.frameon": True,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "0.7",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,      # embed TrueType -> genuinely vector text
    "ps.fonttype": 42,
})

C_M0, C_M1, C_M2 = "#8c2d19", "#8a8a8a", "#b5b5b5"
C_M3, C_M4, C_M5 = "#c44e52", "#1f4e79", "#2d7d46"
NAVY, GREY = "#1f4e79", "#666666"
# Figure 5 embeds Chinese text, so a CJK face must be available. Rather than
# hard-coding a Windows-only font, probe for any installed CJK family that
# actually has a regular weight (some systems only register a Thin subset of
# Noto Sans SC, which would silently turn every bold label thin).
# Override with SG_CJK_FONT=<font family name>.
def _pick_cjk_font():
    from matplotlib import font_manager
    weights = {}
    for f in font_manager.fontManager.ttflist:
        weights.setdefault(f.name, set()).add(f.weight)

    env = os.environ.get("SG_CJK_FONT")
    if env:
        if env not in weights:
            print("WARNING: SG_CJK_FONT=%r is not installed." % env)
        return env

    candidates = ("Source Han Sans SC", "Source Han Sans", "Noto Sans CJK SC",
                  "Noto Sans SC", "Microsoft YaHei", "SimHei", "SimSun",
                  "PingFang SC", "Heiti SC")
    for need_bold in (True, False):          # prefer a family with bold too
        for name in candidates:
            ws = weights.get(name)
            if not ws or 400 not in ws:
                continue
            if need_bold and 700 not in ws:
                continue
            if not need_bold:
                print("WARNING: CJK font %r has no bold face; bold figure "
                      "labels will render in the regular weight." % name)
            return name
    print("WARNING: no CJK font with a regular weight found; Figure 5 will "
          "render tofu boxes. Install Source Han Sans / Noto Sans CJK and set "
          "SG_CJK_FONT.")
    return "DejaVu Sans"


CJK_FONT = _pick_cjk_font()

R = json.load(open(os.path.join(RES, "results.json"), encoding="utf-8"))

ISSUES = []          # collected audit violations
_BOXES = []          # (label, x0, y0, x1, y1) in data coords, for overlap audit
DEBUG = bool(os.environ.get("FIG_DEBUG"))


# ============================================================ small helpers
def _save(fig, name):
    p = os.path.join(OUT, name + ".pdf")
    fig.savefig(p)
    plt.close(fig)
    print("  wrote  %s  (%.2f x %.2f in)" % (name + ".pdf", *fig.get_size_inches()))


def _is_wide(ch):
    return unicodedata.east_asian_width(ch) in ("W", "F")


def _tokenize(text):
    """Atomic tokens: one CJK/wide char, or a run of ASCII characters."""
    toks, cur = [], ""
    for ch in text:
        if _is_wide(ch) or ch == " ":
            if cur:
                toks.append(cur)
                cur = ""
            toks.append(ch)
        else:
            cur += ch
    if cur:
        toks.append(cur)
    return toks


def _extent_data(ax, fig, t):
    """Bounding box of artist `t` in data coordinates: (x0, y0, x1, y1)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bb = t.get_window_extent(renderer=r)
    (x0, y0), (x1, y1) = ax.transData.inverted().transform(
        [(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def fit_text(ax, fig, t, box, tag="", pad_x=0.16, pad_y=0.10, min_fs=5.0,
             step=0.12):
    """Shrink artist `t` until its measured bbox fits inside `box`.

    `box` = (x, y, w, h) in data coordinates.  Returns the final font size.
    """
    x, y, w, h = box
    fs = t.get_fontsize()
    for _ in range(60):
        x0, y0, x1, y1 = _extent_data(ax, fig, t)
        ok_x = (x1 - x0) <= w - pad_x
        ok_y = (y1 - y0) <= h - pad_y
        inb = (x0 >= x - 0.02 and x1 <= x + w + 0.02 and
               y0 >= y - 0.02 and y1 <= y + h + 0.02)
        if ok_x and ok_y and inb:
            return fs
        if fs - step < min_fs:
            ISSUES.append(
                "OVERFLOW %s: fs=%.2f, text=(w %.2f>%.2f | h %.2f>%.2f) %r"
                % (tag, fs, x1 - x0, w - pad_x, y1 - y0, h - pad_y,
                   t.get_text().replace("\n", " / ")[:44]))
            return fs
        fs -= step
        t.set_fontsize(fs)
    return fs


def audit_boxes():
    """Report any pair of boxes that geometrically overlap."""
    for i in range(len(_BOXES)):
        for j in range(i + 1, len(_BOXES)):
            n1, a0, b0, a1, b1 = _BOXES[i]
            n2, c0, d0, c1, d1 = _BOXES[j]
            ox = min(a1, c1) - max(a0, c0)
            oy = min(b1, d1) - max(b0, d0)
            if ox > 0.01 and oy > 0.01:
                ISSUES.append("OVERLAP: %r with %r (%.2f x %.2f units)"
                              % (n1, n2, ox, oy))


def reset_boxes():
    del _BOXES[:]


# ================================================================== Fig. 1
def fig1_architecture():
    """Pipeline drawn strictly top-down: every row is a horizontal band, so no
    box can ever encroach on another row."""
    reset_boxes()
    fig, ax = plt.subplots(figsize=(5.15, 4.10))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#eef3f8", ec=NAVY, fs=8.0, bold=False,
            name=""):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.16",
            linewidth=1.0, edgecolor=ec, facecolor=fc, zorder=2))
        t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                    fontsize=fs, zorder=3, linespacing=1.38, color="#111111",
                    fontweight="bold" if bold else "normal")
        fs2 = fit_text(ax, fig, t, (x, y, w, h), tag="fig1:" + (name or text[:18]))
        _BOXES.append((name or text[:16], x - 0.05, y - 0.05,
                       x + w + 0.05, y + h + 0.05))
        return fs2

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
            linewidth=0.9, color="#555555", zorder=1, shrinkA=0, shrinkB=0))

    # ---- rows (each occupies its own horizontal band) -------------------
    box(3.20, 8.30, 5.60, 0.90, "Incoming SMS message",
        fc="#f4f4f4", ec=GREY, bold=True, name="input")

    box(0.90, 6.45, 10.20, 1.42,
        "Variant-resistant normalization\n"
        "full-width $\\rightarrow$ ASCII   $\\cdot$   symbol / emoji strip   "
        "$\\cdot$   traditional $\\rightarrow$ simplified",
        bold=True, name="norm")

    box(0.20, 4.35, 5.55, 1.62,
        "Character $n$-gram TF-IDF model\n"
        "($n \\in [2,4]$, 60K features)",
        fc="#f2f2f2", ec="#555555", name="ngram")

    box(6.25, 4.35, 5.55, 1.62,
        "Fraud-script rule library\n"
        "7 scripts  $\\cdot$  138 patterns\n"
        "+ inducement actions",
        fc="#fdf3e7", ec="#b3651a", name="rules")

    box(0.90, 2.78, 10.20, 1.10,
        "Feature fusion  $[\\,\\varphi(N(x))\\,;\\,s(x)\\,]$   "
        "$\\rightarrow$   linear classifier (logistic regression)",
        bold=True, name="fusion")

    box(0.20, 0.95, 5.55, 1.35,
        "Decision\nfraud / non-fraud + script type",
        fc="#eaf4ea", ec="#2f6b2f", name="decision")

    box(6.25, 0.95, 5.55, 1.35,
        "Reviewable evidence\nmatched rules + offending spans",
        fc="#eaf4ea", ec="#2f6b2f", name="evidence")

    # ---- purely vertical connectors, all inside the inter-row gaps ------
    arrow(6.00, 8.30, 6.00, 7.87)                     # input -> norm
    arrow(2.97, 6.45, 2.97, 5.97)                     # norm  -> ngram
    arrow(9.03, 6.45, 9.03, 5.97)                     # norm  -> rules
    arrow(2.97, 4.35, 2.97, 3.88)                     # ngram -> fusion
    arrow(9.03, 4.35, 9.03, 3.88)                     # rules -> fusion
    arrow(2.97, 2.78, 2.97, 2.30)                     # fusion -> decision
    arrow(9.03, 2.78, 9.03, 2.30)                     # fusion -> evidence

    ax.text(6.00, 0.42, "0.26 ms per message on a single CPU core",
            ha="center", va="center", fontsize=6.8, style="italic",
            color="#444444")

    audit_boxes()
    _save(fig, "fig1")


# ================================================================== Fig. 2
def fig2_profiles():
    prof = R["T5_profiles"]
    order = ["structural", "mixed", "destructive"]
    models = [("M0_keyword_filter", "M0 Keyword filter", C_M0),
              ("M1", "M1 TF-IDF+LR", C_M1),
              ("M3", "M3 Rule-only", C_M3),
              ("M4", "M4 ScriptGuard", C_M4)]

    fig, ax = plt.subplots(figsize=(4.35, 3.05))
    x = np.arange(len(order))
    w = 0.20
    for i, (key, lab, col) in enumerate(models):
        vals = [prof[p][key]["posF1"] for p in order]
        off = (i - 1.5) * w
        ax.bar(x + off, vals, w, label=lab, color=col,
               edgecolor="white", linewidth=0.5, zorder=3)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.018, "%.3f" % v, ha="center", va="bottom",
                    fontsize=5.4, rotation=90, color="#333333", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(["structural", "mixed", "destructive"])
    ax.set_xlabel("Attack profile ($p = 0.5$, seed-42 split)")
    ax.set_ylabel("fraud-class $F_1$")
    ax.set_ylim(0, 1.16)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    # legend sits fully outside the axes -> it can never cover the tallest bar
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              handlelength=1.1, columnspacing=1.2, borderpad=0.35)
    _save(fig, "fig2")


# ================================================================== Fig. 3
def fig3_length():
    L = R["T6_length"]
    keys = ["short(<=40)", "mid(41-80)", "long(>80)"]
    names = ["short\n($\\leq$40)", "mid\n(41\u201380)", "long\n($>$80)"]
    m1 = [L[k]["M1"]["posF1"] for k in keys]
    m4 = [L[k]["M4"]["posF1"] for k in keys]
    ns = [L[k]["n"] for k in keys]

    fig, ax = plt.subplots(figsize=(4.15, 2.85))
    x = np.arange(len(keys))
    w = 0.34
    b1 = ax.bar(x - w / 2, m1, w, label="M1 TF-IDF+LR", color=C_M1,
                edgecolor="white", linewidth=0.5, zorder=3)
    b2 = ax.bar(x + w / 2, m4, w, label="M4 ScriptGuard", color=C_M4,
                edgecolor="white", linewidth=0.5, zorder=3)
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.0020,
                    "%.4f" % r.get_height(), ha="center", va="bottom",
                    fontsize=5.9, color="#333333", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n$n = %s$" % (n, v) for n, v in zip(names, ns)])
    ax.set_xlabel("Message length (characters)")
    ax.set_ylabel("fraud-class $F_1$")
    ax.set_ylim(0.892, 1.012)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2,
              handlelength=1.1, columnspacing=1.4, borderpad=0.35)
    _save(fig, "fig3")


# ================================================================== Fig. 4
def fig4_confusion():
    z = np.load(os.path.join(RES, "per_class_13way.npz"), allow_pickle=True)
    labels = [str(s) for s in z["labels"]]
    cm = z["cm4"].astype(float)

    short = {"AD_Loan": "AD-Loan", "AD_Network_service": "AD-NetSvc",
             "AD_Other": "AD-Other", "AD_Real_estate": "AD-RealEst",
             "AD_Retail": "AD-Retail", "FR_Financial": "FR-Fin",
             "FR_Other": "FR-Other", "FR_Phishing_Bank": "FR-PhBank",
             "FR_Phishing_Other": "FR-PhOther", "IL_Escort_service": "IL-Escort",
             "IL_Fake_ID_and_invoice": "IL-FakeID", "IL_Gambling": "IL-Gambling",
             "IL_Political_propaganda": "IL-Politics"}
    ticks = [short.get(l, l.replace("_", "-")) for l in labels]

    row = cm.sum(axis=1, keepdims=True)
    row[row == 0] = 1
    norm = cm / row

    fig, ax = plt.subplots(figsize=(4.75, 4.20))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(ticks)))
    ax.set_yticks(range(len(ticks)))
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=6.4)
    ax.set_yticklabels(ticks, fontsize=6.4)
    ax.set_xlabel("Predicted category", fontsize=8)
    ax.set_ylabel("True category", fontsize=8)
    ax.tick_params(length=2, pad=1.5)
    for s in ax.spines.values():
        s.set_linewidth(0.6)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            if v == 0:
                continue
            ax.text(j, i, str(v), ha="center", va="center", fontsize=4.7,
                    color="white" if norm[i, j] > 0.55 else "#222222")

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("row-normalised frequency", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    _save(fig, "fig4")


# ================================================================== Fig. 5
def fig5_case_study():
    s = R["adv_samples"][0]
    orig, adv, norm = s["orig"], s["adv"], s["norm"]
    label = s["label"]

    def diff_marks(ref, txt):
        """Indices in `txt` that are inserted or substituted w.r.t. `ref`."""
        out = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                None, ref, txt, autojunk=False).get_opcodes():
            if tag in ("replace", "insert"):
                out.extend(range(j1, j2))
        return out

    adv_marks = diff_marks(orig, adv)      # ['*', ?], '▲', '#', ' '
    norm_marks = diff_marks(orig, norm)    # the look-alike that survived

    rows = [
        ("Original", orig, []),
        ("Attacked", adv, adv_marks),
        ("Normalized", norm, norm_marks),
        ("M0 filter", "pass (missed)", []),
        ("M4 ScriptGuard", "fraud  /  " + label, []),
        ("Evidence", "product(\u4fe1\u7528\u5361)  \u00b7  "
                     "bank-suffix(\u94f6\u884c)  \u00b7  "
                     "inducement(\u8fdb\u5165URL\u586b\u5199\u7533\u8bf7)", []),
    ]

    fs = 7.4
    fig, ax = plt.subplots(figsize=(5.2, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ax_px = ax.get_window_extent().width

    def wfrac(txt, size=fs, weight="normal"):
        t = ax.text(0, 0.5, txt, fontsize=size, family=CJK_FONT, alpha=0,
                    fontweight=weight)
        w = t.get_window_extent(renderer=rend).width / ax_px
        t.remove()
        return w

    # the labels are printed in bold, so they must be measured in bold
    lab_w = max(wfrac(n + ":", weight="bold") for n, _, _ in rows) + 0.026
    x_lab = 0.010
    x_txt = x_lab + lab_w
    max_w = 0.986 - x_txt

    def wrap(text):
        """Wrap by measured width and return [(line, start_offset), ...]."""
        lines, cur = [], ""
        for tok in _tokenize(text):
            if cur and wfrac(cur + tok) > max_w:
                lines.append(cur)
                cur = tok
            else:
                cur += tok
        if cur:
            lines.append(cur)
        out, off = [], 0
        for ln in lines:
            out.append((ln, off))
            off += len(ln)
        return out

    wrapped = [wrap(txt) for _, txt, _ in rows]
    total_lines = sum(len(w) for w in wrapped)

    pad_top, pad_bot = 0.955, 0.045
    step = min(0.150, (pad_top - pad_bot) / max(total_lines - 1, 1))
    y = pad_top

    for (name, txt, spans), lines in zip(rows, wrapped):
        ax.text(x_lab, y, name + ":", ha="left", va="center", fontsize=fs,
                fontweight="bold", family=CJK_FONT)
        for li, (ln, off) in enumerate(lines):
            yy = y - li * step
            for idx in spans:                       # highlight first, text on top
                sa = idx - off
                if sa < 0 or sa >= len(ln):
                    continue
                x0 = x_txt + wfrac(ln[:sa])
                x1 = x_txt + wfrac(ln[:sa + 1])
                ax.add_patch(Rectangle((x0 - 0.0020, yy - step * 0.36),
                                       (x1 - x0) + 0.004, step * 0.72,
                                       facecolor="#ffdf7a", edgecolor="none",
                                       zorder=2))
            ax.text(x_txt, yy, ln, ha="left", va="center", fontsize=fs,
                    family=CJK_FONT, zorder=3)
        y -= step * len(lines)

    ax.add_patch(Rectangle((0.005, 0.015), 0.990, 0.970, transform=ax.transAxes,
                           fill=False, edgecolor="#9aa5b1", linewidth=0.8,
                           zorder=4, clip_on=False))

    print("  fig5 highlight offsets: attacked=%s normalized=%s"
          % (adv_marks, norm_marks))
    _save(fig, "fig5")


def _poly_display(transform, xs, ys, per=60):
    """Densely sample a polyline and map it to display coordinates."""
    pts = []
    for i in range(len(xs) - 1):
        for tt in np.linspace(0.0, 1.0, per, endpoint=False):
            pts.append((xs[i] + tt * (xs[i + 1] - xs[i]),
                        ys[i] + tt * (ys[i + 1] - ys[i])))
    pts.append((xs[-1], ys[-1]))
    return transform.transform(np.asarray(pts, dtype=float))


def _hits(pts, bb, pad=1.5):
    return bool(np.any((pts[:, 0] > bb.x0 - pad) & (pts[:, 0] < bb.x1 + pad) &
                       (pts[:, 1] > bb.y0 - pad) & (pts[:, 1] < bb.y1 + pad)))


def _overlaps(a, b, pad=6.0):
    """Labels need a visible gap, not merely non-intersection
    (pad is in display pixels; ~5 pt at the default 100 dpi canvas)."""
    return (a.x0 < b.x1 + pad and b.x0 < a.x1 + pad and
            a.y0 < b.y1 + pad and b.y0 < a.y1 + pad)


# candidate (dx, dy, ha) offsets, tried in order: above, below, then sideways
CANDIDATES = [(0, 9, "center"), (0, -12, "center"), (0, -17, "center"),
              (6, 8, "left"), (-6, 8, "right"),
              (11, 7, "left"), (-11, 7, "right"),
              (11, -10, "left"), (-11, -10, "right"),
              (0, 15, "center"), (0, 21, "center"),
              (-13, 4, "right"), (13, 4, "left"),
              (-13, -7, "right"), (13, -7, "left"),
              (8, 15, "left"), (-8, 15, "right"),
              (8, -18, "left"), (-8, -18, "right")]


def place_labels(ax, fig, items, obstacles, tag=""):
    """Place value labels near their data points without touching any curve.

    `items`    : [(x, y, text, colour, fontsize, xycoords), ...]  where xycoords
                 is the transform the point belongs to (needed on twin axes).
    `obstacles`: list of (N, 2) display-coordinate polylines that must stay clear
    Returns the number of labels that had to fall back to their first choice.
    """
    accepted, fallback = [], 0
    ax_bb = ax.get_window_extent()
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def new(x, y, text, colour, fs, cand, xycoords):
        dx, dy, ha = cand
        return ax.annotate(text, (x, y), xycoords=xycoords, xytext=(dx, dy),
                           textcoords="offset points", ha=ha, va="center",
                           color=colour, fontsize=fs, zorder=6,
                           bbox=dict(facecolor="white", edgecolor="none",
                                     pad=0.35, alpha=0.95))

    for x, y, text, colour, fs, xycoords in items:
        best = None
        for cand in CANDIDATES:
            t = new(x, y, text, colour, fs, cand, xycoords)
            fig.canvas.draw()
            bb = t.get_window_extent(renderer=rend)
            inside = (bb.x0 >= ax_bb.x0 - 0.5 and bb.x1 <= ax_bb.x1 + 0.5 and
                      bb.y0 >= ax_bb.y0 - 0.5 and bb.y1 <= ax_bb.y1 + 0.5)
            if (inside and not any(_hits(p, bb) for p in obstacles)
                    and not any(_overlaps(bb, o) for o in accepted)):
                best = bb
                break
            if DEBUG:
                print("    [%s] %r cand%s %s -> inside=%s hit=%s ovl=%s"
                      % (tag, text, cand, "REJECT",
                         inside,
                         [i for i, p in enumerate(obstacles)
                          if _hits(p, bb)],
                         [i for i, o in enumerate(accepted)
                          if _overlaps(bb, o)]))
            t.remove()
        if best is None:                       # nothing clean -> best effort
            fallback += 1
            t = new(x, y, text, colour, fs, CANDIDATES[0], xycoords)
            fig.canvas.draw()
            best = t.get_window_extent(renderer=rend)
            ISSUES.append("LABEL %s: no clean slot for %r at (%.3f, %.3f)"
                          % (tag, text, x, y))
        accepted.append(best)
    return fallback


# ================================================================== Fig. 6
def fig6_drift():
    cf = json.load(open(os.path.join(BASE, "chifraud_baseline.json"),
                        encoding="utf-8"))
    lm = json.load(open(os.path.join(BASE, "lamda_drift.json"),
                        encoding="utf-8"))

    cf_years = ["2020\u201322\n(in-time)", "2022", "2023"]
    cf_f1 = [cf["train(训练集自评)"]["f1"], cf["t2022 时间外"]["f1"],
             cf["t2023 时间外"]["f1"]]
    cf_rc = [cf["train(训练集自评)"]["recall"], cf["t2022 时间外"]["recall"],
             cf["t2023 时间外"]["recall"]]

    lm_years = sorted(lm.keys())
    lm_f1 = [lm[y]["f1"] for y in lm_years]
    lm_share = [lm[y]["malicious"] / (lm[y]["malicious"] + lm[y]["benign"])
                for y in lm_years]

    fig, axes = plt.subplots(1, 2, figsize=(5.35, 2.50))

    # ---- (a) ChiFraud text drift ---------------------------------------
    ax = axes[0]
    x = np.arange(len(cf_f1))
    ax.plot(x, cf_f1, "o-", color=C_M4, linewidth=1.3, markersize=3.6,
            label="macro-$F_1$", zorder=3)
    ax.plot(x, cf_rc, "s--", color=C_M3, linewidth=1.2, markersize=3.4,
            label="fraud recall", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(cf_years, fontsize=6.8)
    ax.set_xlim(-0.28, 2.34)          # leaves room for the last value label
    ax.set_ylim(0.05, 1.19)
    ax.set_yticks(np.arange(0.2, 1.01, 0.2))
    ax.set_ylabel("score", fontsize=8)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", handlelength=1.2, borderpad=0.3,
              labelspacing=0.35)
    ax.set_title("(a) ChiFraud (text), trained on 2020-era", fontsize=7.2, pad=5)

    obs = [_poly_display(ax.transData, x, cf_f1),
           _poly_display(ax.transData, x, cf_rc)]
    items = [(xi, v, "%.3f" % v, C_M4, 6.0, ax.transData)
             for xi, v in zip(x, cf_f1)]
    items += [(xi, v, "%.3f" % v, C_M3, 6.0, ax.transData)
              for xi, v in zip(x, cf_rc)]
    place_labels(ax, fig, items, obs, tag="fig6a")

    # ---- (b) LAMDA app drift -------------------------------------------
    ax = axes[1]
    x = np.arange(len(lm_years))
    ax.plot(x, lm_f1, "D-", color=C_M4, linewidth=1.3, markersize=3.2,
            label="macro-$F_1$", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(lm_years, rotation=45, fontsize=6.4)
    ax.set_xlim(-0.60, 8.30)          # leaves room for the last value label
    ax.set_ylim(0.0, 1.20)
    ax.set_yticks(np.arange(0.0, 1.01, 0.2))
    ax.set_ylabel("macro-$F_1$", fontsize=8)
    ax.grid(linestyle=":", linewidth=0.5, alpha=0.75, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(b) LAMDA (Android apps), anchored on 2018", fontsize=7.2,
                 pad=5)

    # The right-hand scale is set to 0-0.80 (= 1.5x the peak malicious share)
    # so that the share curve is guaranteed to stay *below* the macro-F1 curve
    # at every year: 0.4819 * 1.5 = 0.72 < min(F1 before collapse) = 0.728.
    ax2 = ax.twinx()
    ax2.plot(x, lm_share, "^:", color="#a0522d", linewidth=1.0,
             markersize=3.0, label="malicious share", zorder=2)
    ax2.set_ylim(0.0, 0.80)
    ax2.set_yticks(np.arange(0.0, 0.81, 0.2))
    ax2.set_ylabel("malicious share", fontsize=7.2, color="#a0522d")
    ax2.tick_params(axis="y", labelcolor="#a0522d", labelsize=6.4)
    ax2.spines["right"].set_color("#a0522d")

    obsb = [_poly_display(ax.transData, x, lm_f1),
            _poly_display(ax2.transData, x, lm_share)]
    itemsb = [(xi, v, "%.2f" % v, C_M4, 5.9, ax.transData)
              for xi, v in zip(x, lm_f1)]
    # label only the two ends of the imbalance story: the peak and the collapse
    ipk = int(np.argmax(lm_share))
    itemsb += [(0, lm_share[0], "%.2f" % lm_share[0], "#a0522d", 5.7,
                ax2.transData),
               (ipk, lm_share[ipk], "%.2f" % lm_share[ipk], "#a0522d", 5.7,
                ax2.transData)]
    place_labels(ax, fig, itemsb, obsb, tag="fig6b")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # lower left: the only region clear of both curves
    ax.legend(h1 + h2, l1 + l2, loc="lower left", handlelength=1.2,
              borderpad=0.3, fontsize=6.2, labelspacing=0.35)

    fig.subplots_adjust(wspace=0.54)
    _save(fig, "fig6")


# ======================================================================= main
if __name__ == "__main__":
    print("Regenerating vector figures ->", OUT)
    fig1_architecture()
    fig2_profiles()
    fig3_length()
    fig4_confusion()
    fig5_case_study()
    fig6_drift()

    if ISSUES:
        print("\n" + "=" * 68)
        for w in ISSUES:
            print("[ISSUE] " + w)
        print("=" * 68)
        print("%d layout issue(s) found" % len(ISSUES))
        sys.exit(1)
    print("\nlayout audit: no overflow, no overlap detected")
