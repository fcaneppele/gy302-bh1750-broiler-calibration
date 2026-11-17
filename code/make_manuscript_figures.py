#!/usr/bin/env python3
"""
Figures 1-3 of the revised manuscript (direct-error framework).

Fig. 1: (a) raw GY-302 vs reference with Eq. (9) fit; (b) corrected GY-302 vs
        reference with identity line (no second regression).
Fig. 2: Bland-Altman plot of Stage-2 direct errors.
Fig. 3: Direct relative error vs reference illuminance, with the reference
        instrument tolerance (+/-6 %, LED) and the magnitude of one 1-lx
        quantisation step.
Usage: python make_manuscript_figures.py [output_dir]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results" / "manuscript_figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, RED, GREY = "#1f4e79", "#c0504d", "#7f7f7f"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False})

s1 = pd.read_csv(DATA / "calibration_stage1_table5.csv")
s2 = pd.read_csv(DATA / "validation_stage2_table6.csv")
raw, r1 = s1.gy302_raw_lx.values.astype(float), s1.ref_itr3000_lx.values
cor, r2 = s2.gy302_corrected_lx.values.astype(float), s2.ref_itr3000_lx.values
fit = stats.linregress(raw, r1)

# ---- Figure 1 (aspect 2.417 to match the manuscript frame) ----
fig, ax = plt.subplots(1, 2, figsize=(7.25, 3.0))
xx = np.linspace(0, 56, 100)
ax[0].scatter(raw, r1, s=16, color=RED, zorder=3, label="Observed pairs (n = 26)")
ax[0].plot(xx, fit.intercept + fit.slope * xx, color=BLUE, lw=1.2,
           label=f"Eq. (1): y = {fit.slope:.4f}x − {abs(fit.intercept):.4f}\nR² = {fit.rvalue**2:.4f}")
ax[0].plot([0, 56], [0, 56], color=GREY, lw=0.8, ls="--", label="Identity (y = x)")
ax[0].set_xlabel("Raw GY-302 reading (lx)"); ax[0].set_ylabel("ITR-3000 reference (lx)")
ax[0].set_title("(a) Stage 1 — calibration", fontsize=9, loc="left")
ax[0].legend(fontsize=7, frameon=False, loc="upper left")
ax[1].scatter(r2, cor, s=16, color=BLUE, zorder=3, label="Observed pairs (n = 22)")
ax[1].plot([0, 46], [0, 46], color=GREY, lw=0.8, ls="--", label="Identity (y = x)")
ax[1].set_xlabel("ITR-3000 reference (lx)"); ax[1].set_ylabel("Corrected GY-302 output (lx)")
ax[1].set_title("(b) Stage 2 — independent validation", fontsize=9, loc="left")
ax[1].legend(fontsize=7, frameon=False, loc="upper left")
for a in ax: a.set_xlim(0, None); a.set_ylim(0, None)
fig.tight_layout(); fig.savefig(OUT / "figure1.png", dpi=300); plt.close(fig)

# ---- Figure 2: Bland-Altman (aspect 1.567) ----
d = cor - r2; m = (cor + r2) / 2
bias, sd = d.mean(), d.std(ddof=1); lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
fig, ax = plt.subplots(figsize=(6.3, 4.02))
ax.scatter(m, d, s=22, color=BLUE, zorder=3)
ax.axhline(0, color=GREY, lw=0.6)
ax.axhline(bias, color=BLUE, lw=1.2)
ax.axhline(lo, color=RED, lw=1, ls="--"); ax.axhline(hi, color=RED, lw=1, ls="--")
ax.text(45, bias, f"Mean bias = {bias:.2f} lx", ha="right", va="bottom", fontsize=8, color=BLUE)
ax.text(45, hi, f"Upper LoA = {hi:+.2f} lx", ha="right", va="bottom", fontsize=8, color=RED)
ax.text(45, lo, f"Lower LoA = {lo:+.2f} lx", ha="right", va="top", fontsize=8, color=RED)
ax.set_xlim(0, 45.5); ax.set_ylim(-2.2, 1.4)
ax.set_xlabel("Mean of corrected GY-302 and ITR-3000 readings (lx)")
ax.set_ylabel("Corrected GY-302 − ITR-3000 (lx)")
fig.tight_layout(); fig.savefig(OUT / "figure2.png", dpi=300); plt.close(fig)

# ---- Figure 3: relative direct error (aspect 1.567) ----
pct = 100 * d / r2
xx = np.linspace(1.2, 45, 300)
fig, ax = plt.subplots(figsize=(6.3, 4.02))
ax.fill_between(xx, -6, 6, color=GREY, alpha=0.15, lw=0, label="Reference accuracy, LED (±6 %)")
ax.axhline(20, color=RED, lw=0.9, ls="--"); ax.axhline(-20, color=RED, lw=0.9, ls="--",
                                                        label="BH1750FVI datasheet accuracy (±20 %)")
ax.plot(xx, -100 / xx, color=GREY, lw=0.9, ls=":", label="One 1-lx output step (−100/E %)")
ax.scatter(r2, pct, s=22, color=BLUE, zorder=3, label="Corrected GY-302 (n = 22)")
ax.axhline(0, color=GREY, lw=0.6)
ax.set_xlim(0, 45.5); ax.set_ylim(-50, 25)
ax.set_xlabel("ITR-3000 reference illuminance, E (lx)")
ax.set_ylabel("Direct relative error (%)")
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "figure3.png", dpi=300); plt.close(fig)
print("figures written to", OUT)
