#!/usr/bin/env python3
"""
Calibration and agreement analysis of the GY-302 (BH1750FVI) light sensor.

Data source: Carvalho Neto (dissertation, FCA/Unesp), Tables 5 and 6.
  Stage 1 (Table 5, n=26): raw GY-302 readings vs ITR-3000 reference.
  Stage 2 (Table 6, n=22): firmware-corrected GY-302 readings vs ITR-3000.

Firmware correction actually implemented (dissertation, Appendix code):
    uint16_t lux = lightMeter.readLightLevel();
    int lux2 = 0.7777*lux - 0.4562;   // C float->int conversion TRUNCATES
So the reported corrected value is trunc(0.7777*raw - 0.4562).

Answers reviewer requests R1.1, R2.1, R2.2, R2.5: direct errors
(corrected - reference) WITHOUT refitting a second regression, Bland-Altman
agreement, same-pair raw vs corrected comparison (LOOCV on Stage 1),
and a GUM-style uncertainty budget using the ITR-3000 manual spec
(+/-6 % of reading for non-incandescent sources, i.e. LED).

Usage: python reanalysis_direct_errors.py  (paths are relative to this file)
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "results"
OUT.mkdir(exist_ok=True)

A_FW, B_FW = 0.7777, -0.4562          # Eq. (9) as hard-coded in firmware
REF_ACC_LED = 0.06                    # ITR-3000 manual: +/-6 % other light sources
REF_ACC_INC = 0.03                    # +/-3 % incandescent 2854 K (for reference)
QUANT_LX = 1.0                        # BH1750 high-res mode / int output: 1 lx step


def fw_trunc(x):
    """C-style float->int conversion (truncation toward zero)."""
    return np.trunc(x)


def agreement(pred, ref):
    """Direct-error agreement statistics, e = pred - ref."""
    pred, ref = np.asarray(pred, float), np.asarray(ref, float)
    e = pred - ref
    n = len(e)
    bias = e.mean()
    sd = e.std(ddof=1)
    t = stats.t.ppf(0.975, n - 1)
    se_bias = sd / math.sqrt(n)
    loa_lo, loa_hi = bias - 1.96 * sd, bias + 1.96 * sd
    se_loa = math.sqrt(3 * sd**2 / n)          # Bland & Altman (1999) approx.
    mean_pair = (pred + ref) / 2
    # proportional bias: regress difference on pair mean
    pb = stats.linregress(mean_pair, e)
    pct = 100 * e / ref
    return {
        "n": n,
        "bias_lx": bias,
        "bias_95CI_lx": [bias - t * se_bias, bias + t * se_bias],
        "bias_ttest_p": float(stats.ttest_1samp(e, 0).pvalue),
        "sd_diff_lx": sd,
        "MAE_lx": np.abs(e).mean(),
        "RMSE_lx": math.sqrt((e**2).mean()),
        "max_abs_err_lx": np.abs(e).max(),
        "MAPE_pct": np.abs(pct).mean(),
        "max_abs_pct": np.abs(pct).max(),
        "LoA_lx": [loa_lo, loa_hi],
        "LoA_lower_95CI": [loa_lo - t * se_loa, loa_lo + t * se_loa],
        "LoA_upper_95CI": [loa_hi - t * se_loa, loa_hi + t * se_loa],
        "prop_bias_slope": pb.slope,
        "prop_bias_p": pb.pvalue,
        "n_within_ref_LED_spec_6pct": int((np.abs(e) <= REF_ACC_LED * ref).sum()),
        "n_within_20pct": int((np.abs(pct) <= 20).sum()),
    }


def ols(x, y):
    r = stats.linregress(x, y)
    yhat = r.intercept + r.slope * np.asarray(x)
    resid = np.asarray(y) - yhat
    n = len(x)
    return {
        "slope": r.slope, "intercept": r.intercept, "R2": r.rvalue**2,
        "slope_SE": r.stderr, "intercept_SE": r.intercept_stderr,
        "resid_RMSE_lx": math.sqrt((resid**2).mean()),
        "resid_SER_lx": math.sqrt((resid**2).sum() / (n - 2)),
        "resid_MAE_lx": np.abs(resid).mean(),
    }


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    return o


def main():
    s1 = pd.read_csv(DATA / "calibration_stage1_table5.csv")
    s2 = pd.read_csv(DATA / "validation_stage2_table6.csv")
    raw, ref1 = s1.gy302_raw_lx.values.astype(float), s1.ref_itr3000_lx.values
    cor2, ref2 = s2.gy302_corrected_lx.values.astype(float), s2.ref_itr3000_lx.values

    assert (s1.gy302_raw_lx == s1.gy302_raw_movavg_lx).all()
    assert (s2.gy302_corrected_lx == s2.gy302_corrected_movavg_lx).all()

    res = {}

    # ---------- Stage 1: calibration fit (reproduces Eq. 9) ----------
    fit1 = ols(raw, ref1)
    res["stage1_fit_ref_on_raw"] = fit1

    # ---------- Stage 1: raw sensor direct error ----------
    res["stage1_raw_direct"] = agreement(raw, ref1)

    # ---------- Stage 1: in-sample corrected (float and firmware-truncated) ----------
    res["stage1_insample_eq9_float"] = agreement(A_FW * raw + B_FW, ref1)
    res["stage1_insample_eq9_fw_trunc"] = agreement(fw_trunc(A_FW * raw + B_FW), ref1)
    res["stage1_insample_eq9_round"] = agreement(np.floor(A_FW * raw + B_FW + 0.5), ref1)

    # ---------- Stage 1: leave-one-out CV on the SAME pairs (R2.2) ----------
    loo_f, loo_t, loo_r = np.empty_like(ref1), np.empty_like(ref1), np.empty_like(ref1)
    for i in range(len(raw)):
        m = np.ones(len(raw), bool); m[i] = False
        r = stats.linregress(raw[m], ref1[m])
        loo_f[i] = r.intercept + r.slope * raw[i]
        loo_t[i] = fw_trunc(loo_f[i])
        loo_r[i] = np.floor(loo_f[i] + 0.5)   # round-to-nearest
    res["stage1_LOOCV_float"] = agreement(loo_f, ref1)
    res["stage1_LOOCV_fw_trunc"] = agreement(loo_t, ref1)
    res["stage1_LOOCV_round"] = agreement(loo_r, ref1)

    # ---------- Stage 2: DIRECT errors, no refit (R1.1, R2.1) ----------
    res["stage2_direct_fw_output"] = agreement(cor2, ref2)
    # Constant-offset sensitivity: +0.5 lx is the EXPECTED effect of rounding vs
    # truncation. Exact rounding cannot be simulated on Stage 2 (raw readings not
    # recorded); it is simulated exactly on Stage 1 above. Not measured data.
    res["stage2_sensitivity_round_equiv"] = agreement(cor2 + 0.5, ref2)
    d2 = cor2 - ref2
    res["stage2_lag1_autocorr_differences"] = float(np.corrcoef(d2[:-1], d2[1:])[0, 1])

    # ---------- Stage 2: old regression (for traceability with manuscript) ----------
    res["stage2_old_refit_ref_on_corrected"] = ols(cor2, ref2)

    # ---------- Uncertainty budget (GUM, k=2) ----------
    sd2 = res["stage2_direct_fw_output"]["sd_diff_lx"]
    budget = []
    for L in [2, 5, 10, 20, 30, 40]:
        u_ref = REF_ACC_LED * L / math.sqrt(3)          # Type B, rectangular
        u_q = QUANT_LX / math.sqrt(12)                  # sensor quantization
        u_A = sd2                                       # Type A: dispersion of Stage-2 differences
        # u_A already contains quantization + reference repeatability; u_q listed for info only
        u_c = math.sqrt(u_A**2 + u_ref**2)
        budget.append({"E_lx": L, "u_ref_typeB_lx": u_ref, "u_quant_lx_info": u_q,
                       "u_typeA_lx": u_A, "u_c_lx": u_c, "U_k2_lx": 2 * u_c,
                       "U_k2_pct": 200 * u_c / L})
    res["uncertainty_budget_corrected_sensor"] = budget

    # ---------- range breakdown (low vs high illuminance) ----------
    lo, hi = ref2 < 10, ref2 >= 10
    res["stage2_direct_ref_lt_10lx"] = agreement(cor2[lo], ref2[lo])
    res["stage2_direct_ref_ge_10lx"] = agreement(cor2[hi], ref2[hi])

    res = clean(res)
    (OUT / "metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- per-point table ----------
    t2 = s2.copy()
    t2["direct_error_lx"] = cor2 - ref2
    t2["direct_error_pct"] = 100 * (cor2 - ref2) / ref2
    t2["ref_LED_tolerance_lx(+/-6%)"] = REF_ACC_LED * ref2
    t2["within_ref_tolerance"] = np.abs(cor2 - ref2) <= REF_ACC_LED * ref2
    old = stats.linregress(cor2, ref2)
    t2["OLD_stage2_refit_residual_lx"] = ref2 - (old.intercept + old.slope * cor2)
    t1 = s1.copy()
    t1["raw_direct_error_lx"] = raw - ref1
    t1["eq9_float_lx"] = A_FW * raw + B_FW
    t1["eq9_fw_trunc_lx"] = fw_trunc(A_FW * raw + B_FW)
    t1["LOOCV_pred_float_lx"] = loo_f
    t1["LOOCV_error_float_lx"] = loo_f - ref1
    t1["LOOCV_pred_trunc_lx"] = loo_t
    t1["LOOCV_error_trunc_lx"] = loo_t - ref1
    t1.round(4).to_csv(OUT / "stage1_pointwise.csv", index=False)
    t2.round(4).to_csv(OUT / "stage2_pointwise.csv", index=False)

    # ---------- figures ----------
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    # Fig A: Bland-Altman Stage 2
    a = res["stage2_direct_fw_output"]
    m = (cor2 + ref2) / 2; d = cor2 - ref2
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(m, d, s=22, color="#1f4e79", zorder=3)
    ax.axhline(a["bias_lx"], color="#1f4e79", lw=1.2, label=f"Mean bias = {a['bias_lx']:.2f} lx")
    for v in a["LoA_lx"]:
        ax.axhline(v, color="#c0504d", lw=1, ls="--")
    ax.axhline(0, color="grey", lw=0.6)
    xx = np.linspace(0, 45, 100)
    ax.plot(xx, -REF_ACC_LED * xx, color="grey", ls=":", lw=1, label="Reference accuracy ±6% (LED)")
    ax.plot(xx, REF_ACC_LED * xx, color="grey", ls=":", lw=1)
    ax.text(44, a["LoA_lx"][1], f"+1.96 SD = {a['LoA_lx'][1]:.2f}", ha="right", va="bottom", fontsize=8, color="#c0504d")
    ax.text(44, a["LoA_lx"][0], f"−1.96 SD = {a['LoA_lx'][0]:.2f}", ha="right", va="top", fontsize=8, color="#c0504d")
    ax.set_xlabel("Mean of corrected GY-302 and ITR-3000 (lx)")
    ax.set_ylabel("Corrected GY-302 − ITR-3000 (lx)")
    ax.set_xlim(0, 45); ax.legend(loc="lower left", fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig(OUT / "fig_bland_altman_stage2.png", dpi=300); plt.close(fig)

    # Fig B: same-pair comparison (Stage 1) raw vs LOOCV-corrected
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].scatter(ref1, raw, s=18, label="Raw GY-302", color="#c0504d")
    ax[0].scatter(ref1, loo_t, s=18, label="Corrected (LOOCV, firmware truncation)", color="#1f4e79")
    ax[0].plot([0, 55], [0, 55], color="grey", lw=0.8)
    ax[0].set_xlabel("ITR-3000 reference (lx)"); ax[0].set_ylabel("Sensor output (lx)")
    ax[0].legend(fontsize=8, frameon=False); ax[0].set_title("(a) Stage 1, same 26 pairs", fontsize=9)
    ax[1].scatter(ref2, cor2, s=18, color="#1f4e79")
    ax[1].plot([0, 45], [0, 45], color="grey", lw=0.8)
    ax[1].set_xlabel("ITR-3000 reference (lx)"); ax[1].set_ylabel("Corrected GY-302 (lx)")
    ax[1].set_title("(b) Stage 2, independent 22 pairs", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "fig_raw_vs_corrected.png", dpi=300); plt.close(fig)

    # ---------- spreadsheet ----------
    rows = []
    for k in ["stage1_raw_direct", "stage1_insample_eq9_fw_trunc", "stage1_LOOCV_float",
              "stage1_LOOCV_fw_trunc", "stage1_LOOCV_round", "stage2_direct_fw_output", "stage2_direct_ref_lt_10lx",
              "stage2_direct_ref_ge_10lx", "stage2_sensitivity_round_equiv"]:
        r = res[k]
        rows.append({"analysis": k, "n": r["n"], "bias_lx": r["bias_lx"],
                     "bias_95CI": f"{r['bias_95CI_lx'][0]:.3f} to {r['bias_95CI_lx'][1]:.3f}",
                     "SD_lx": r["sd_diff_lx"], "MAE_lx": r["MAE_lx"], "RMSE_lx": r["RMSE_lx"],
                     "max_abs_err_lx": r["max_abs_err_lx"], "MAPE_pct": r["MAPE_pct"],
                     "LoA_lx": f"{r['LoA_lx'][0]:.3f} to {r['LoA_lx'][1]:.3f}",
                     "prop_bias_slope": r["prop_bias_slope"], "prop_bias_p": r["prop_bias_p"],
                     "n_within_±6%_ref": r["n_within_ref_LED_spec_6pct"]})
    readme = pd.DataFrame({"item": [
        "Source", "Stage 1", "Stage 2", "Firmware correction", "Reference instrument",
        "Error definition", "Sensitivity row", "Script"], "description": [
        "Carvalho Neto, dissertation FCA/Unesp, Tables 5 and 6 (transcribed verbatim).",
        "Raw GY-302 (with/without moving average: identical) vs ITR-3000, n=26.",
        "Firmware-corrected GY-302 vs ITR-3000, independent run, n=22.",
        "int lux2 = 0.7777*lux - 0.4562;  C float->int conversion truncates toward zero.",
        "Instrutherm/Instrubras ITR-3000: accuracy ±3% (incandescent 2854 K), ±6% other sources (LED); resolution 0.01 lx (20 lx range).",
        "Direct error e = sensor - reference; no second regression is fitted.",
        "stage1_LOOCV_round simulates round-to-nearest exactly on Stage 1. stage2_sensitivity_round_equiv adds a constant +0.5 lx (expected effect of rounding): sensitivity analysis, not measured data.",
        "code/reanalysis_direct_errors.py"]})
    with pd.ExcelWriter(OUT / "GY302_calibration_dataset_and_metrics.xlsx") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        t1.round(4).to_excel(xw, sheet_name="Stage1_Table5", index=False)
        t2.round(4).to_excel(xw, sheet_name="Stage2_Table6", index=False)
        pd.DataFrame(rows).round(4).to_excel(xw, sheet_name="Agreement_metrics", index=False)
        pd.DataFrame(res["uncertainty_budget_corrected_sensor"]).round(4).to_excel(xw, sheet_name="Uncertainty_budget", index=False)
        pd.DataFrame([dict(fit="stage1 ref~raw (Eq.9)", **res["stage1_fit_ref_on_raw"]),
                      dict(fit="stage2 ref~corrected (OLD manuscript, not an accuracy metric)",
                           **res["stage2_old_refit_ref_on_corrected"])]).round(5).to_excel(xw, sheet_name="Regression_fits", index=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
