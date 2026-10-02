"""Spectral-overlap model of disorder-induced radiative loss.

Radius disorder with power spectrum S_m couples the Gamma-point mode of an
N x 1 supercell to the folded Bloch modes at k_m = 2 pi min(m, N - m) / N.

  D = (2 / f0) sum_{0 < k_m < k0} S_m                                   (direct)
  I = (2 / f0) sum_{m != 0} S_m g_m / ((f0 - f_m)^2 + g_m^2 / 4)       (indirect)

with k0 = 2 pi f0 the light line, f_m the clean-band frequencies and g_m = 2 Im f_m
their full widths. The model is <1/Q> - 1/Q_clean = A_d D + A_i I. It assumes a
direct coupling independent of m inside the light cone and an indirect matrix
element |V_m|^2 = A_i S_m with one constant A_i for every m.
"""
import numpy as np
from . import disorder


def folded_k(N):
    m = np.arange(N)
    return 2 * np.pi * np.minimum(m, N - m) / N


def lorentzian_terms(N, f0, f_m, g_m):
    """g_m / ((f0 - f_m)^2 + g_m^2 / 4) for m = 0..N-1 (zero at m = 0)."""
    out = np.zeros(N)
    m = np.arange(1, N)
    out[m] = g_m[m] / ((f0 - f_m[m]) ** 2 + g_m[m] ** 2 / 4)
    return out


def channels(N, sigma, lc, f0, f_m, g_m):
    """Direct and indirect channel weights (D, I)."""
    S = disorder.psd(N, sigma, lc)
    k = folded_k(N)
    inside = (k > 0) & (k < 2 * np.pi * f0)
    D = float(2.0 / f0 * S[inside].sum())
    I = float(2.0 / f0 * np.sum(S * lorentzian_terms(N, f0, f_m, g_m)))
    return D, I


def ratio_prediction(N, lc, f0, f_m, g_m):
    """Parameter-free prediction I(lc) / I(0) of the enhancement over white
    disorder; the amplitude A_i and sigma cancel."""
    return channels(N, 1.0, lc, f0, f_m, g_m)[1] / channels(N, 1.0, 0.0, f0, f_m, g_m)[1]


def fit_one_channel(points):
    """Weighted least squares for y = A_i I. Returns (A_i, se, chi2, dof)."""
    I = np.array([p["I"] for p in points], float)
    y = np.array([p["y"] for p in points], float)
    w = 1.0 / np.array([p["ysem"] for p in points], float) ** 2
    A = float(np.sum(w * I * y) / np.sum(w * I * I))
    chi2 = float(np.sum(w * (y - A * I) ** 2))
    return A, float(1.0 / np.sqrt(np.sum(w * I * I))), chi2, max(len(y) - 1, 1)


def fit_two_channel(points):
    """Unconstrained weighted least squares for y = A_d D + A_i I.
    Returns (A_d, A_i, se_d, se_i, condition number of the weighted design)."""
    s = np.array([p["ysem"] for p in points], float)
    M = np.array([[p["D"], p["I"]] for p in points], float) / s[:, None]
    y = np.array([p["y"] for p in points], float) / s
    coef, *_ = np.linalg.lstsq(M, y, rcond=None)
    cov = np.linalg.inv(M.T @ M)
    return (float(coef[0]), float(coef[1]), float(np.sqrt(cov[0, 0])),
            float(np.sqrt(cov[1, 1])), float(np.linalg.cond(M)))
