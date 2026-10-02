"""Rigorous coupled-wave reference for the stripe-dimer grating (grcwa).

Normal incidence, electric field along the stripes. The quality factor follows
from a Fano fit of the reflectance around the resonance. The permittivity grid
uses sub-pixel averaging, so stripe widths are represented exactly rather than
rounded to whole pixels; rounding would change the width asymmetry delta, and
with it Q ~ delta^-2, by an amount that grows as delta decreases.
"""
import numpy as np
import grcwa
from scipy.optimize import curve_fit


def fano(f, f0, gamma, q, A, B):
    """Fano lineshape; gamma is the full width at half maximum."""
    e = 2 * (f - f0) / gamma
    return A * (q + e) ** 2 / (1 + e ** 2) + B


def grating_profile(w, delta, eps_slab, nx=1024):
    """Permittivity of the period-2 grating on nx pixels spanning x in [-1, 1),
    each pixel holding the area-weighted average of the two media. For the
    electric field along the stripes this average is the exact effective medium."""
    edges = np.linspace(-1.0, 1.0, nx + 1)
    fill = np.zeros(nx)
    for xc, ww in ((-0.5, w * (1 - delta / 2)), (0.5, w * (1 + delta / 2))):
        lo, hi = xc - ww / 2, xc + ww / 2
        fill += np.clip(np.minimum(edges[1:], hi) - np.maximum(edges[:-1], lo), 0.0, None)
    fill /= edges[1] - edges[0]
    return 1.0 + (eps_slab - 1.0) * fill


def reflectance(freqs, d, w, delta, eps_slab, n_orders=101, nx=1024, ny=16):
    grid = np.tile(grating_profile(w, delta, eps_slab, nx)[:, None], (1, ny))
    R = np.empty(len(freqs))
    for i, f in enumerate(freqs):
        obj = grcwa.obj(n_orders, [2.0, 0.0], [0.0, 1.0], float(f), 1e-9, 0.0, verbose=0)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Add_LayerGrid(d, nx, ny)
        obj.Add_LayerUniform(1.0, 1.0)
        obj.Init_Setup()
        obj.GridLayer_geteps(grid)
        obj.MakeExcitationPlanewave(0.0, 0.0, 1.0, 0.0, order=0)
        R[i], _ = obj.RT_Solve(normalize=1)
    return R


def fit_fano(freqs, R):
    """Fano fit of a reflectance trace; returns (Q, f0, fwhm, rms residual)."""
    bg = np.median(R)
    j = int(np.argmax(np.abs(R - bg)))
    prom = np.abs(R - bg)
    half = prom > prom[j] / 2
    width = max(freqs[half].max() - freqs[half].min(), 2 * (freqs[1] - freqs[0]))
    best = None
    for q0 in (1.7, -1.7, 0.5, -0.5):
        try:
            p, _ = curve_fit(fano, freqs, R, p0=[freqs[j], width, q0, np.ptp(R) / 2, R.min()],
                             maxfev=50000)
        except RuntimeError:
            continue
        res = float(np.sqrt(np.mean((fano(freqs, *p) - R) ** 2)))
        if best is None or res < best[0]:
            best = (res, p)
    if best is None:
        raise RuntimeError("Fano fit did not converge")
    f0, gamma = best[1][0], abs(best[1][1])
    return float(f0 / gamma), float(f0), float(gamma), best[0]


def resonance_q(f_guess, q_guess, d, w, delta, eps_slab, n_orders=101, n1=220, n2=140,
                span=40.0):
    """Locate the resonance near f_guess and return the Fano fit.

    A first sweep covers +/- span linewidths (estimated from q_guess) around
    f_guess; a second sweep of +/- 8 fitted linewidths is centred on the feature.
    """
    lw = f_guess / q_guess
    f1 = np.linspace(f_guess - span * lw, f_guess + span * lw, n1)
    R1 = reflectance(f1, d, w, delta, eps_slab, n_orders)
    Q1, fc, g1, _ = fit_fano(f1, R1)
    half = max(8 * g1, 6 * (f1[1] - f1[0]))
    f2 = np.linspace(fc - half, fc + half, n2)
    return fit_fano(f2, reflectance(f2, d, w, delta, eps_slab, n_orders))
