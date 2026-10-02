"""Statistics for skewed loss ensembles and power-law fits."""
import numpy as np
from scipy import stats as sps


def boot_mean(x, n_boot=20000, seed=12345):
    """Mean with a bootstrap standard error and 95 per cent percentile interval."""
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bm = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return dict(mean=float(x.mean()), sem=float(bm.std(ddof=1)),
                lo=float(np.percentile(bm, 2.5)), hi=float(np.percentile(bm, 97.5)),
                n=int(len(x)))


def ratio_ci(a, b, paired=False, statistic="mean", n_boot=20000, seed=777):
    """Ratio stat(a) / stat(b) with a 95 per cent percentile bootstrap interval.

    paired=True resamples realisation indices jointly (common random numbers);
    otherwise the two ensembles are resampled independently. se_log is the
    bootstrap standard deviation of the log ratio.
    """
    f = np.mean if statistic == "mean" else np.median
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    if paired:
        n = min(len(a), len(b))
        a, b = a[:n], b[:n]
        idx = rng.integers(0, n, size=(n_boot, n))
        ra, rb = f(a[idx], axis=1), f(b[idx], axis=1)
    else:
        ra = f(a[rng.integers(0, len(a), size=(n_boot, len(a)))], axis=1)
        rb = f(b[rng.integers(0, len(b), size=(n_boot, len(b)))], axis=1)
    r = ra / rb
    return dict(ratio=float(f(a) / f(b)), lo=float(np.percentile(r, 2.5)),
                hi=float(np.percentile(r, 97.5)),
                se_log=float(np.std(np.log(r), ddof=1)), n_a=int(len(a)), n_b=int(len(b)))


def pool_log_ratios(log_r, se):
    """Inverse-variance pooling of log ratios with Cochran's heterogeneity test
    and the DerSimonian-Laird random-effects estimate."""
    log_r, se = np.asarray(log_r, float), np.asarray(se, float)
    w = 1.0 / se ** 2
    fixed = float(np.sum(w * log_r) / np.sum(w))
    Q = float(np.sum(w * (log_r - fixed) ** 2))
    dof = len(log_r) - 1
    tau2 = max(0.0, (Q - dof) / (np.sum(w) - np.sum(w ** 2) / np.sum(w)))
    w_re = 1.0 / (se ** 2 + tau2)
    rand = float(np.sum(w_re * log_r) / np.sum(w_re))
    return dict(fixed=fixed, se_fixed=float(1 / np.sqrt(np.sum(w))),
                random=rand, se_random=float(1 / np.sqrt(np.sum(w_re))),
                Q=Q, dof=dof, p_heterogeneity=float(sps.chi2.sf(Q, dof)),
                tau2=float(tau2), I2=float(max(0.0, (Q - dof) / Q)) if Q > 0 else 0.0)


def dist_summary(x):
    """Shape of a positive, right-skewed ensemble. For a sum of k independent
    squared Gaussian amplitudes (chi-squared with k degrees of freedom)
    CV = sqrt(2 / k), so k_eff = 2 / CV^2 counts effective radiation channels."""
    x = np.asarray(x, float)
    m, sd = float(x.mean()), float(x.std(ddof=1))
    cv = sd / m
    return dict(n=int(len(x)), mean=m, median=float(np.median(x)), sd=sd, cv=float(cv),
                skew=float(sps.skew(x)), k_eff=float(2.0 / cv ** 2),
                max_over_mean=float(x.max() / m))


def wls_loglog(x, y, ysig):
    """Weighted fit of log y = b log x + a with weights (y / ysig)^2.

    Returns the slope b, its nominal standard error, chi2 per degree of freedom
    and se_inflated = se * sqrt(max(chi2/dof, 1)), the value to quote when the
    points scatter more than their error bars allow.
    """
    x, y, ysig = (np.asarray(v, float) for v in (x, y, ysig))
    lx, ly = np.log(x), np.log(y)
    w = (y / ysig) ** 2
    W = w.sum()
    xb, yb = (w * lx).sum() / W, (w * ly).sum() / W
    Sxx = (w * (lx - xb) ** 2).sum()
    b = (w * (lx - xb) * (ly - yb)).sum() / Sxx
    a = yb - b * xb
    se = 1.0 / np.sqrt(Sxx)
    dof = max(len(lx) - 2, 1)
    chi2 = float((w * (ly - a - b * lx) ** 2).sum())
    return dict(slope=float(b), intercept=float(a), se=float(se), chi2=chi2, dof=int(dof),
                chi2_red=float(chi2 / dof), p_value=float(sps.chi2.sf(chi2, dof)),
                se_inflated=float(se * np.sqrt(max(chi2 / dof, 1.0))), n=int(len(lx)))


def local_slopes(x, y):
    """Point-to-point log-log slopes; a pure power law gives a constant."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    s = np.diff(np.log(y)) / np.diff(np.log(x))
    return dict(slopes=[float(v) for v in s], drift=float(s.max() - s.min()))


def powerlaw_fit(x, y):
    """Unweighted log-log fit; misfit_factor = exp(max |residual|)."""
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    c, *_ = np.linalg.lstsq(A, ly, rcond=None)
    r = ly - A @ c
    return dict(slope=float(c[0]), misfit_factor=float(np.exp(np.abs(r).max())))
