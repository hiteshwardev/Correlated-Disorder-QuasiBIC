"""Guided-mode expansion runs and radiative-loss estimators.

legume solves the Hermitian eigenproblem for the magnetic field expanded in the
guided modes of an effective homogeneous slab; the imaginary part of each
eigenfrequency follows from first-order (golden-rule) coupling of that mode to
the radiative continuum. Eigenvectors come from a dense Hermitian solver and are
orthonormal, which the projected estimator relies on.
"""
import numpy as np
import legume
from . import config as C, disorder, geometry


def average_eps(phc):
    return float(phc.layers[0].eps_avg)


def run(phc, kpoints, gmode, numeig, gmax=C.GMAX_PROD, compute_im=True,
        eps_eff=None, eig_sigma=0.0):
    """Run legume on `phc`. If eps_eff is given, the guided-mode basis is built
    for that fixed slab permittivity, so clean and disordered supercells share
    one expansion basis."""
    kp = np.asarray(kpoints, float)
    if kp.ndim == 1:
        kp = kp.reshape(2, 1)
    elif kp.shape[0] != 2 and kp.shape[1] == 2:
        kp = kp.T
    solver = legume.GuidedModeExp(phc, gmax=gmax)
    options = dict(kpoints=kp, gmode_inds=list(gmode), numeig=int(numeig),
                   compute_im=bool(compute_im), verbose=False,
                   eig_sigma=float(eig_sigma))
    if eps_eff is not None:
        phc.layers[0].eps_eff = float(eps_eff)
        for cladding in phc.claddings:
            cladding.eps_eff = float(getattr(cladding, "eps_avg",
                                             getattr(cladding, "eps_b", 1.0)))
        options["eps_eff"] = "custom"
    solver.run(**options)
    return solver


def quality_factors(solver, ik=0):
    f = np.array(solver.freqs[ik])
    im = np.array(solver.freqs_im[ik])
    with np.errstate(divide="ignore"):
        return np.where(im > 0, f / (2 * im), np.inf)


def band_q(d, r, gmode, ks, band=C.SCAN_BAND, gmax=C.GMAX_PROD):
    """Frequency and Q of one band of the unit cell along Gamma-X at wavevectors ks."""
    ks = np.asarray(ks, float)
    solver = run(geometry.unit_cell(d, r), np.vstack([ks, np.zeros_like(ks)]),
                 gmode, band + 2, gmax=gmax)
    f = np.array(solver.freqs)[:, band]
    im = np.array(solver.freqs_im)[:, band]
    with np.errstate(divide="ignore"):
        Q = np.where(im > 0, f / (2 * im), np.inf)
    return f, im, Q


def find_mode(solver, f_window, select="max_q", f_ref=None, ik=0):
    """Index of the quasi-BIC inside f_window.

    select="max_q" takes the highest-Q mode (used whenever the basis is fixed).
    select="nearest_f" takes the mode closest to f_ref; it is needed when the
    guided-mode basis changes, because enrichment can place a different high-Q
    mode inside the window.
    """
    f = np.array(solver.freqs[ik])
    Q = quality_factors(solver, ik)
    mask = (f > C.F_MIN) & (f >= f_window[0]) & (f <= f_window[1])
    if not mask.any():
        raise RuntimeError("no mode inside the frequency window")
    if select == "nearest_f":
        return int(np.argmin(np.where(mask, np.abs(f - f_ref), np.inf)))
    return int(np.argmax(np.where(mask, np.where(np.isfinite(Q), Q, 1e32), -1.0)))


def loss_estimators(clean, dis, n_mode, ik=0):
    """Projected (E1) and maximum-overlap (E2) inverse quality factors.

    E1 = 2 sum_n |c_n|^2 Im f_n / (Re f_0 sum_n |c_n|^2), c_n = <v_clean|v_n>, is the
    decay rate of the clean mode expanded in the disordered eigenbasis.
    E2 = 2 Im f_j / Re f_j for the mode j with the largest overlap.
    Returns (E1, E2, sum_n |c_n|^2).
    """
    vc = np.array(clean.eigvecs[ik])[:, n_mode]
    c2 = np.abs(vc.conj() @ np.array(dis.eigvecs[ik])) ** 2
    f0 = float(np.array(clean.freqs[ik])[n_mode])
    im = np.array(dis.freqs_im[ik])
    weight = float(c2.sum())
    e1 = float(2.0 * np.sum(c2 * im) / max(weight, 1e-300) / f0)
    j = int(np.argmax(c2))
    f = np.array(dis.freqs[ik])
    e2 = float(2.0 * im[j] / f[j]) if f[j] > 0 else float("nan")
    return e1, e2, weight


class Reference:
    """Clean N x 1 supercell of a structure in config.STRUCTURES: solver, mode
    index, frequency, clean inverse Q and the pinned slab permittivity."""

    def __init__(self, tag, N, gmax=C.GMAX_PROD, gmode=None, select="max_q"):
        cfg = C.STRUCTURES[tag]
        self.tag, self.N, self.gmax = tag, N, gmax
        self.gmode = list(cfg["gmode"] if gmode is None else gmode)
        phc = geometry.supercell(N, cfg["d"], cfg["r"])
        self.eps_eff = average_eps(phc)
        self.solver = run(phc, [list(C.KPOINT_GAMMA)], self.gmode, N + C.NUMEIG_PAD,
                          gmax=gmax, eps_eff=self.eps_eff, eig_sigma=cfg["f0"])
        window = (cfg["f0"] - cfg["f_win"], cfg["f0"] + cfg["f_win"])
        self.mode = find_mode(self.solver, window, select=select, f_ref=cfg["f0"])
        self.f0 = float(np.array(self.solver.freqs[0])[self.mode])
        if abs(self.f0 - cfg["f0"]) > C.F0_TOL:
            raise RuntimeError(f"{tag}: mode at f = {self.f0:.4f}, expected {cfg['f0']}")
        self.inv_q = float(2 * np.array(self.solver.freqs_im[0])[self.mode] / self.f0)


def realization(ref, sigma, lc, seed):
    """Loss of one disordered supercell. The expansion basis is enlarged (at most
    twice) if the clean mode keeps less than OVERLAP_GUARD of its weight."""
    cfg = C.STRUCTURES[ref.tag]
    dr = disorder.sample(ref.N, sigma, lc, np.random.default_rng(seed))
    numeig = ref.N + C.NUMEIG_PAD
    for attempt in range(3):
        phc = geometry.supercell(ref.N, cfg["d"], cfg["r"], dr)
        dis = run(phc, [list(C.KPOINT_GAMMA)], ref.gmode, numeig, gmax=ref.gmax,
                  eps_eff=ref.eps_eff, eig_sigma=cfg["f0"])
        e1, e2, weight = loss_estimators(ref.solver, dis, ref.mode)
        if weight >= C.OVERLAP_GUARD or attempt == 2:
            break
        numeig += C.NUMEIG_PAD
    return dict(seed=int(seed), numeig=int(numeig), inv_q_proj=e1, inv_q_mode=e2,
                overlap_sum=weight,
                radius_floor_hits=geometry.radius_floor_hits(cfg["r"], dr))


def outside_block_weight(solver, n_mode, block_size, ik=0):
    """Weight of eigenvector n_mode outside its first block_size components,
    i.e. on guided modes added beyond the reference basis."""
    v = np.array(solver.eigvecs[ik])[:, n_mode]
    return float(np.sum(np.abs(v[block_size:]) ** 2) / np.sum(np.abs(v) ** 2))
