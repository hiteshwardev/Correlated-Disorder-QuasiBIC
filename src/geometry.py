"""Photonic-crystal slab geometries (legume objects). Lengths in units of a."""
import numpy as np
import legume
from . import config as C

R_FLOOR = 0.05


def supercell(N, d, r0, delta_r=None, eps_slab=C.EPS_SLAB, eps_hole=C.EPS_HOLE):
    """N x 1 supercell with hole radii r0 + delta_r.

    The lattice vectors are (N, 0) and (0, 1), so the structure repeats along y
    with period a and the disorder is one-dimensional.
    """
    lattice = legume.Lattice([N, 0], [0, 1])
    phc = legume.PhotCryst(lattice)
    phc.add_layer(d=d, eps_b=eps_slab)
    dr = np.zeros(N) if delta_r is None else np.asarray(delta_r, float)
    if dr.shape != (N,):
        raise ValueError("delta_r must have length N")
    for i in range(N):
        phc.add_shape(legume.Circle(x_cent=-N / 2 + 0.5 + i, y_cent=0.0,
                                    r=max(r0 + dr[i], R_FLOOR), eps=eps_hole))
    return phc


def radius_floor_hits(r0, delta_r):
    return int(np.sum(np.asarray(delta_r, float) + r0 < R_FLOOR))


def unit_cell(d, r0, eps_slab=C.EPS_SLAB, eps_hole=C.EPS_HOLE):
    phc = legume.PhotCryst(legume.Lattice("square"))
    phc.add_layer(d=d, eps_b=eps_slab)
    phc.add_shape(legume.Circle(x_cent=0.0, y_cent=0.0, r=r0, eps=eps_hole))
    return phc


def dimer_cell(d, r0, alpha, eps_slab=C.EPS_SLAB, eps_hole=C.EPS_HOLE):
    """1 x 2 cell with two holes of radii r0 (1 -/+ alpha/2)."""
    phc = legume.PhotCryst(legume.Lattice([1, 0], [0, 2]))
    phc.add_layer(d=d, eps_b=eps_slab)
    for yc, sgn in ((-0.5, -1), (0.5, 1)):
        phc.add_shape(legume.Circle(x_cent=0.0, y_cent=yc,
                                    r=r0 * (1 + sgn * alpha / 2), eps=eps_hole))
    return phc


def stripe_dimer_cell(d, w, delta, eps_slab=C.EPS_SLAB):
    """Period-2 grating of dielectric stripes with widths w (1 -/+ delta/2)."""
    phc = legume.PhotCryst(legume.Lattice([2, 0], [0, 1]))
    phc.add_layer(d=d, eps_b=1.0)
    for xc, ww in ((-0.5, w * (1 - delta / 2)), (0.5, w * (1 + delta / 2))):
        x = [xc - ww / 2, xc + ww / 2, xc + ww / 2, xc - ww / 2]
        y = [-0.5, -0.5, 0.5, 0.5]
        phc.add_shape(legume.Poly(eps=eps_slab, x_edges=x, y_edges=y))
    return phc
