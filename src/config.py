"""Parameters of the study.

Units: lattice constant a = 1 and c = 1, so the reduced frequency is
f = omega a / (2 pi c) and in-plane wavevectors are in rad/a. Radii, slab
thicknesses, correlation lengths and disorder amplitudes are in units of a.
"""

EPS_SLAB = 12.0
EPS_HOLE = 1.0

# Structures studied: square lattice of air holes of radius r in a slab of
# thickness d. Band 1 (counting from 0) of the TE-like guided-mode basis hosts a
# symmetry-protected bound state at Gamma near f0. Both geometries are part of
# the parameter survey.
STRUCTURES = {
    "A": dict(structure_id=1, d=0.782, r=0.18, gmode=[0], band=1,
              f0=0.3240, f_win=0.020),
    "B": dict(structure_id=2, d=0.9225, r=0.18, gmode=[0], band=1,
              f0=0.3178, f_win=0.012),
}
F0_TOL = 0.005

# Parameter survey: unit cell, band SCAN_BAND, basis of the fundamental TE-like
# or TM-like guided mode.
R_SCAN = (0.14, 0.18, 0.22, 0.26, 0.30)
D_SCAN = tuple(sorted({round(0.60 + 0.04 * i, 3) for i in range(9)} | {0.782, 0.9225}))
GMODE_SURVEY = ([0], [1])
SCAN_BAND = 1

# Plane-wave cutoff (units of 2 pi / a) and guided-mode basis.
GMAX_LADDER = (4.001, 5.001, 6.001, 7.001, 8.001)
GMAX_PROD = 6.001
GMAX_CHECK = 8.001
GMODE_PROD = [0]
GMODE_ENRICHED = [0, 2]
KPOINT_GAMMA = (1e-4, 0.0)
NUMEIG_PAD = 8
OVERLAP_GUARD = 0.95
F_MIN = 0.02

# Disorder and ensemble design.
SIGMAS = (0.005, 0.01, 0.02)
LCS = (0.0, 0.5, 1.0, 2.0)
N_MAIN = 12
N_SEEDS_MAIN = 40
N_SIZES = (10, 12, 16)
SIGMA_SIZE = 0.005
N_SEEDS_SIZE = 20
N_SEEDS_CHECK = 10
N_SEEDS_LADDER = 10
LADDER_SEED0 = 900000
N_BOOT = 20000

GRID_INDEX_MAX = 100
REPLICATE_MAX = 1000


def seed_for(structure_id, grid_index, replicate):
    """Integer seed with a disjoint block for every (structure, grid point)."""
    if not 0 <= grid_index < GRID_INDEX_MAX:
        raise ValueError(f"grid index {grid_index} outside [0, {GRID_INDEX_MAX})")
    if not 0 <= replicate < REPLICATE_MAX:
        raise ValueError(f"replicate {replicate} outside [0, {REPLICATE_MAX})")
    return 100000 * structure_id + 1000 * grid_index + replicate


def lc_allowed(lc, N):
    """Periodisation guard: lc <= N / (2 sqrt 6) keeps the wrap-around covariance
    C(N/2) below exp(-3) sigma^2, i.e. below 5 per cent of the variance."""
    return lc == 0.0 or lc <= N / (2 * 6 ** 0.5)


def main_grid():
    """(grid_index, sigma, lc, N) for the amplitude x correlation-length grid."""
    pts = [(s, lc, N_MAIN) for s in SIGMAS for lc in LCS]
    return [(g, *p) for g, p in enumerate(pts)]


def size_grid():
    """(grid_index, sigma, lc, N) for the supercell-size series.

    Seed blocks 50-61 are reserved for this series. Points with N = N_MAIN
    coincide with the main grid at sigma = SIGMA_SIZE and are taken from it.
    """
    pts = [(SIGMA_SIZE, lc, N) for N in N_SIZES for lc in LCS]
    return [(50 + g, *p) for g, p in enumerate(pts)]


def check_grid():
    """Main-grid points repeated at GMAX_CHECK with the same seeds."""
    return [p for p in main_grid() if p[1] == SIGMA_SIZE]
