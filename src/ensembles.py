"""Disorder ensembles with per-realisation checkpointing."""
import json
from pathlib import Path
from . import config as C, gme, records


def run_id(tag, gmax, sigma, lc, N):
    return f"{tag}_g{gmax:g}_s{sigma:g}_lc{lc:g}_N{N}"


def grid_point(tag, N, gmax, sigma, lc, grid_index, outdir, n_seeds, references,
               recompute=False):
    """Ensemble of n_seeds realisations at one (sigma, lc, N) point.

    Realisations are appended to a checkpoint file as they finish, so an
    interrupted run resumes where it stopped. A stored record with fewer
    realisations than requested is extended rather than recomputed. The clean
    reference is built only when a realisation has to be computed and is shared
    through the `references` dictionary.
    """
    name = run_id(tag, gmax, sigma, lc, N)
    Path(outdir).mkdir(parents=True, exist_ok=True)
    path = Path(outdir, name + ".json")
    partial = Path(outdir, name + ".partial.jsonl")
    rows = []
    if path.exists() and not recompute:
        stored = records.load(path)
        if stored["n_seeds"] >= n_seeds:
            return stored
        rows = stored["rows"]
    elif partial.exists() and not recompute:
        rows = [json.loads(line) for line in partial.read_text().splitlines()]
    sid = C.STRUCTURES[tag]["structure_id"]
    if (tag, N, gmax) not in references:
        references[(tag, N, gmax)] = gme.Reference(tag, N, gmax)
    ref = references[(tag, N, gmax)]
    for r in range(len(rows), n_seeds):
        row = gme.realization(ref, sigma, lc, C.seed_for(sid, grid_index, r))
        rows.append(row)
        with partial.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
    cfg = C.STRUCTURES[tag]
    record = dict(run_id=name, structure=tag, structure_id=sid, d=cfg["d"],
                  r0=cfg["r"], gmode=ref.gmode, gmax=gmax, N=N, sigma=sigma,
                  lc=lc, grid_index=grid_index, n_seeds=n_seeds, f0=ref.f0,
                  inv_q_clean=ref.inv_q, rows=rows[:n_seeds])
    records.save(path, record)
    if partial.exists():
        partial.unlink()
    return record
