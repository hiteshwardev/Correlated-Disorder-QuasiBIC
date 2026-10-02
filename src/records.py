"""JSON result records and on-disk caching of expensive stages."""
import json
from pathlib import Path
import numpy as np


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_plain(obj), indent=1))
    return path


def load(path):
    return json.loads(Path(path).read_text())


def cached(path, compute, recompute=False):
    """Return the record at `path`, computing and saving it first if it is
    missing or recompute is True."""
    path = Path(path)
    if path.exists() and not recompute:
        return load(path)
    result = compute()
    save(path, result)
    return result
