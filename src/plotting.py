"""Figure style, layout and quality checks shared by all figures.

Every figure has a single panel. Legends sit in a band above the axes, axis
labels and tick labels outside the axes, and captions are written in the text,
not in the figure. `finish` refuses to save a figure in which any two text
elements overlap or any text element enters the axes rectangle, and checks
that nothing touches the border of the saved image and that the image is no
wider than MAX_WIDTH_CM, so that all figures print at a common scale.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from PIL import Image

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#5A5A5A"
MARKERS = ["o", "s", "^", "D", "v", "P"]
CM = 1 / 2.54
DPI = 600
WIDTH_CM = 18.0
MAX_WIDTH_CM = 20.0


def apply_style():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": DPI,
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 12.5, "axes.labelsize": 14, "legend.fontsize": 12.5,
        "xtick.labelsize": 12.5, "ytick.labelsize": 12.5,
        "axes.linewidth": 1.2, "lines.linewidth": 2.0, "lines.markersize": 8.5,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.pad": 6, "ytick.major.pad": 6, "axes.labelpad": 9,
        "legend.frameon": False, "errorbar.capsize": 4,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def new_figure(height_cm=11.0, legend_rows=1, width_cm=WIDTH_CM,
               left_cm=2.6, right_cm=0.5, bottom_cm=2.0):
    """Single-panel figure with a band of legend_rows rows reserved above the axes."""
    fig = plt.figure(figsize=(width_cm * CM, height_cm * CM))
    top_cm = 0.3 + 0.75 * legend_rows if legend_rows else 0.4
    ax = fig.add_axes([left_cm / width_cm, bottom_cm / height_cm,
                       1 - (left_cm + right_cm) / width_cm,
                       1 - (bottom_cm + top_cm) / height_cm])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def legend_above(fig, ax, handles=None, labels=None, ncol=3):
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    return fig.legend(handles, labels, loc="upper center", ncol=ncol,
                      bbox_to_anchor=(0.5 + (ax.get_position().x0 - 1 + ax.get_position().x1) / 2, 1.0),
                      frameon=False, handlelength=2.0, columnspacing=2.0,
                      handletextpad=0.6, borderaxespad=0.2)


def _tick_text(v):
    if 0.1 <= abs(v) < 100:
        return f"{v:g}"
    mant, expo = f"{v:.0e}".split("e")
    expo = int(expo)
    return rf"$10^{{{expo}}}$" if mant == "1" else rf"${mant}\times10^{{{expo}}}$"


def log_ticks(axis, values):
    """Label a logarithmic axis at explicit values, e.g. (0.5, 1, 2, 4) or
    (3e-7, 1e-6, 3e-6, 1e-5), when it spans too few decades for the default."""
    axis.set_major_locator(FixedLocator(list(values)))
    axis.set_major_formatter(FuncFormatter(lambda v, _: _tick_text(v)))


def _labelled_ticks(axis, lo, hi, renderer, horizontal):
    count = 0
    for t in axis.get_ticklabels(which="major"):
        if not t.get_visible() or not t.get_text():
            continue
        bb = t.get_window_extent(renderer)
        c = (bb.x0 + bb.x1) / 2 if horizontal else (bb.y0 + bb.y1) / 2
        count += lo - 2 <= c <= hi + 2
    return count


def _text_boxes(fig, ax, renderer):
    hidden = set()
    axbb = ax.get_window_extent(renderer)
    for axis, lo, hi, horiz in ((ax.xaxis, axbb.x0, axbb.x1, True),
                                (ax.yaxis, axbb.y0, axbb.y1, False)):
        for t in axis.get_ticklabels(which="both"):
            bb = t.get_window_extent(renderer)
            c = (bb.x0 + bb.x1) / 2 if horiz else (bb.y0 + bb.y1) / 2
            if not t.get_text() or c < lo - 2 or c > hi + 2:
                hidden.add(id(t))
    boxes, seen = [], set()
    for t in fig.findobj(match=lambda o: hasattr(o, "get_text") and hasattr(o, "get_window_extent")):
        if id(t) in hidden or not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer)
        key = tuple(round(v, 1) for v in (bb.x0, bb.y0, bb.x1, bb.y1))
        if bb.width <= 0 or key in seen:
            continue
        seen.add(key)
        boxes.append((t.get_text(), bb))
    return boxes, axbb


def layout_problems(fig, ax, tol=1.0):
    """Pairs of overlapping text elements and text elements inside the axes."""
    fig.canvas.draw()
    boxes, axbb = _text_boxes(fig, ax, fig.canvas.get_renderer())
    def overlap(a, b):
        return (min(a.x1, b.x1) - max(a.x0, b.x0) > tol and
                min(a.y1, b.y1) - max(a.y0, b.y0) > tol)
    pairs = [(boxes[i][0], boxes[j][0]) for i in range(len(boxes))
             for j in range(i + 1, len(boxes)) if overlap(boxes[i][1], boxes[j][1])]
    inside = [t for t, bb in boxes if overlap(bb, axbb)]
    return pairs, inside


def border_clear(png, border=4, tol=6):
    img = np.asarray(Image.open(png).convert("L"), dtype=int)
    bg = img[0, 0]
    edge = np.concatenate([img[:border].ravel(), img[-border:].ravel(),
                           img[:, :border].ravel(), img[:, -border:].ravel()])
    return bool(np.all(np.abs(edge - bg) <= tol))


def finish(fig, ax, name, outdir="figures"):
    """Hide minor tick labels, check the layout, save PDF (without a time stamp,
    so that repeated runs give identical files) and 600 dpi PNG, and return a
    small quality record."""
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_minor_formatter(NullFormatter())
    pairs, inside = layout_problems(fig, ax)
    if pairs or inside:
        raise RuntimeError(f"{name}: overlapping text {pairs}, text inside axes {inside}")
    renderer = fig.canvas.get_renderer()
    bb = ax.get_window_extent(renderer)
    for axis, lo, hi, horiz in ((ax.xaxis, bb.x0, bb.x1, True), (ax.yaxis, bb.y0, bb.y1, False)):
        labelled = _labelled_ticks(axis, lo, hi, renderer, horiz)
        if axis.get_visible() and axis.get_ticklocs().size and labelled < 2:
            raise RuntimeError(f"{name}: fewer than two labelled ticks on an axis")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    for ext, meta in (("pdf", {"CreationDate": None}), ("png", None)):
        fig.savefig(out / f"{name}.{ext}", dpi=DPI, bbox_inches="tight", pad_inches=0.15,
                    metadata=meta)
    png = out / f"{name}.png"
    with Image.open(png) as im:
        w_px, h_px = im.size
    quality = dict(figure=name, width_cm=round(w_px / DPI * 2.54, 2),
                   height_cm=round(h_px / DPI * 2.54, 2), dpi=DPI,
                   border_clear=border_clear(png))
    if not quality["border_clear"]:
        raise RuntimeError(f"{name}: content touches the image border")
    if quality["width_cm"] > MAX_WIDTH_CM:
        raise RuntimeError(f"{name}: {quality['width_cm']} cm wide; a legend is too long for one row")
    return quality
