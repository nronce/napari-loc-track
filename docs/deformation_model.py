"""Figure: the growth deformation model and how localizations are corrected with it.

Run from anywhere: writes deformation_model.pdf/.svg/.png next to this script.
The numbers in the panels are those of the calibration and of a typical
recording (2026-09-25 1.3.2/009, elongation zone, 23 min); the drawings are
schematic, with the deformations exaggerated where noted.
"""
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7, "axes.linewidth": 0.6,
    "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "legend.fontsize": 6,
    "svg.fonttype": "none", "pdf.fonttype": 42,
})
INK = "#222222"
AXIS = "#c0392b"          # the root axis a
ACROSS = "#2e86c1"        # the direction m across it
GREY = "#b8b8b8"
WL = "#7f8c8d"            # white-light camera
FL = "#27ae60"            # fluorescence camera
WARM = "#e67e22"


def label(ax, letter, x=-0.02, y=1.02):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom",
            ha="right")


def arrow(ax, p, q, color=INK, lw=0.9, style="-|>", ms=7, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, color=color, lw=lw,
                                 shrinkA=0, shrinkB=0, **kw))


# --- (a) the two cameras on the sample -------------------------------------------------

def panel_cameras(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-6, 172)
    ax.set_ylim(172, -22)
    ax.axis("off")
    theta = math.radians(92)
    a = np.array([math.cos(theta), math.sin(theta)])
    m = np.array([-a[1], a[0]])
    c = np.array([83.0, 88.0])
    s = np.linspace(-84, 82, 50)
    for off in np.arange(-36, 37, 12):                       # cell files
        line = c + np.outer(s, a) + off * m
        ax.plot(line[:, 0], line[:, 1], color="#d7ccc8" if abs(off) < 36 else "#8d6e63",
                lw=0.5 if abs(off) < 36 else 1.0, zorder=0)
    for t in np.arange(-80, 81, 23):                          # cross walls
        seg = c + t * a + np.outer(np.array([-36, 36]), m)
        ax.plot(seg[:, 0], seg[:, 1], color="#d7ccc8", lw=0.5, zorder=0)
    ax.add_patch(Rectangle((0, 0), 166.5, 166.5, fill=False, ec=WL, lw=1.1))
    ax.text(0, -3, "white light (Basler): 166 µm, 81.28 nm/px", color=WL, va="bottom", fontsize=6)
    w, h = 368 * 0.16187, 675 * 0.16187
    rot = math.radians(0.80)
    R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
    corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]]) @ R.T + c
    ax.add_patch(Polygon(corners, closed=True, fill=True, fc=FL, alpha=0.13, ec=FL, lw=1.1))
    ax.text(corners[1, 0] + 21, corners[1, 1] + 2, "fluorescence\nROI (Kuro)\n161.87 nm/px",
            color=FL, fontsize=6, va="top")
    lo, hi = corners.min(axis=0) - 12.2, corners.max(axis=0) + 12.2
    ax.add_patch(Rectangle(lo, *(hi - lo), fill=False, ec=INK, lw=0.6, ls=(0, (3, 2))))
    ax.text(hi[0] + 2, hi[1] - 2, "growth\nmeasured\nhere", fontsize=5.5, va="bottom")
    arrow(ax, c - 50 * a, c + 50 * a, color=AXIS, lw=1.3)
    ax.text(*(c - 50 * a + np.array([4, 2])), "a", color=AXIS, fontsize=8, va="top",
            fontweight="bold")
    arrow(ax, c, c + 20 * m, color=ACROSS, lw=1.0)
    ax.text(*(c + 22 * m + np.array([-1, -2])), "m", color=ACROSS, fontsize=8, ha="right",
            fontweight="bold")
    ax.text(0.5, -0.03, "camera map C:  q = J b + T,   J = 0.502 R(0.80°)\n"
                        "b white-light px, q fluorescence px (Argo-SIM, 5-13 nm)\n"
                        "a: the root axis, from the direction of the walls",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8)
    ax.set_title("Two cameras, one sample plane", loc="left")


# --- (b) the seven terms ------------------------------------------------------------

TERMS = [
    ("g₀ a", "slides along", lambda s, n: (0.25 + 0 * s, 0 * s)),
    ("g₁ s a", "stretches along", lambda s, n: (0.35 * s, 0 * s)),
    ("g₂ s² a", "stretches faster\nfurther along", lambda s, n: (0.35 * s * s, 0 * s)),
    ("g₃ n a", "shears along", lambda s, n: (0.3 * n, 0 * s)),
    ("h₀ m", "slides across", lambda s, n: (0 * s, 0.25 + 0 * s)),
    ("h₁ s m", "shears across\n(with g₃: turns)", lambda s, n: (0 * s, 0.3 * s)),
    ("h₂ n m", "stretches across", lambda s, n: (0 * s, 0.2 * n)),
]


def mini_grid(ax, term, title, subtitle):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-1.4, 1.75)
    ax.set_ylim(-1.35, 1.35)
    lines = np.linspace(-1, 1, 5)
    t = np.linspace(-1, 1, 30)
    for v in lines:
        for s_, n_ in ((t, np.full_like(t, v)), (np.full_like(t, v), t)):
            ax.plot(s_, n_, color=GREY, lw=0.5)
            ds, dn = term(s_, n_)
            ax.plot(s_ + ds, n_ + dn, color=INK, lw=0.7)
    ax.text(0.17, 1.3, title, ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(0.17, -1.3, subtitle, ha="center", va="top", fontsize=5.5, color="#555555")


def panel_terms(fig, spec):
    sub = spec.subgridspec(2, 4, wspace=0.08, hspace=0.55)
    first = None
    for i, (title, subtitle, term) in enumerate(TERMS):
        ax = fig.add_subplot(sub[i // 4, i % 4])
        mini_grid(ax, term, title, subtitle)
        if i == 0:
            first = ax
    key = fig.add_subplot(sub[1, 3])
    key.axis("off")
    key.set_xlim(0, 1)
    key.set_ylim(0, 1)
    arrow(key, (0.05, 0.9), (0.35, 0.9), color=AXIS, lw=1.0, ms=6)
    key.text(0.38, 0.9, "a: along (s)", color=AXIS, fontsize=6, va="center")
    arrow(key, (0.05, 0.9), (0.05, 0.62), color=ACROSS, lw=1.0, ms=6)
    key.text(0.09, 0.66, "m: across (n)", color=ACROSS, fontsize=6, va="center")
    key.text(0.0, 0.5, "grey: before\nblack: moved by the\nterm (exaggerated)",
             fontsize=5.5, va="top", color="#555555")
    return first


def formula(fig, x, y):
    fig.text(x, y, "Φₖ(p) = p + (g₀ + g₁s + g₂s² + g₃n) a + (h₀ + h₁s + h₂n) m",
             fontsize=7.5, ha="center", va="top")
    fig.text(x, y - 0.018, "Snapshot k onto the last one's geometry: an affine map (6 numbers)\n"
             "plus the one quadratic term a growing root needs. s, n: position along\n"
             "and across the root axis, from the field centre.", fontsize=5.8, ha="center",
             va="top", color="#444444")


# --- (c) the estimation -------------------------------------------------------------

def panel_estimation(ax):
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    K = 12
    xs = np.linspace(0.3, 5.4, K + 1)
    y0 = 5.6
    for lag, color in ((1, "#555555"), (2, "#777777"), (4, "#999999"), (8, "#bbbbbb")):
        for k in range(0, K + 1 - lag, 1 if lag < 4 else 2):
            ax.add_patch(FancyArrowPatch((xs[k], y0), (xs[k + lag], y0),
                                         connectionstyle="arc3,rad=-0.4", arrowstyle="-",
                                         color=color, lw=0.5))
    for k in range(0, K, 3):
        ax.add_patch(FancyArrowPatch((xs[k], y0), (xs[K], y0), connectionstyle="arc3,rad=0.3",
                                     arrowstyle="-", color=WARM, lw=0.6, ls=(0, (2, 1.5))))
    ax.plot(xs, np.full_like(xs, y0), "o", color=INK, ms=3.0, zorder=3)
    ax.text(xs[0], y0 + 0.5, "first", ha="center", va="bottom", fontsize=5.5)
    ax.text(xs[-1], y0 + 0.5, "last", ha="center", va="bottom", fontsize=5.5)
    ax.text(2.85, 1.6, "snapshots every 30 s; pairs 1, 2, 4,\n8, 16 apart, and each against the\n"
            "last (orange): every map tied to\nthe last by many paths",
            ha="center", va="top", fontsize=5.8)
    # patches over long cells, with their directional weights
    ox, oy, cw, ch = 6.1, 1.6, 3.3, 7.4
    for i in range(4):
        ax.plot([ox + i * cw / 3] * 2, [oy, oy + ch], color="#8d6e63", lw=1.0)
    for yy in (oy + 2.0, oy + 5.6):
        ax.plot([ox + cw / 3, ox + 2 * cw / 3], [yy, yy], color="#8d6e63", lw=1.0)
    for cx, cy, wide, tall in ((ox + 0.55, oy + 1.8, 0.9, 0.3), (ox + 1.65, oy + 2.0, 0.75, 0.75),
                               (ox + 2.75, oy + 1.8, 0.9, 0.3), (ox + 0.55, oy + 5.4, 0.9, 0.3),
                               (ox + 1.65, oy + 5.6, 0.75, 0.75), (ox + 2.75, oy + 5.4, 0.9, 0.3)):
        ax.add_patch(Rectangle((cx - 0.5, cy - 0.7), 1.0, 1.4, fill=False, ec=GREY, lw=0.5))
        ax.add_patch(Ellipse((cx, cy), wide, tall, fc=ACROSS, alpha=0.35, ec=ACROSS, lw=0.5))
    arrow(ax, (ox + cw + 0.25, oy + 0.2), (ox + cw + 0.25, oy + 1.8), color=AXIS, lw=0.9, ms=5)
    ax.text(ox + cw + 0.4, oy + 1.0, "a", color=AXIS, fontsize=6, va="center")
    ax.text(ox + cw / 2, oy - 0.25, "patches of 31 µm, weighted\nby their walls: long walls fix\n"
            "only the shift across them", ha="center", va="top", fontsize=5.5)
    ax.text(10.1, 9.4,
            "1  rough maps: neighbours\n"
            "    chained\n"
            "2  every snapshot warped into\n"
            "    the last one's geometry\n"
            "3  every pair registered, patch\n"
            "    by patch:  rⱼₖ = eⱼ − eₖ\n"
            "    (e: what a map has wrong)\n"
            "4  one weighted, robust fit\n"
            "    over all pairs: every eₖ\n"
            "5  maps updated; repeated\n"
            "    until < 0.05 px (4 nm)",
            va="top", fontsize=5.8, linespacing=1.4)
    ax.set_title("Estimation, on the white-light snapshots", loc="left")


# --- (d) in time --------------------------------------------------------------------

def panel_time(ax):
    rng = np.random.default_rng(4)
    t_snap = np.arange(0, 23.01, 0.5)
    stretch = 9.0 * (1 - t_snap / 23.0) ** 1.05 + rng.normal(0, 0.05, t_snap.size)
    stretch[-1] = 0
    t = np.linspace(0, 23, 2000)
    ax.plot(t, np.interp(t, t_snap, stretch), color=AXIS, lw=0.8)
    ax.plot(t_snap, stretch, "o", color=AXIS, ms=2.0)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("stretch still to come (%)")
    ax.set_xlim(0, 23.5)
    ax.set_ylim(-0.5, 10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(0.6, 0.6, "each map's numbers\ninterpolated between\nsnapshots", color=AXIS, fontsize=5.8,
            va="bottom")
    inset = ax.inset_axes([0.46, 0.5, 0.52, 0.42])
    tt = np.linspace(6, 9, 1800)
    smooth = 0.4 * (tt - 6)
    jolt = 0.35 / (1 + np.exp(-(tt - 7.3) * 40)) - 0.35 / (1 + np.exp(-(tt - 7.45) * 40))
    record = smooth + jolt + rng.normal(0, 0.01, tt.size)
    ts = np.arange(6, 9.01, 0.5)
    snap = np.interp(ts, tt, smooth + jolt)
    inset.plot(tt, record, color=WL, lw=0.6, label="10 Hz drift record")
    inset.plot(ts, snap, "o-", color=INK, ms=2, lw=0.6, label="between snapshots")
    inset.fill_between(tt, np.interp(tt, ts, snap), record, color=WARM, alpha=0.4, lw=0,
                       label="fast motion f(t), added")
    inset.set_xticks([])
    inset.set_yticks([])
    low, high = record.min(), record.max()
    inset.set_ylim(low - 0.05 * (high - low), high + 0.9 * (high - low))   # room for the legend
    inset.set_title("a jolt between two snapshots", fontsize=5.5, pad=2)
    inset.legend(fontsize=4.8, loc="upper left", frameon=False, handlelength=1.2)
    ax.set_title("In time", loc="left")


# --- (e) one localization -----------------------------------------------------------

def box(ax, xy, text, color=INK, fc="white", w=1.9, h=1.0, fs=5.8):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.04,rounding_size=0.12", fc=fc, ec=color, lw=0.8))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=INK)


def panel_path(ax):
    ax.axis("off")
    ax.set_xlim(0, 14.4)
    ax.set_ylim(0, 3.2)
    y = 2.0
    chain = [
        (1.05, "localization q\nat frame time t", FL),
        (3.35, "b = C⁻¹(q)\nwhite-light px", WL),
        (5.65, "b − f(t)\nfast motion out", WARM),
        (7.95, "Φₜ(b − f(t))\ngrowth map at t", AXIS),
        (10.25, "q_final = C(·)\nfinal geometry", INK),
    ]
    for x, text, color in chain:
        box(ax, (x, y), text, color=color)
    for (x0, _t, _c), (x1, _t2, _c2) in zip(chain[:-1], chain[1:]):
        arrow(ax, (x0 + 1.0, y), (x1 - 1.0, y), lw=0.8, ms=6)
    box(ax, (13.1, 2.65), "linking, rendering,\nthe WL overlay", w=2.3, h=0.85, fc="#f4f6f7")
    box(ax, (13.1, 1.35), "q_local = Jₜ⁻¹(q_final − c) + c\n→ D, distances, immobility",
        w=2.5, h=0.95, fc="#fdf2e9", color=WARM, fs=5.3)
    arrow(ax, (11.2, 2.2), (11.95, 2.6), lw=0.8, ms=6)
    arrow(ax, (11.2, 1.8), (11.85, 1.4), lw=0.8, ms=6)
    ax.text(0.05, 0.55,
            "Every localization lands where its piece of tissue is at the end. For D, the local "
            "stretch still to come (Jₜ) is taken back out:\nlinked in the final geometry, a step "
            "between neighbouring frames would be stretched by it, and D inflated by (1 + ε)².",
            fontsize=5.8, va="center", color="#444444")
    ax.set_title("One localization, corrected", loc="left")


def main():
    fig = plt.figure(figsize=(7.2, 8.6))
    outer = fig.add_gridspec(3, 3, height_ratios=[1.05, 0.78, 0.36], hspace=0.34, wspace=0.34,
                             left=0.05, right=0.985, top=0.955, bottom=0.03)
    ax_a = fig.add_subplot(outer[0, 0])
    panel_cameras(ax_a)
    label(ax_a, "a", x=0.0)
    right = outer[0, 1:].subgridspec(2, 1, height_ratios=[1, 0.16], hspace=0.12)
    panel_terms(fig, right[0])
    box_b = right[1].get_position(fig)
    formula(fig, (box_b.x0 + box_b.x1) / 2, box_b.y1)
    pos = right[0].get_position(fig)
    fig.text(pos.x0 + 0.01, pos.y1 + 0.012, "The growth model: seven numbers per snapshot",
             fontsize=7.5, va="bottom")
    fig.text(pos.x0, pos.y1 + 0.012, "b", fontsize=10, fontweight="bold", va="bottom", ha="right")
    ax_c = fig.add_subplot(outer[1, 0:2])
    panel_estimation(ax_c)
    label(ax_c, "c", x=0.0)
    ax_d = fig.add_subplot(outer[1, 2])
    panel_time(ax_d)
    label(ax_d, "d", x=-0.18)
    ax_e = fig.add_subplot(outer[2, :])
    panel_path(ax_e)
    label(ax_e, "e", x=0.0)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(HERE / f"deformation_model.{ext}", dpi=300 if ext == "png" else None)
    print("written", HERE / "deformation_model.pdf")


if __name__ == "__main__":
    main()
