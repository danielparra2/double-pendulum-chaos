"""Double pendulum: from regular motion to chaos.

Two point masses on massless rods. Units: m1 = m2 = L1 = L2 = g = 1, angles
measured from the downward vertical. Everything is integrated with a
hand-written, vectorised fourth-order Runge-Kutta scheme.

Run:
    python double_pendulum.py     # prints the checks, writes figures/*
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).parent / "figures"
M1 = M2 = L1 = L2 = G = 1.0

CHAOTIC = (2.2, 0.0, 2.2, 0.0)    # both arms raised to 126 degrees, released from rest
REGULAR = (0.4, 0.0, 0.4, 0.0)    # both arms at 23 degrees, released from rest


# --------------------------------------------------------------------------
# Physics
# --------------------------------------------------------------------------
def rhs(y: np.ndarray) -> np.ndarray:
    """Equations of motion. y = (theta1, omega1, theta2, omega2); works on arrays."""
    th1, w1, th2, w2 = y
    d = th1 - th2
    s, c = np.sin(d), np.cos(d)
    den = 2 * M1 + M2 - M2 * np.cos(2 * d)
    a1 = (-G * (2 * M1 + M2) * np.sin(th1) - M2 * G * np.sin(th1 - 2 * th2)
          - 2 * s * M2 * (w2**2 * L2 + w1**2 * L1 * c)) / (L1 * den)
    a2 = (2 * s * (w1**2 * L1 * (M1 + M2) + G * (M1 + M2) * np.cos(th1)
                   + w2**2 * L2 * M2 * c)) / (L2 * den)
    return np.array([w1, a1, w2, a2])


def rk4_step(y: np.ndarray, dt: float) -> np.ndarray:
    k1 = rhs(y)
    k2 = rhs(y + 0.5 * dt * k1)
    k3 = rhs(y + 0.5 * dt * k2)
    k4 = rhs(y + dt * k3)
    return y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def euler_step(y: np.ndarray, dt: float) -> np.ndarray:
    return y + dt * rhs(y)


def integrate(y0, dt: float, n_steps: int, every: int = 1, step=rk4_step) -> np.ndarray:
    """Return the state every `every` steps; shape (n_steps // every + 1, 4, ...)."""
    y = np.array(y0, dtype=float)
    out = np.empty((n_steps // every + 1,) + y.shape)
    out[0] = y
    for i in range(1, n_steps + 1):
        y = step(y, dt)
        if i % every == 0:
            out[i // every] = y
    return out


def energy(y) -> np.ndarray:
    th1, w1, th2, w2 = y
    kinetic = 0.5 * M1 * (L1 * w1) ** 2 + 0.5 * M2 * (
        (L1 * w1) ** 2 + (L2 * w2) ** 2 + 2 * L1 * L2 * w1 * w2 * np.cos(th1 - th2))
    potential = -(M1 + M2) * G * L1 * np.cos(th1) - M2 * G * L2 * np.cos(th2)
    return kinetic + potential


def positions(th1, th2):
    x1, y1 = L1 * np.sin(th1), -L1 * np.cos(th1)
    return x1, y1, x1 + L2 * np.sin(th2), y1 - L2 * np.cos(th2)


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def fan(y0, spread: float, n: int, t_end: float, frames: int, dt: float = 1e-3):
    """n pendulums whose initial theta1 differ by at most `spread`."""
    y = np.tile(np.array(y0, dtype=float)[:, None], (1, n))
    y[0] += np.linspace(0.0, spread, n)
    every = int(round(t_end / frames / dt))
    traj = integrate(y, dt, frames * every, every)           # (frames + 1, 4, n)
    return np.arange(frames + 1) * every * dt, traj


def divergence(y0, delta: float = 1e-9, t_end: float = 40.0, dt: float = 1e-3):
    """Phase-space distance between two trajectories started `delta` apart."""
    y = np.tile(np.array(y0, dtype=float)[:, None], (1, 2))
    y[0, 1] += delta
    traj = integrate(y, dt, int(t_end / dt), every=20)
    return np.arange(len(traj)) * 20 * dt, np.linalg.norm(traj[:, :, 1] - traj[:, :, 0], axis=1)


def lyapunov_fit(t, sep, lo: float = 1e-8, hi: float = 1e-2):
    """Slope of ln(separation) while it grows from `lo` to `hi` (finite-time estimate)."""
    i0 = np.argmax(sep > lo)
    i1 = np.argmax(sep > hi) if np.any(sep > hi) else len(sep) - 1
    slope, intercept = np.polyfit(t[i0:i1], np.log(sep[i0:i1]), 1)
    return slope, intercept, (t[i0], t[i1])


def mode_period(ratio: float, amp: float = 1e-3, t_end: float = 60.0, dt: float = 5e-3):
    """Period of a small oscillation started in the normal mode theta2 = ratio * theta1."""
    traj = integrate((amp, 0.0, ratio * amp, 0.0), dt, int(t_end / dt))
    th = traj[:, 0]
    i = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
    crossings = (i - th[i] / (th[i + 1] - th[i])) * dt
    return np.mean(np.diff(crossings))


def flip_times(n: int = 360, t_max: float = 60.0, dt: float = 0.02):
    """Time until either arm first flips over, for a grid of release angles (from rest).

    Only theta1 >= 0 is integrated; the other half follows from the symmetry
    (theta1, theta2) -> (-theta1, -theta2).
    """
    th = np.linspace(-np.pi, np.pi, n)
    TH1, TH2 = np.meshgrid(th[n // 2:], th)
    y = np.zeros((4, TH1.size))
    y[0], y[2] = TH1.ravel(), TH2.ravel()
    t_flip = np.full(TH1.size, np.nan)
    active = np.arange(TH1.size)
    for i in range(1, int(t_max / dt) + 1):
        y = rk4_step(y, dt)
        flipped = (np.abs(y[0]) > np.pi) | (np.abs(y[2]) > np.pi)
        if flipped.any():
            t_flip[active[flipped]] = i * dt
            y, active = y[:, ~flipped], active[~flipped]
    right = t_flip.reshape(TH1.shape)
    return th, np.hstack([right[::-1, ::-1], right])


def poincare(energies, t_end: float = 1500.0, dt: float = 0.01, n_ic: int = 14):
    """Poincare section theta2 = 0 (crossing upwards): points (theta1, p1) per energy.

    p1 = 2 omega1 + omega2 cos(theta1 - theta2) is the momentum conjugate to theta1.
    """
    cols, label = [], []
    for k, E in enumerate(energies):
        a = 0.97 * (np.arccos(-(E + 1) / 2) if E < 1 else np.pi)
        for th1 in np.linspace(-a, a, n_ic):                       # omega1 = 0
            cols.append((th1, 0.0, 0.0, np.sqrt(2 * (E + 1 + 2 * np.cos(th1)))))
            label.append(k)
        for w1 in np.linspace(-0.95 * np.sqrt(2 * (E + 3)), 0.95 * np.sqrt(E + 3), n_ic):
            cols.append((0.0, w1, 0.0, -w1 + np.sqrt(2 * (E + 3) - w1**2)))   # theta1 = 0
            label.append(k)
    y = np.array(cols).T
    label = np.array(label)
    e0 = energy(y)
    pts = [[] for _ in energies]

    def p1_of(state):
        return 2 * state[1] + state[3] * np.cos(state[0] - state[2])

    for _ in range(int(t_end / dt)):
        y_new = rk4_step(y, dt)
        s0, s1 = np.sin(y[2]), np.sin(y_new[2])
        hit = np.where((s0 < 0) & (s1 >= 0) & (np.cos(y_new[2]) > 0))[0]
        if hit.size:
            f = -s0[hit] / (s1[hit] - s0[hit])
            th1 = y[0, hit] + f * (y_new[0, hit] - y[0, hit])
            p1 = p1_of(y[:, hit]) + f * (p1_of(y_new[:, hit]) - p1_of(y[:, hit]))
            for k in range(len(energies)):
                m = label[hit] == k
                pts[k].append(np.column_stack([wrap(th1[m]), p1[m], hit[m]]))
        y = y_new
    drift = np.max(np.abs(energy(y) - e0))
    return [np.vstack(p) for p in pts], drift


def run_checks() -> dict:
    print("Checks (m1 = m2 = L1 = L2 = g = 1)")
    # 1. Small oscillations: exact normal-mode frequencies w^2 = 2 -+ sqrt(2)
    for name, ratio, w2 in (("in-phase", np.sqrt(2), 2 - np.sqrt(2)),
                            ("out-of-phase", -np.sqrt(2), 2 + np.sqrt(2))):
        exact = 2 * np.pi / np.sqrt(w2)
        got = mode_period(ratio)
        print(f"  normal mode {name:12s}  T = {got:.6f}   exact = {exact:.6f}"
              f"   rel. error = {abs(got - exact) / exact:.1e}")

    # 2. Energy conservation on the chaotic orbit: RK4 vs Euler, same step
    dt, t_end = 1e-3, 40.0
    t = np.arange(0, int(t_end / dt) + 1, 20) * dt
    e_rk4 = energy(integrate(CHAOTIC, dt, int(t_end / dt), 20).T)
    e_eul = energy(integrate(CHAOTIC, dt, int(t_end / dt), 20, step=euler_step).T)
    print(f"  energy drift over t = {t_end:.0f} (dt = {dt}):  RK4 = "
          f"{np.max(np.abs(e_rk4 - e_rk4[0])):.1e}   Euler = {np.max(np.abs(e_eul - e_eul[0])):.1e}")

    # 3. Sensitivity to initial conditions
    out = {"t": t, "e_rk4": e_rk4, "e_eul": e_eul}
    for name, y0 in (("chaotic", CHAOTIC), ("regular", REGULAR)):
        out[name] = divergence(y0)
    slope, intercept, window = lyapunov_fit(*out["chaotic"])
    out["fit"] = (slope, intercept, window)
    print(f"  chaotic orbit: separation grows as exp({slope:.2f} t)  "
          f"(finite-time Lyapunov estimate, fitted on t = {window[0]:.1f}..{window[1]:.1f})")
    print(f"  regular orbit: separation after t = 40 is "
          f"{out['regular'][1][-1] / out['regular'][1][0]:.0f} x the initial one (no exponential growth)")

    # 4. How long is the computed chaotic path itself trustworthy?
    a = integrate(CHAOTIC, 1e-3, 60000, 100)
    b = integrate(CHAOTIC, 5e-4, 120000, 200)
    err = np.linalg.norm(a - b, axis=1)
    out["horizon"] = np.argmax(err > 1e-3) * 0.1 if np.any(err > 1e-3) else np.inf
    print(f"  numerical horizon: runs with dt and dt/2 agree to 1e-3 until t = {out['horizon']:.1f}")
    return out


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
BG, TEXT, DIM = "#0a0a12", "#e6e6f0", "#3a3a5c"
CYAN, PURPLE, MAGENTA, YELLOW = "#00f5d4", "#9b5de5", "#f15bb5", "#fee440"
NEON = LinearSegmentedColormap.from_list("neon", [CYAN, PURPLE, MAGENTA, YELLOW])
FLIP = LinearSegmentedColormap.from_list(
    "flip", [YELLOW, MAGENTA, PURPLE, "#3a0ca3", "#10103a"])
FLIP.set_bad(BG)

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": TEXT, "axes.labelcolor": TEXT, "axes.edgecolor": DIM,
    "xtick.color": TEXT, "ytick.color": TEXT, "font.size": 12,
    "mathtext.fontset": "cm",
})
PI_TICKS = ([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
            [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def animate_fan(path: Path, y0, title: str, color: str, spread: float = 1e-4, n: int = 16,
                t_end: float = 30.0, frames: int = 240, fps: int = 24, trail: int = 22):
    t, traj = fan(y0, spread, n, t_end, frames)
    x1, y1, x2, y2 = positions(traj[:, 0], traj[:, 2])              # (frames + 1, n)
    gap = np.max(np.hypot(x2 - x2[:, :1], y2 - y2[:, :1]), axis=1)  # tip spread
    colors = NEON(np.linspace(0, 1, n))

    fig, (ax, ax_s) = plt.subplots(1, 2, figsize=(10.5, 5.0), dpi=80,
                                   gridspec_kw=dict(width_ratios=[1, 1.15]))
    fig.subplots_adjust(left=0.02, right=0.97, top=0.86, bottom=0.15, wspace=0.2)
    fig.suptitle(title, color=color, fontsize=16)

    ax.set_xlim(-2.25, 2.25)
    ax.set_ylim(-2.25, 2.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ring = np.linspace(0, 2 * np.pi, 200)
    ax.plot(2 * np.sin(ring), 2 * np.cos(ring), color=DIM, lw=0.8, ls=(0, (2, 4)))
    trails = [LineCollection([], linewidths=2.2, capstyle="round") for _ in range(n)]
    for tr in trails:
        ax.add_collection(tr)
    rods = [ax.plot([], [], color=c, lw=1.6, alpha=0.9, solid_capstyle="round")[0]
            for c in colors]
    bobs = [ax.plot([], [], "o", color=c, ms=7, mec="white", mew=0.5)[0] for c in colors]
    ax.plot([0], [0], "o", color=TEXT, ms=5)
    clock = ax.text(0.02, 0.02, "", transform=ax.transAxes, color=TEXT, fontsize=12)
    fade = np.linspace(0.9, 0.0, trail - 1) ** 1.5

    ax_s.set_yscale("log")
    ax_s.set_xlim(0, t_end)
    ax_s.set_ylim(spread / 20, 10)
    ax_s.set_xlabel(r"time  $t\,\sqrt{g/L}$")
    ax_s.set_ylabel("largest gap between the tips  (units of $L$)")
    ax_s.grid(color=DIM, lw=0.4, alpha=0.5)
    ax_s.axhline(0.1, color=YELLOW, lw=1, ls="--", alpha=0.8)
    ax_s.text(0.3, 0.13, "visible to the eye", color=YELLOW, fontsize=10, va="bottom")
    ax_s.set_title(f"{n} pendulums, released {spread:g} rad apart", fontsize=12, color=TEXT)
    curve, = ax_s.plot([], [], color=color, lw=2)
    ax_s.plot(t[1:], gap[1:], color=color, lw=1, alpha=0.15)
    dot, = ax_s.plot([], [], "o", color=color, ms=8, mec="white", mew=0.8)

    def update(i):
        lo = max(0, i - trail + 1)
        for k in range(n):
            pts = np.column_stack([x2[lo:i + 1, k], y2[lo:i + 1, k]])[::-1]
            if len(pts) > 1:
                rgba = np.tile(colors[k], (len(pts) - 1, 1))
                rgba[:, 3] = fade[:len(pts) - 1]
                trails[k].set_segments(np.stack([pts[:-1], pts[1:]], axis=1))
                trails[k].set_color(rgba)
            rods[k].set_data([0, x1[i, k], x2[i, k]], [0, y1[i, k], y2[i, k]])
            bobs[k].set_data([x2[i, k]], [y2[i, k]])
        j = max(i, 1)
        curve.set_data(t[1:j + 1], gap[1:j + 1])
        dot.set_data([t[j]], [gap[j]])
        clock.set_text(f"t = {t[i]:4.1f}")
        return ()

    order = list(range(frames + 1)) + [frames] * fps          # hold the last frame 1 s
    FuncAnimation(fig, update, frames=order, interval=1000 / fps).save(
        path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return t[np.argmax(gap > 0.1)] if np.any(gap > 0.1) else np.inf


def flip_map_png(path: Path, th, t_flip):
    fig, ax = plt.subplots(figsize=(8.6, 7.4), dpi=160)
    im = ax.imshow(np.log10(t_flip), extent=[-np.pi, np.pi, -np.pi, np.pi], origin="lower",
                   cmap=FLIP, interpolation="nearest", vmin=np.log10(0.8),
                   vmax=np.log10(np.nanmax(t_flip)))
    TH1, TH2 = np.meshgrid(th, th)
    ax.contour(TH1, TH2, 2 * np.cos(TH1) + np.cos(TH2), levels=[1.0], colors=CYAN,
               linewidths=1.3, linestyles="--")
    ax.text(0, 0, "not enough energy\nto flip", color=CYAN, ha="center", va="center",
            fontsize=11)
    ax.set_xticks(*PI_TICKS)
    ax.set_yticks(*PI_TICKS)
    ax.set_xlabel(r"release angle of the upper arm  $\theta_1$")
    ax.set_ylabel(r"release angle of the lower arm  $\theta_2$")
    ax.set_title("How long until the double pendulum flips over?", pad=12, fontsize=15)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    ticks = [1, 3, 10, 30]
    cb.set_ticks(np.log10(ticks), labels=[str(v) for v in ticks])
    cb.set_label(r"time to first flip  $t\,\sqrt{g/L}$   (dark = no flip yet)")
    cb.outline.set_edgecolor(DIM)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def poincare_png(path: Path, energies, sections):
    fig, axes = plt.subplots(1, len(energies), figsize=(4.3 * len(energies), 4.9), dpi=160)
    for ax, E, pts in zip(axes, energies, sections):
        ids = pts[:, 2]
        shade = (ids - ids.min()) / max(np.ptp(ids), 1)
        ax.scatter(pts[:, 0], pts[:, 1], s=0.5, c=NEON(shade), linewidths=0, rasterized=True)
        ax.set_title(f"$E = {E:g}$", fontsize=14)
        ax.set_xlabel(r"$\theta_1$")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_xticks([-np.pi, 0, np.pi], [r"$-\pi$", "0", r"$\pi$"])
        ax.grid(color=DIM, lw=0.3, alpha=0.4)
    axes[0].set_ylabel(r"$p_1$")
    fig.suptitle(r"Poincaré sections  ($\theta_2 = 0$, crossing upwards):"
                 "  order dissolves into chaos as the energy grows", fontsize=14)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def diagnostics_png(path: Path, d: dict):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.6), dpi=160)
    slope, intercept, window = d["fit"]
    a.semilogy(*d["chaotic"], color=MAGENTA, lw=1.8, label="chaotic orbit")
    a.semilogy(*d["regular"], color=CYAN, lw=1.8, label="regular orbit")
    tt = np.array(window)
    a.semilogy(tt, np.exp(intercept + slope * tt) * 6, color=YELLOW, lw=1.4, ls="--",
               label=rf"$\propto e^{{{slope:.2f}\,t}}$")
    a.set_xlabel(r"time  $t\,\sqrt{g/L}$")
    a.set_ylabel("distance between two orbits in phase space")
    a.set_title("Two orbits started $10^{-9}$ rad apart", fontsize=13)
    b.semilogy(d["t"][1:], np.abs(d["e_eul"] - d["e_eul"][0])[1:], color=YELLOW, lw=1.6,
               label="Euler")
    b.semilogy(d["t"][1:], np.abs(d["e_rk4"] - d["e_rk4"][0])[1:] + 1e-17, color=CYAN,
               lw=1.6, label="RK4")
    b.set_xlabel(r"time  $t\,\sqrt{g/L}$")
    b.set_ylabel(r"energy error  $|E(t) - E(0)|$")
    b.set_title(r"Same orbit, same step ($\Delta t = 10^{-3}$)", fontsize=13)
    for ax in (a, b):
        ax.grid(color=DIM, lw=0.4, alpha=0.5)
        ax.legend(facecolor=BG, edgecolor=DIM, labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


POINCARE_ENERGIES = (-2.6, -1.8, -1.0, 0.5)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    d = run_checks()
    diagnostics_png(FIG_DIR / "diagnostics.png", d)

    t_split = animate_fan(FIG_DIR / "chaos.gif", CHAOTIC,
                          "High energy: chaos", MAGENTA)
    print(f"  chaos.gif: the 16 tips are more than 0.1 L apart from t = {t_split:.1f}")
    animate_fan(FIG_DIR / "regular.gif", REGULAR, "Low energy: regular motion", CYAN)

    th, t_flip = flip_times()
    forbidden = 2 * np.cos(th)[None, :] + np.cos(th)[:, None] > 1
    print(f"  flip map: {np.count_nonzero(~np.isnan(t_flip[forbidden]))} flips inside the "
          f"energetically forbidden region (must be 0), "
          f"{100 * np.mean(np.isnan(t_flip[~forbidden])):.1f} % of the allowed grid has not flipped by t = 60")
    flip_map_png(FIG_DIR / "flip_map.png", th, t_flip)

    sections, drift = poincare(POINCARE_ENERGIES)
    print(f"  Poincare sections: max energy drift over t = 1500 is {drift:.1e}")
    poincare_png(FIG_DIR / "poincare.png", POINCARE_ENERGIES, sections)
    print("wrote", ", ".join(sorted(p.name for p in FIG_DIR.iterdir())))


if __name__ == "__main__":
    main()
