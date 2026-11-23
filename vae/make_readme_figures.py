"""Draw the README figures from the saved results of every video's CVAE-GAN notebooks.

Reads vae/<video>/results/*.csv (written by the three experiment notebooks) and writes into images/:

    tau_choice_<video>.png   validation RMSE against tau_1/tau_2 for all 18 weights, original and normalised frames
    test_comparison.png      test RMSE of the four ways of setting up the model, every video side by side

Usage (from the repo root or from vae/):
    python vae/make_readme_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "images")
BLUE, ORANGE, GREEN, GREY = "#1f5fbf", "#d9480f", "#2b8a3e", "#7a7a7a"
KEY = ["tau_1", "tau_2", "tau_3"]


def videos_with_results():
    return sorted(v for v in os.listdir(HERE)
                  if os.path.exists(os.path.join(HERE, v, "results", "summary_test.csv")))


def fmt(t):
    return "1/3" if abs(t - 1 / 3) < 1e-9 else f"{t:g}"


def summarise(df):
    s = df.groupby(KEY).agg(rmse=("rmse", "mean"), std=("rmse", "std"), active=("active", "mean"),
                            mean_frame=("mean_frame_rmse", "mean")).reset_index()
    s["ratio"] = s.tau_1 / s.tau_2
    return s


def tau_choice(video):
    res = os.path.join(HERE, video, "results")
    panels = [("Original frames", summarise(pd.read_csv(os.path.join(res, "diff_weights_validation.csv")))),
              ("Normalised frames", summarise(pd.read_csv(os.path.join(res, "normalised_and_diff_weights_validation.csv"))))]
    best = min(panels, key=lambda p: p[1].rmse.min())[1].rmse.min()
    top = max(p[1].rmse.max() + p[1]["std"].max() for p in panels)
    fig, axes = plt.subplots(1, 2, figsize=(20, 6.8), constrained_layout=True, sharey=True)
    for ax, (title, s) in zip(axes, panels):
        for t3, color, ls, mk, off in zip([1 / 3, 0.1, 0.01], [BLUE, ORANGE, GREEN], ["-", "--", "-."], ["o", "s", "^"],
                                          [1.5, 1.15, 1.0]):
            d = s[np.isclose(s.tau_3, t3)].sort_values("ratio")
            ax.errorbar(d.ratio * off, d.rmse, yerr=d["std"], ls=ls, marker=mk, color=color, capsize=3,
                        label=f"$\\tau_3$ = {fmt(t3)}")
            if t3 == 1 / 3:
                for x, y, e, a in zip(d.ratio * off, d.rmse, d["std"], d.active):
                    ax.text(x, y - e - 0.0012, f"{a:.0f}", color=BLUE, ha="center", va="top", fontsize=9)
        ax.axhline(s.mean_frame.mean(), color=GREY, ls=":", lw=1.2)
        ax.text(1.2e-5, s.mean_frame.mean() + 0.0004, f"mean frame (no latent), {s.mean_frame.mean():.4f}",
                color=GREY, fontsize=9)
        b = s.loc[s.rmse.idxmin()]
        if np.isclose(b.rmse, best):           # the circle marks the lowest over both panels, at its tau_3's offset
            off = 1.5 if np.isclose(b.tau_3, 1 / 3) else 1.15 if np.isclose(b.tau_3, 0.1) else 1.0
            ax.plot(b.ratio * off, b.rmse, "o", ms=22, mfc="none", mec="black", mew=2)
        ax.set_xscale("log"); ax.set_xlabel("$\\tau_1/\\tau_2 = \\sigma_x^2$   (log scale)")
        ax.set_title(title, fontsize=11)
        sec = ax.secondary_xaxis("top", functions=(np.sqrt, np.square))
        sec.set_xlabel("$\\sigma_x$ = assumed noise per tile")
        ax.legend(fontsize=9, frameon=False, loc="upper center", ncol=3)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("validation RMSE (mean ± std of 25 runs)")
    axes[0].set_ylim(0, top * 1.08)
    fig.suptitle(f"{video}: choosing $\\tau$ on the validation folds. Numbers under the blue line = active latent "
                 "slots of 10 (mean of 25 runs); circle = lowest mean validation RMSE", fontsize=12)
    path = os.path.join(OUT, f"tau_choice_{video}.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def test_comparison(videos):
    rows = []
    for v in videos:
        res = os.path.join(HERE, v, "results")
        eq = pd.read_csv(os.path.join(res, "normalised_tiles_test.csv"))
        sets = [("original,\nequal weights", eq[~eq.normalised]),
                ("original,\nchosen weights", pd.read_csv(os.path.join(res, "diff_weights_test.csv"))),
                ("normalised,\nequal weights", eq[eq.normalised]),
                ("normalised,\nchosen weights", pd.read_csv(os.path.join(res, "normalised_and_diff_weights_test.csv")))]
        for name, d in sets:
            rows.append({"video": v, "setup": name, "rmse": d.rmse.mean(), "std": d.rmse.std(), "active": d.active.mean()})
        rows.append({"video": v, "setup": "mean frame", "rmse": eq.mean_frame_rmse.mean()})
    df = pd.DataFrame(rows)
    setups = [s for s in df.setup.unique() if s != "mean frame"]
    colors = [GREY, "#b0b0b0", BLUE, "#7fa6e6"]
    fig, ax = plt.subplots(figsize=(3.0 * len(videos) + 2, 5.2), constrained_layout=True)
    w = 0.2
    for i, (s, c) in enumerate(zip(setups, colors)):
        d = df[df.setup == s].set_index("video").loc[videos]
        x = np.arange(len(videos)) + (i - 1.5) * w
        ax.bar(x, d.rmse, w * 0.92, yerr=d["std"], capsize=3, color=c, label=s.replace("\n", " "))
        for xi, r, e, a in zip(x, d.rmse, d["std"], d.active):
            ax.text(xi, r + e + 0.0006, f"{r:.4f}\n{a:.1f}", ha="center", fontsize=7)
    mf = df[df.setup == "mean frame"].set_index("video").loc[videos].rmse
    for j, m in enumerate(mf):
        ax.plot([j - 2 * w, j + 2 * w], [m, m], color=ORANGE, ls=":", lw=2, label="mean frame (no latent)" if j == 0 else None)
    ax.set_xticks(np.arange(len(videos))); ax.set_xticklabels(videos)
    ax.set_ylabel("test RMSE (mean ± std of 25 runs)")
    ax.set_title("Test RMSE on the held-out viewers (5 folds x 5 repeats per bar); numbers: RMSE and active slots of 10",
                 fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper left")
    ax.set_ylim(0, (df.rmse + df["std"].fillna(0)).max() * 1.25)
    ax.grid(alpha=0.25, axis="y"); ax.spines[["top", "right"]].set_visible(False)
    path = os.path.join(OUT, "test_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    vids = videos_with_results()
    assert vids, "no vae/<video>/results/summary_test.csv yet: run vae/make_video_vaes.py --run first"
    os.makedirs(OUT, exist_ok=True)
    for v in vids:
        print("wrote", os.path.relpath(tau_choice(v), os.path.join(HERE, "..")))
    print("wrote", os.path.relpath(test_comparison(vids), os.path.join(HERE, "..")))


if __name__ == "__main__":
    main()
