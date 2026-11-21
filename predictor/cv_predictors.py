"""Held-out RMSE of every video's FoV predictor, by 5-fold cross-validation over viewers.

Each notebook trains its predictor on all 50 viewers, so the RMSE it prints is a training-data fit. This script
scores the SAME network and the SAME training recipe (Cfg in predictor/coaster1/fov_predictor_coaster1.ipynb) on viewers
the predictor never saw: viewer i goes to fold i mod 5 (the fold rule of the VAE notebooks), the predictor is trained
on the other 40, and two RMSEs are taken on the held-out 10, at the notebook's 7 target times (k = 6..12):

  single : predictor on each held-out viewer's own last 5 maps  vs  that viewer's actual map
  crowd  : predictor on the held-out 10's averaged maps (the crowd frames the VAE uses)  vs  the actual crowd frame

It adds cv_* columns (mean and std over the 5 folds) to predictor/summary_rmse.csv.

Usage:  python predictor/cv_predictors.py [--videos pacman sport]
"""
import argparse, glob, os, re, sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
TILE_DIR = os.path.join(HERE, "..", "dataset", "360dataset", "sensory", "tile")
# the notebook's Cfg (keep in step with predictor/coaster1/fov_predictor_coaster1.ipynb)
H, W, FPS, EVERY_S, N, WIDTH, SEED = 10, 20, 30, 5, 5, 16, 0
LR, EPOCHS, STRIDE, BATCH = 1e-3, 20, 5, 256
GAP = EVERY_S * FPS
TARGETS = [k * GAP - 1 for k in range(N + 1, 60 // EVERY_S + 1)]   # the frames of G_6..G_12, as fov_samples picks them


def parse_tile_csv(path):
    occ = []
    with open(path) as f:
        f.readline()
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or parts[0] == "":
                continue
            g = np.zeros((H, W), np.float32)
            for x in (int(p) for p in parts[1:] if p != ""):
                j, i = divmod(x, W)
                if 0 <= j < H and 0 <= i < W:
                    g[j, i] = 1.0
            occ.append(g)
    return np.stack(occ)


def load(video):
    paths = sorted(glob.glob(os.path.join(TILE_DIR, video, "*_tile.csv")),
                   key=lambda p: re.search(r"user(\d+)", p).group(1))
    assert len(paths) == 50, f"{video}: expected 50 viewers, found {len(paths)}"
    occ = [parse_tile_csv(p) for p in paths]
    nf = min(len(o) for o in occ)
    return [o[:nf] for o in occ]


def pairs(occ, ts):
    # input: 5 maps 5 s apart, oldest first, (B, H, W, n); target: the map 5 s after the newest
    X = [np.moveaxis(o[[t - GAP * (N - j) for j in range(N)]], 0, -1) for o in occ for t in ts]
    Y = [o[t] for o in occ for t in ts]
    return torch.from_numpy(np.stack(X)), torch.from_numpy(np.stack(Y))


class CircularPad(nn.Module):
    def __init__(self, p):
        super().__init__(); self.p = p
    def forward(self, x):
        p = self.p
        x = torch.cat([x[..., -p:], x, x[..., :p]], dim=3)
        return nn.functional.pad(x, (0, 0, p, p))


class Order5Predictor(nn.Module):   # the notebook's network, 3,073 weights
    def __init__(self, base_rate):
        super().__init__()
        self.pad1, self.conv1 = CircularPad(1), nn.Conv2d(N, WIDTH, 3)
        self.pad2, self.conv2 = CircularPad(2), nn.Conv2d(WIDTH, WIDTH, 3, dilation=2)
        self.head = nn.Conv2d(WIDTH, 1, 1)
        for c in (self.conv1, self.conv2, self.head):
            nn.init.xavier_uniform_(c.weight); nn.init.zeros_(c.bias)
        nn.init.constant_(self.head.bias, float(np.log(base_rate / (1 - base_rate))))
    def forward(self, x):
        h = x.permute(0, 3, 1, 2)
        h = torch.relu(self.conv1(self.pad1(h)))
        h = torch.relu(self.conv2(self.pad2(h)))
        return torch.sigmoid(self.head(h)).squeeze(1)


def train(occ):
    X, Y = pairs(occ, range(N * GAP, len(occ[0]), STRIDE))
    _, Y7 = pairs(occ, TARGETS)
    torch.manual_seed(SEED)
    m = Order5Predictor(float(Y7.mean()))         # base rate over the 7 sample pairs, as in the notebook
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    for _ in range(EPOCHS):
        for b in torch.randperm(len(X)).split(BATCH):
            loss = nn.functional.binary_cross_entropy(m(X[b]), Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def rmse(p, y):
    return float(torch.sqrt(torch.mean((p - y) ** 2)))


def one_fold(video, fold):
    torch.set_num_threads(1)
    occ = load(video)
    te = [o for i, o in enumerate(occ) if i % 5 == fold]
    m = train([o for i, o in enumerate(occ) if i % 5 != fold])
    Xs, Ys = pairs(te, TARGETS)
    Xc, Yc = pairs([np.mean(te, axis=0)], TARGETS)
    with torch.no_grad():
        return dict(video=video, fold=fold,
                    cv_single_rmse=rmse(m(Xs), Ys), cv_single_repeat_last=rmse(Xs[..., -1], Ys),
                    cv_crowd_rmse=rmse(m(Xc), Yc), cv_crowd_repeat_last=rmse(Xc[..., -1], Yc))


def self_check():
    # the dense pairs at the 7 target frames are exactly the notebook's make_pairs pairs
    o = np.random.default_rng(0).integers(0, 2, (1800, H, W)).astype(np.float32)
    samples = o[[k * GAP - 1 for k in range(1, 13)]]
    X, Y = pairs([o], TARGETS)
    assert np.array_equal(X[0].numpy(), np.moveaxis(samples[0:5], 0, -1)) and np.array_equal(Y[0].numpy(), samples[5])
    assert sum(p.numel() for p in Order5Predictor(0.16).parameters()) == 3073


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="*")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = ap.parse_args()
    self_check()
    vids = a.videos or sorted(d for d in os.listdir(TILE_DIR) if os.path.isdir(os.path.join(TILE_DIR, d)))
    rows = Parallel(n_jobs=a.jobs)(delayed(one_fold)(v, f) for v in vids for f in range(5))
    folds = pd.DataFrame(rows)
    cols = [c for c in folds.columns if c.startswith("cv_")]
    g = folds.groupby("video")[cols]
    cv = g.mean().join(g.std()[["cv_single_rmse", "cv_crowd_rmse"]].add_suffix("_std")).round(4)
    print(cv.sort_values("cv_crowd_rmse").to_string())

    out = os.path.join(HERE, "summary_rmse.csv")
    df = pd.read_csv(out, index_col="video") if os.path.exists(out) else pd.DataFrame(index=cv.index)
    df = df.drop(columns=[c for c in df.columns if c.startswith("cv_")]).join(cv, how="outer")
    if not a.videos:                    # a full run: drop rows of videos no longer in the dataset
        df = df.loc[df.index.intersection(vids)]
    df.to_csv(out, index_label="video")
    print("updated", out)
    plot(df)


def plot(df):
    # held-out RMSE per video, single viewer and crowd, each next to "repeat the last map"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    df = df.sort_values("cv_crowd_rmse")
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.55 * len(df) + 1.6), sharey=True, constrained_layout=True)
    for ax, kind, title in ((axes[0], "single", "single viewer"), (axes[1], "crowd", "10-viewer crowd (VAE input)")):
        ax.errorbar(df[f"cv_{kind}_rmse"], y, xerr=df[f"cv_{kind}_rmse_std"], fmt="o", color="#1f77b4",
                    capsize=3, label="FoV predictor (mean ± std, 5 folds)")
        ax.plot(df[f"cv_{kind}_repeat_last"], y, "s", color="#7f7f7f", label="repeat the last map")
        ax.set_title(f"Held-out RMSE, {title}", fontsize=10); ax.set_xlabel("RMSE (lower is better)")
        ax.grid(axis="x", alpha=.3)
    axes[0].set_yticks(y, df.index); axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=9)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    fig.savefig(os.path.join(HERE, "figures", "rmse_by_video.png"), dpi=140)
    print("saved figures/rmse_by_video.png")


if __name__ == "__main__":
    sys.exit(main())
