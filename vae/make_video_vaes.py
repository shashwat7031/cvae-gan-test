"""Build (and optionally run) the four CVAE-GAN notebooks of every video in the dataset.

The templates are the four notebooks in vae/coaster1/. Every other video's notebooks are copies of them with only the
video changed: the FoV predictor they load (../../predictor/<video>/predictor_weights_<video>.pt, which also gives the
tile folder and file prefix of the video through its cfg), and the file names and texts that name the video. Each
video's notebooks live in vae/<video>/ with their results/, figures/ and the final frozen encoder <video>_frozen_wt.pt.

The notebooks are run in this order, because the last experiment reads the results of the two before it, and the
base notebook prints the summary of all three:
    <video>_vae_normalised_tiles.ipynb  ->  <video>_vae_diff_weights.ipynb
    ->  <video>_vae_normalised_and_diff_weights.ipynb  ->  <video>_vae.ipynb

Usage (from the repo root or from vae/):
    python vae/make_video_vaes.py                      # build notebooks only
    python vae/make_video_vaes.py --run                # build, execute, and write vae/summary_tau.csv
    python vae/make_video_vaes.py --run --videos pacman sport
    python vae/make_video_vaes.py --summary            # only (re)write vae/summary_tau.csv from saved results
"""
import argparse, glob, os, shutil, sys, time, traceback

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_VIDEO = "coaster1"
TEMPLATE_DIR = os.path.join(HERE, TEMPLATE_VIDEO)
SUFFIXES = ["_vae_normalised_tiles", "_vae_diff_weights", "_vae_normalised_and_diff_weights", "_vae"]   # run order
TILE_DIR = os.path.join(HERE, "..", "dataset", "360dataset", "sensory", "tile")
PRED_DIR = os.path.join(HERE, "..", "predictor")


def videos_in_dataset():
    # one video per sub-folder of sensory/tile/ that holds <prefix>_userNN_tile.csv files
    return sorted(d for d in os.listdir(TILE_DIR)
                  if os.path.isdir(os.path.join(TILE_DIR, d)) and glob.glob(os.path.join(TILE_DIR, d, "*_tile.csv")))


def nb_path(video, suffix):
    return os.path.join(HERE, video, f"{video}{suffix}.ipynb")


def build_notebooks(video):
    pred = os.path.join(PRED_DIR, video, f"predictor_weights_{video}.pt")
    assert os.path.exists(pred), f"{pred} not found; run predictor/make_video_predictors.py --run first"
    out = []
    for suffix in SUFFIXES:
        nb = nbformat.read(nb_path(TEMPLATE_VIDEO, suffix), as_version=4)
        for cell in nb.cells:
            if video != TEMPLATE_VIDEO:
                cell.source = cell.source.replace(TEMPLATE_VIDEO, video)
            if cell.cell_type == "code":
                cell.outputs, cell.execution_count = [], None
        code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
        if video != TEMPLATE_VIDEO:   # sanity: the video really changed and no stale names remain
            assert TEMPLATE_VIDEO not in "\n".join(c.source for c in nb.cells), f"{video}{suffix}: '{TEMPLATE_VIDEO}' left"
        assert f'"predictor", "{video}", "predictor_weights_{video}.pt"' in code, f"{video}{suffix}: predictor path"
        assert '"..", "..", "dataset"' in code, f"{video}{suffix}: dataset path"
        for sub in ("figures", "results"):
            os.makedirs(os.path.join(HERE, video, sub), exist_ok=True)
        if video != TEMPLATE_VIDEO:
            nbformat.write(nb, nb_path(video, suffix))
        out.append(nb_path(video, suffix))
    # the architecture diagram is a drawn figure, not made by the notebooks: every video's base notebook shows it
    arch = os.path.join("figures", "cvaegan_architecture.png")
    if video != TEMPLATE_VIDEO:
        shutil.copyfile(os.path.join(TEMPLATE_DIR, arch), os.path.join(HERE, video, arch))
    return out


def run_notebook(path):
    from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
    nb = nbformat.read(path, as_version=4)
    ep = ExecutePreprocessor(timeout=None, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": os.path.dirname(path)}})
        ok = True
    except CellExecutionError:
        traceback.print_exc()
        ok = False
    nbformat.write(nb, path)          # saved with its outputs (and the error, if it failed)
    return ok


def summary_row(video):
    import pandas as pd, torch
    d = os.path.join(HERE, video)
    enc_file, res = os.path.join(d, f"{video}_frozen_wt.pt"), os.path.join(d, "results")
    if not os.path.exists(enc_file):
        return None
    ck = torch.load(enc_file, map_location="cpu", weights_only=False)
    key = ["normalised", "tau_1", "tau_2", "tau_3"]
    val = pd.concat([pd.read_csv(os.path.join(res, f)) for f in
                     ("diff_weights_validation.csv", "normalised_and_diff_weights_validation.csv")])
    g = val.groupby(key).rmse.agg(["mean", "std"]).reset_index().sort_values("mean").iloc[0]
    test = pd.read_csv(os.path.join(res, "summary_test.csv")).set_index("how the model is set up")["test RMSE"]
    eq = pd.read_csv(os.path.join(res, "normalised_tiles_test.csv"))
    return {"video": video, "normalised": ck["normalised"],
            "tau_1": ck["tau"][0], "tau_2": ck["tau"][1], "tau_3": ck["tau"][2],
            "val_rmse": round(g["mean"], 4), "val_rmse_std": round(g["std"], 4),
            "test_rmse_chosen": test["frames and weights both chosen by validation"],
            "test_rmse_equal_original": test["original frames, equal weights"],
            "test_rmse_equal_normalised": test["normalised frames, equal weights"],
            "mean_frame_rmse": round(eq.mean_frame_rmse.mean(), 4),
            "encoder_train_rmse": round(ck["train_rmse"], 4), "encoder_active_slots": ck["active"]}


def write_summary(videos):
    import pandas as pd
    rows = [r for r in (summary_row(v) for v in videos) if r]
    if not rows:
        return
    df = pd.DataFrame(rows).set_index("video")
    out = os.path.join(HERE, "summary_tau.csv")
    if os.path.exists(out):           # merge a partial rerun into the existing summary
        old = pd.read_csv(out, index_col="video")
        df = pd.concat([old.drop(index=df.index, errors="ignore"), df]).sort_index()
    df.to_csv(out)
    print("\n", df.to_string(), sep="")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="execute each video's notebooks after building them")
    ap.add_argument("--summary", action="store_true", help="only write summary_tau.csv from saved results")
    ap.add_argument("--videos", nargs="*", help="subset of videos (default: all in the dataset)")
    a = ap.parse_args()

    vids = a.videos or videos_in_dataset()
    assert vids, f"no tile CSVs in {TILE_DIR}"
    print("videos:", ", ".join(vids), flush=True)
    if a.summary:
        return write_summary(vids)

    failed = []
    for v in vids:
        paths = build_notebooks(v)
        print(f"[{v}] built {len(paths)} notebooks in vae/{v}/", flush=True)
        if not a.run:
            continue
        for p in paths:
            t0 = time.time()
            print(f"[{v}] running {os.path.basename(p)} ...", flush=True)
            ok = run_notebook(p)
            print(f"[{v}] {os.path.basename(p)} {'done' if ok else 'FAILED'} in {(time.time() - t0) / 60:.1f} min", flush=True)
            if not ok:
                failed.append(os.path.relpath(p, HERE))
                break                 # the later notebooks of this video depend on this one
        write_summary([v])
    if failed:
        print("\nFAILED:", ", ".join(failed))
        return 1


if __name__ == "__main__":
    sys.exit(main())
