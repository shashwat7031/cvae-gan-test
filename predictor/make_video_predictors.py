"""Build (and optionally run) one FoV-predictor notebook per video in the dataset.

The template is predictor/coaster1/fov_predictor_coaster1.ipynb. Every other video's notebook is a copy of it with
only the video changed: Cfg.video / Cfg.folder / Cfg.prefix (its folder under sensory/tile/ and the filename prefix of
its CSVs, e.g. folder coaster1 holds coaster_userNN_tile.csv), the paths and texts that name the video, and the
checkpoint name predictor_weights_<video>.pt. Each notebook lives in predictor/<video>/ with its figures/ and .pt.

Usage (from the repo root or from predictor/):
    python predictor/make_video_predictors.py            # build notebooks only
    python predictor/make_video_predictors.py --run      # build, execute, and write summary_rmse.csv
    python predictor/make_video_predictors.py --run --videos diving sport
"""
import argparse, glob, os, re, sys

import nbformat

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_VIDEO, TEMPLATE_PREFIX = "coaster1", "coaster"
TEMPLATE = os.path.join(HERE, TEMPLATE_VIDEO, f"fov_predictor_{TEMPLATE_VIDEO}.ipynb")
TILE_DIR = os.path.join(HERE, "..", "dataset", "360dataset", "sensory", "tile")


def videos_in_dataset():
    # one video per sub-folder of sensory/tile/ that holds <prefix>_userNN_tile.csv files
    return sorted(d for d in os.listdir(TILE_DIR)
                  if os.path.isdir(os.path.join(TILE_DIR, d)) and glob.glob(os.path.join(TILE_DIR, d, "*_tile.csv")))


def prefix_of(folder):
    rx = re.compile(r"^(.*)_user\d+_tile\.csv$")
    prefixes = {m.group(1) for p in glob.glob(os.path.join(TILE_DIR, folder, "*_tile.csv"))
                if (m := rx.match(os.path.basename(p)))}
    assert len(prefixes) == 1, f"expected one filename prefix in {folder}/, found {sorted(prefixes)}"
    return prefixes.pop()


def replace_once(src, old, new, where):
    assert old in src, f"template changed: '{old}' not found in {where}"
    return src.replace(old, new)


def build_notebook(video):
    prefix = prefix_of(video)
    nb = nbformat.read(TEMPLATE, as_version=4)
    for cell in nb.cells:
        s = cell.source
        if cell.cell_type == "code" and 'video: str = "' in s:      # the Cfg cell: set video, folder, prefix
            s = replace_once(s, f'video: str = "{TEMPLATE_VIDEO}" ', f'video: str = "{video}" ', "Cfg cell")
            s = replace_once(s, f'folder: str = "{TEMPLATE_VIDEO}" ', f'folder: str = "{video}" ', "Cfg cell")
            s = replace_once(s, f'prefix: str = "{TEMPLATE_PREFIX}" ', f'prefix: str = "{prefix}" ', "Cfg cell")
        else:                                                       # paths, file names and titles naming the video
            s = s.replace(f"{TEMPLATE_VIDEO}/{TEMPLATE_PREFIX}_user", f"{video}/{prefix}_user")
            s = s.replace(TEMPLATE_VIDEO, video)
        cell.source = s
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    # sanity: the video really changed and no stale paths remain
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert f'video: str = "{video}"' in code and f"predictor_weights_{video}.pt" in code
    out_dir = os.path.join(HERE, video)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"fov_predictor_{video}.ipynb")
    nbformat.write(nb, path)
    return path


def run_notebook(path):
    from nbconvert.preprocessors import ExecutePreprocessor
    nb = nbformat.read(path, as_version=4)
    ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(
        nb, {"metadata": {"path": os.path.dirname(path)}})
    nbformat.write(nb, path)
    return nb


def baselines_from_outputs(nb):
    # parse the section-5 table printed by the notebook: '<name>  <all 50>  <test group>'
    out = {}
    for c in nb.cells:
        if c.cell_type == "code" and "rmse_table" in c.source:
            text = "".join(o.get("text", "") for o in c.outputs if o.get("output_type") == "stream")
            for name, key in (("average occupancy", "avg_occupancy"), ("repeat last map", "repeat_last")):
                m = re.search(rf"{name}\s+([\d.]+)\s+([\d.]+)", text)
                if m:
                    out[f"{key}_all_50"], out[f"{key}_test_group"] = float(m.group(1)), float(m.group(2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", help="execute each notebook after building it")
    ap.add_argument("--videos", nargs="*", help="subset of videos (default: all in the dataset)")
    a = ap.parse_args()

    vids = a.videos or videos_in_dataset()
    assert vids, f"no tile CSVs in {TILE_DIR}"
    print("videos:", ", ".join(vids))

    rows = []
    for v in vids:
        p = build_notebook(v)
        print(f"[{v}] built {os.path.relpath(p, HERE)}", flush=True)
        if a.run:
            nb = run_notebook(p)
            import torch
            ck = torch.load(os.path.join(HERE, v, f"predictor_weights_{v}.pt"), weights_only=True)
            rows.append({"video": v, "rmse_all_50": ck["rmse"]["all_50"],
                         "rmse_test_group": ck["rmse"]["test_group"],
                         **baselines_from_outputs(nb), "base_rate": ck["base_rate"]})
            print(f"[{v}] done: RMSE all 50 {ck['rmse']['all_50']:.4f}, "
                  f"test group {ck['rmse']['test_group']:.4f}", flush=True)

    if rows:
        import pandas as pd
        df = pd.DataFrame(rows).set_index("video").round(4)
        out = os.path.join(HERE, "summary_rmse.csv")
        if a.videos and os.path.exists(out):          # merge a partial rerun into the existing summary
            old = pd.read_csv(out, index_col="video")
            df = pd.concat([old.drop(index=df.index, errors="ignore"), df]).sort_index()
        df.to_csv(out)
        print("\n", df.to_string(), sep="")


if __name__ == "__main__":
    sys.exit(main())
