# Changes done

Date: 2026-11-21 · branch `prediction-check` · nothing committed yet (all changes are in the working tree).

## Summary

1. Kept the **5 videos with the lowest predictor error**: pacman, sport, coaster2, coaster1, game. Removed the other 5:
   landscape, panel, ride, diving, drive.
2. **Retrained every video's predictor with its own weights**, using a better training recipe. The CNN, its input and
   output, and the checkpoint format are unchanged, so the CVAE-GAN loads the predictors exactly as before.
3. Added **honest, held-out scores** (5-fold cross-validation over viewers). On crowd frames, the input of the
   CVAE-GAN, the RMSE is **0.10–0.12**.
4. Updated the docs, and added `WORKFLOW.md` and this file.

## Results

Held-out RMSE (scored on viewers the predictor never trained on; mean ± std over 5 folds):

| video | **crowd RMSE** (VAE input) | crowd: repeat last | single-viewer RMSE | single: repeat last | single-viewer RMSE before (training fit) |
|---|---|---|---|---|---|
| game | **0.0970 ± 0.0087** | 0.1153 | 0.2552 ± 0.0108 | 0.3277 | 0.2781 |
| coaster1 | **0.0997 ± 0.0084** | 0.1207 | 0.2546 ± 0.0191 | 0.3175 | 0.2729 |
| coaster2 | **0.1030 ± 0.0028** | 0.1150 | 0.2554 ± 0.0063 | 0.3112 | 0.2640 |
| pacman | **0.1215 ± 0.0073** | 0.1567 | 0.2310 ± 0.0154 | 0.3067 | 0.2479 |
| sport | **0.1224 ± 0.0095** | 0.1879 | 0.2418 ± 0.0142 | 0.3214 | 0.2602 |

- **Crowd RMSE is the reported number** (as agreed), because the CVAE-GAN takes the 10-viewer crowd frame.
- **Single-viewer RMSE cannot reach 0.1 with this CNN.** BCE, MSE, and 20 or 60 passes all ended within
  0.001 of each other. Going lower needs a bigger network or a different input, and both were ruled
  out to keep the CVAE-GAN working.

## 1. Videos removed (285 files, all still in git history)

| removed | why |
|---|---|
| `dataset/360dataset/sensory/tile/{landscape,panel,ride,diving,drive}/` | 50 tile CSVs each |
| `predictor/{landscape,panel,ride,diving,drive}/` | notebook, `.pt` weights, 5 figures each |

Ranking used (old single-viewer RMSE, all 50 viewers): pacman 0.2479, sport 0.2602, coaster2 0.2640, coaster1 0.2729,
game 0.2781 | removed: drive 0.2880, landscape 0.3221, panel 0.3237, ride 0.3643, diving 0.3903.

To restore one: `git checkout HEAD -- dataset/360dataset/sensory/tile/<video> predictor/<video>`.

## 2. Predictor training changed (template `predictor/coaster1/fov_predictor_coaster1.ipynb`)

**Unchanged:** the network (`Order5Predictor`, 3,073 weights); the input (last 5 maps, 5 s apart, shape `(10, 20, 5)`)
and output (`(10, 20)` probabilities); the checkpoint keys (`predictor`, `cfg`, `base_rate`, `U`, `U_test`, `rmse`,
`input_shape`, `output_shape`); the loader `load_pretrained_predictor` / `predict_fov`; and every old `cfg` field the
VAE notebooks read.

| | before | after |
|---|---|---|
| training pairs | 7 per viewer (only at 5, 10, …, 60 s), 350 per video | same input and target, but a pair starts every 5 frames: 210 per viewer, 10,500 per video |
| loss | squared error summed over tiles | binary cross-entropy |
| optimiser | 494 plain gradient-descent steps, one 10-viewer group each, $\eta_e = 0.1/(0.5+e)$ | Adam, lr $10^{-3}$, batch 256, 20 passes over all pairs |
| new `Cfg` fields | — | `loss`, `lr_adam`, `adam_epochs`, `pair_stride`, `batch` |

Code cells edited: `Cfg` (cell 3), data (cell 5: keeps every frame in `OCC`, adds `make_dense_pairs`), training
(cell 13: `train_epoch` with Adam replaces `epoch_batch` / `gd_step`). Markdown in sections 0, 1, 4 and 5 updated to match.

## 3. Five predictors retrained

`python predictor/make_video_predictors.py --run` rebuilt each video's notebook from the template and ran it. Every
notebook's checks passed (the loader reproduces the trained model exactly). Changed for each of coaster1, coaster2, game,
pacman and sport:

- `predictor/<video>/fov_predictor_<video>.ipynb` (with fresh outputs)
- `predictor/<video>/predictor_weights_<video>.pt` (new weights; each video's own)
- `predictor/<video>/figures/*.png` (5 figures)

All 5 checkpoints load with all keys the VAE reads, and give output `(batch, 10, 20)`.

## 4. New file: `predictor/cv_predictors.py`

Scores every video's predictor on held-out viewers: 5 folds (viewer $i$ → fold $i \bmod 5$, as in the VAE notebooks),
the same network and recipe as the notebook, scored at the notebook's 7 target times. It writes `cv_*` columns into
`summary_rmse.csv` and redraws `figures/rmse_by_video.png`. It includes a self-check that its pairs equal the notebook's
pairs and that the network has 3,073 weights. It is deterministic: a rerun reproduced the exploration numbers exactly.

## 5. Other files changed

| file | change |
|---|---|
| `predictor/summary_rmse.csv` | 5 rows only; new training-fit numbers plus `cv_single_*` and `cv_crowd_*` columns |
| `predictor/figures/rmse_by_video.png` | redrawn: held-out RMSE, single viewer and crowd, vs repeat-last |
| `predictor/fov_videos_summary.md` | rewritten for 5 videos, the new recipe and the held-out results |
| `fov_videos_summary.md` (repo root, untracked) | same content as the `predictor/` copy (it was an identical copy before) |
| `predictor/make_video_predictors.py` | usage example now names kept videos (`--videos pacman sport`) |
| `README.md` | layout lists 5 videos and `cv_predictors.py`; section 1 describes the new training, held-out coaster1 RMSE, and a note that the VAE results predate it |
| `WORKFLOW.md` (new, untracked) | project workflow and output log, written earlier this session; stage 1 updated |
| `changes_done.md` (new) | this file |

## 6. Checks run earlier this session (no repo changes)

- Whole pipeline (predictor → 4 VAE notebooks → DDPG notebook, shortened) ran end to end in a scratch copy on the
  earlier branch. That run found the diving predictor nearly flat, one reason diving is now removed.
- A full DDPG run with γ = 0.95 on 8K-20MHz only was started and stopped by Claude Code for low memory before finishing.
  There is no result. The DDPG folder is not on this branch.

## 7. Not done / still open

- **The CVAE-GAN results are stale.** `vae/results/`, `vae/cvaegan_encoder.pt` and README sections 3–7 were produced
  with the old coaster1 predictor. The `vae/` test notebooks also retrain a predictor in each fold with their own copy of
  the old procedure (`train_predictor`). To update: switch `train_predictor` to the new recipe and rerun with
  `RERUN = True`. Not run here: it retrains hundreds of models.
- The crowd frames change little over time. Knowing a held-out crowd's average view over the whole video would already
  score 0.07–0.10 on 4 of the 5 videos. That is not a usable predictor (it looks at the future), but it shows the crowd
  RMSE leaves little room between a good and a trivial prediction.
- `conference_101719.tex` and `Capstone_Phase3_ISA2.pptx` were not checked or changed. If they quote the old RMSE
  values or ten videos, update them by hand.
- The nested `cvae-gan-test/` folder in the repo root (untracked) was left alone.
