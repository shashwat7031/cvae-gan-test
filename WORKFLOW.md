# Project workflow and output log

How the whole project works, end to end, and a record of **every output produced so far** (as of 2026-11-21,
commit `8172ffd`). `README.md` holds the full reasoning for the CVAE-GAN settings; this file is the map and the logbook.
Update section 5 whenever a notebook is rerun or a new output is committed.

---

## 1. The pipeline at a glance

```
 dataset/360dataset/sensory/tile/<video>_userNN_tile.csv   (5 videos x 50 viewers)
        │  tile list per frame  →  10 x 20 FoV map every 5 s (12 maps per viewer)
        ▼
 ┌──────────────────────────────┐
 │ STAGE 1  FoV predictor       │  predictor/fov_predictor.ipynb
 │ last 5 maps → next map       │  out: predictor_weights_trained.pt (coaster)
 └──────────────┬───────────────┘
                │  predicted crowd frame F̂_k (10 x 20, avg of 10 viewers)
                ▼
 ┌──────────────────────────────┐
 │ STAGE 2  CVAE-GAN encoder    │  vae/*.ipynb (explainer + 3 tests)
 │ (F̂ − X̄)/s → z → sigmoid(z)   │  out: cvaegan_encoder.pt, results/*.csv
 │ 200 numbers → 10 numbers     │  final: normalised, τ = (0.1, 0.89, 0.01)
 └──────────────┬───────────────┘
                │  sigmoid(z) ∈ (0,1)^10, both models FROZEN from here on
                ▼
 ┌──────────────────────────────┐
 │ STAGE 3  DDPG policy network │  ddpg_policy_network/DDPG_Policy_Network_FULL.ipynb
 │ state (28) → action (12):    │  out: ddpg_actor/critic_*.pt, logs, figures
 │ bandwidth + power per video  │  STATUS: not run yet (see §5.3)
 └──────────────────────────────┘
```

Goal: predict where a crowd of 360° viewers will look 5 s ahead, compress that prediction into 10 numbers (one per
SVC layer), and let an RL agent use those numbers to split bandwidth and power across 5 videos multicast to 50
headsets.

---

## 2. Setup

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate      Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt        # torch, numpy, pandas, ipykernel, matplotlib, joblib
pip install jupyterlab                 # needed for the DDPG notebook
```

Run every notebook **from its own folder**; all paths are relative. CPU is fine (CUDA requirement removed in
`8172ffd`); the DDPG notebook also uses CUDA or Apple MPS if present.

---

## 3. Run order

| # | notebook | needs | writes | rerun switch |
|---|---|---|---|---|
| 1 | `predictor/fov_predictor.ipynb` | tile CSVs | `predictor/predictor_weights_trained.pt`, `predictor/figures/*` | — |
| 2 | `vae/cvae_gan.ipynb` *(optional explainer)* | step 1 weights | `vae/figures/cvaegan_architecture.png`, `training_curves.png`, `latent_*.png`, `test_group_*.png` | — |
| 3 | `vae/cvae_gan_diff_weights.ipynb` (test 1) | step 1 | `vae/results/diff_weights_{validation,test}.csv`, `vae/figures/diff_weights_*` | `RERUN = True` |
| 4 | `vae/cvae_gan_normalised_tiles.ipynb` (test 2) | step 1 | `vae/results/normalised_tiles_test.csv`, `vae/figures/normalised_tiles_*` | `RERUN = True` |
| 5 | `vae/cvae_gan_normalised_and_diff_weights.ipynb` (test 3, final) | steps 1, 3, 4 | `vae/cvaegan_encoder.pt`, `normalised_and_diff_weights_*.csv`, `summary_test.csv`, figures | `RERUN = True` |
| 6 | copy weights into `ddpg_policy_network/outputs_predictor_vae_combined/` | steps 1, 5 | `cvaegan_encoder.pt`, `predictor_weights_<video>.pt` ×5 | — |
| 7 | `ddpg_policy_network/DDPG_Policy_Network_FULL.ipynb` | step 6 | `ddpg_*` files in the outputs folder (§5.3) | `RETRAIN = True` (default) |

Notebooks 3–5 load their saved CSVs instead of retraining unless `RERUN = True`.

---

## 4. What each stage does

### Stage 1: FoV predictor (`predictor/`)
- **Input:** each viewer's tile CSV → binary 10 × 20 map (rows = latitude, cols = longitude), sampled every 5 s
  → 12 maps per viewer (60-s video at 30 fps).
- **Model:** small CNN, 3,073 weights, circular padding on longitude (360° seam). Input = last `n = 5` maps,
  output = probability per tile for the next map.
- **Training:** 494 epochs, each on a random group of 10 of the 50 viewers (`cfg`: `lr_a=0.1`, `lr_b=0.5`,
  `width=16`, `seed=0`, video `coaster`).
- **Checkpoint keys:** `predictor, cfg, base_rate, U, U_test, rmse, input_shape, output_shape, torch_version`.
  Loaded through `predict_fov(x)` (section 7 of the notebook); the model is frozen (`eval()`, no grad).

### Stage 2: CVAE-GAN (`vae/`)
- **Input X:** predicted crowd frame F̂_k (average of 10 viewers through the predictor), k = 6..12 → 7 frames per
  epoch; 988 epochs.
- **Model (Nie et al. [14]):** encoder → (μ, σ) for L = 10 slots → z = μ + σε → sigmoid → decoder; plus a
  discriminator. Loss = τ₁·KL + τ₂·½‖X − X_recon‖² + τ₃·GAN, with τ₁ + τ₂ + τ₃ = 1.
- **Problem found:** with equal τ the latent collapses (0 active slots) because frames differ from the mean by
  only ≈0.03 while τ₁/τ₂ = σ_x² = 1 assumes noise of ±1.
- **Selection method:** nested CV over viewers. 5 folds of 10 (viewer i → fold i mod 5), next fold = validation,
  predictor retrained per turn, 5 repeats per setting, lowest mean validation RMSE wins (rule fixed in advance).
- **Tests:** (1) 18 τ settings on original frames; (2) normalisation X' = (X − X̄)/s with equal τ;
  (3) normalisation + the 18 τ settings → final model.
- **Final choice:** normalised frames, τ = (0.1, 0.89, 0.01).

### Stage 3: DDPG policy (`ddpg_policy_network/`)
- **Setup:** 5 videos (coaster, coaster2, diving, drive, game) in Zipf rank order (s = 0.8 → 0.39/0.22/0.16/0.13/0.11),
  50 headsets, 10-viewer crowd per video, 7 intervals (k = 6..12) per episode, one step per watched video.
- **Front end (frozen):** one predictor per video + the common encoder; frame normalised with saved X̄, s.
- **State (28):** sigmoid(z) ×10, group attention w ×10, path-loss μ, σ, mean/min buffer, B_rem/B_tot,
  P_rem/P_tot, N_v/50, videos left/V.
- **Action (12):** softmax [b₀ keep-for-later, b₁..b₁₀ per attention group] + sigmoid ρ power share. The last
  video of an interval takes everything left.
- **Tiles → groups:** F̂ cut into 10 equal intervals; a tile in group l targets SVC layers 1..l.
- **Channel:** 3.5 GHz, 20 or 10 MHz, 23 dBm, ring 35–200 m, 8 dB shadowing (50 m decorrelation), Rayleigh fading,
  1.5 m/s walk.
- **Bitrates:** R-Q ladder, QP_k = 38 − 2(k−1), SVC overhead 0.09(k−1); all layers of a 4K frame = 40 Mbit/s
  (160 Mbit/s at 8K).
- **Reward:** Seer's QoE (Kumar et al., TNSE 2023): quality − rebuffering − |Δquality|, weighted by audience share.
- **Networks:** actor 28→128→128→(11 softmax, 1 sigmoid); critic 40→128→128→1. Replay buffer, batch 256,
  2 updates/step, soft update κ = 0.01, ε-greedy Gaussian logit noise, ε × 0.99 per episode.
- **Trainings (6 × 6000 episodes):** γ ∈ {0.9, 0.95, 0.99} on 8K-20MHz; best γ (undiscounted validation return)
  reused on 8K-10MHz, 4K-20MHz, 4K-10MHz.
- **Evaluation:** 50 fixed test episodes per scenario vs 8 rules (group rule: attention / equal per tile /
  demand / random × video rule: equal split / split by audience).

---

## 5. Output log: everything produced so far

### 5.1 Stage 1: predictor

> **Updated 2026-11-21 (branch `prediction-check`):** one predictor per video, top 5 videos only, retrained with
> Adam + BCE on dense pairs. Same CNN, input and output. Details in `changes_done.md` and `predictor/fov_videos_summary.md`.

| file | what it is |
|---|---|
| `predictor/<video>/predictor_weights_<video>.pt` | frozen predictor per video: coaster1, coaster2, game, pacman, sport |
| `predictor/<video>/fov_predictor_<video>.ipynb` | trains it (copy of the coaster1 template) |
| `predictor/<video>/figures/*.png` | timeline, single prediction, all viewers actual/predicted, crowd heatmap |
| `predictor/summary_rmse.csv` | training-fit and held-out (`cv_*`) RMSE per video |
| `predictor/figures/rmse_by_video.png` | held-out RMSE per video, single viewer and crowd |

**Held-out RMSE (5-fold over viewers, lower is better):**

| video | crowd (VAE input) | single viewer |
|---|---|---|
| game | **0.0970** | 0.2552 |
| coaster1 | **0.0997** | 0.2546 |
| coaster2 | **0.1030** | 0.2554 |
| pacman | **0.1215** | 0.2310 |
| sport | **0.1224** | 0.2418 |

The previous single coaster predictor (494 plain-GD steps) scored 0.2729 on all 50 (training) viewers; the
CVAE-GAN results below still come from it.

### 5.2 Stage 2: CVAE-GAN

**Weights:** `vae/cvaegan_encoder.pt`. Keys: `encoder`, `L=10`, `hidden=[128, 64]`, `leak=0.2`,
`input_shape=(10, 20)`, `tau=[0.1, 0.89, 0.01]`, `normalised=True`, `latent_sigmoid=True`, `norm_mean` (X̄),
`norm_scale=0.03034` (s), `repeat=2`, `seed=2111359669`, `train_rmse=0.00978`, `active=10`.

**Result CSVs** (`vae/results/`; columns `fold, stage, tau_1, tau_2, tau_3, normalised, repeat, seed, rmse, active,
kl, mean_frame_rmse`):

| file | rows | contents |
|---|---|---|
| `diff_weights_validation.csv` | 450 | test 1: 18 τ × 5 folds × 5 repeats, validation |
| `diff_weights_test.csv` | 25 | test 1: chosen τ per fold × 5 repeats, test |
| `normalised_tiles_test.csv` | 50 | test 2: original vs normalised, equal τ, 5 folds × 5 repeats |
| `normalised_and_diff_weights_validation.csv` | 450 | test 3: 18 τ on normalised frames, validation |
| `normalised_and_diff_weights_test.csv` | 25 | test 3: chosen τ per fold, test |
| `summary_test.csv` | 5 rows (+ duplicate block, see §6) | final comparison table |

**Final comparison (test folds, 25 runs each):**

| setup | test RMSE | active slots / 10 |
|---|---|---|
| always the average frame | 0.0301 | – |
| original frames, equal τ | 0.0304 ± 0.0067 | 0.0 |
| original frames, τ chosen on validation | 0.0254 ± 0.0061 | 9.1 |
| normalised frames, equal τ | 0.0110 ± 0.0021 | 6.9 |
| **normalised frames, τ chosen on validation** | **0.0101 ± 0.0016** | **10.0** |

**τ chosen per fold (from the test CSVs, mean of 5 repeats):**

| fold | test 1 (original): τ₁, τ₃ → RMSE | test 3 (normalised): τ₁, τ₃ → RMSE | mean-frame baseline |
|---|---|---|---|
| 1 | 1e-4, 1/3 → 0.0177 | 0.01, 1/3 → 0.0094 | 0.0253 |
| 2 | 1e-5, 0.01 → 0.0221 | 0.01, 0.01 → 0.0092 | 0.0266 |
| 3 | 1e-4, 0.01 → 0.0257 | 0.01, 0.1 → 0.0103 | 0.0263 |
| 4 | 1e-4, 0.1 → 0.0315 | 1e-5, 1/3 → 0.0129 | 0.0328 |
| 5 | 1e-4, 0.01 → 0.0300 | 0.1, 0.01 → 0.0087 | 0.0391 |

Over all 36 settings the validation winner is normalised + τ = (0.1, 0.89, 0.01) at 0.0110 ± 0.0023.

**Figures** (`vae/figures/`): `cvaegan_architecture`, `training_curves`, `latent_slots`, `latent_vectors`,
`test_group_frames`, `test_group_reconstruction`; per test, `*_eigenvalues`, `*_example`, `*_example_reconstruction`,
`*_example_slots`, `*_validation` / `*_test`, `*_values`; plus `normalised_and_diff_weights_comparison`.
README figures in `images/`: `normalisation.png`, `normalisation_grid.png`, `tau_choice.png`, `test_comparison.png`.

### 5.3 Stage 3: DDPG

**Present now** (`ddpg_policy_network/outputs_predictor_vae_combined/`, inputs only):
- `cvaegan_encoder.pt`: same as `vae/cvaegan_encoder.pt`
- `predictor_weights_{coaster,coaster2,diving,drive,game}.pt`: one predictor per video (base rates 0.165,
  0.161, 0.192, 0.161, 0.161)

**Not produced yet.** The notebook has no saved cell outputs and none of its output files exist. A full run
(`RETRAIN = True`) will write:

| file | contents |
|---|---|
| `ddpg_actor.pt`, `ddpg_critic.pt`, `ddpg_actor_{tag}.pt`, `ddpg_critic_{tag}.pt` | agents per γ / scenario |
| `ddpg_episode_log.csv`, `ddpg_episode_log_{tag}.csv` | per-episode training log |
| `ddpg_test_summary.csv` | DDPG vs 8 rules, per scenario |
| `ddpg_channel.png`, `ddpg_masks.png`, `ddpg_training.png`, `ddpg_q_values.png`, `ddpg_test_policies.png`, `ddpg_one_interval.png` | figures |

When it has run, add the chosen γ, the test table per scenario and the per-rank bandwidth/power shares here.

### 5.4 History of outputs

| date | commit | author | change |
|---|---|---|---|
| 2026-11-14 | `e218164` | Shashank | predictor, CVAE-GAN notebooks, first results |
| 2026-11-14 | `70a4ea9` | Shashank | Windows compatibility (joblib parallel runs) |
| 2026-11-14 | `69ffaaa` | Shashank | sigmoid added to encoder; tests rerun, results/figures regenerated |
| 2026-11-15 | `48b5e45` | Shashank | README wording: why these weights |
| 2026-11-20 | `261cf6a` | Rayn | DDPG notebook + per-video predictors + encoder copy |
| 2026-11-20 | `8172ffd` | Rayn | CUDA torch requirement removed |

---

## 6. Known issues / open points

- `vae/results/summary_test.csv` holds its header and first two rows twice (an earlier write was not overwritten).
  The last block is the correct table.
- `ddpg_policy_network/README.md` and the notebook intro say `predictor_<video>.pt` and `Predictor_VAE_Combined.ipynb`.
  The files are actually named `predictor_weights_<video>.pt`, and that notebook is not in this repo.
- `outputs_predictor_vae_combined/predictor_weights_coaster.pt` has different weights from
  `predictor/predictor_weights_trained.pt` (same base rate, so most likely a separate training run). Decide which one is canonical.
- The encoder was trained on **coaster** frames. The DDPG notebook warns if the encoder does not list all 5 videos.
  For valid DDPG results, retrain it on the pooled frames of all 5 videos.
- Zipf rank order of the videos is assumed (the dataset has no popularity data), and so is the extra state
  "videos left".
- README setup uses `source .venv/bin/activate`. On Windows it is `.venv\Scripts\activate`.

---

## 7. Keeping this file up to date

After any run that changes an output:
1. Update the matching table in §5 (file, numbers, date).
2. Add a row to §5.4 with the commit.
3. Move fixed items out of §6.
