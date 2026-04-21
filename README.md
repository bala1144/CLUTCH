# CLUTCH: Contextualized Language model for Unlocking Text-Conditioned Hand motion modelling in the wild

**CLUTCH: Contextualized Language model for Unlocking Text-Conditioned Hand motion modelling in the wild**  
Balamurugan Thambiraja, Omid Taheri, Radek Danecek, Giorgio Becherini, Gerard Pons-Moll, Justus Thies  
ICLR 2026  
[[Paper]](https://openreview.net/forum?id=W7YRskO47j) | [[Project Page]](https://balamuruganthambiraja.github.io/CLUTCH/) | [[Video]](https://balamuruganthambiraja.github.io/CLUTCH/)

---

## Overview

CLUTCH is a unified language-model framework for bimanual hand-object interaction generation. It supports:
- **Text-to-Motion (T2M)**: Generate hand motion from text descriptions
- **Motion-to-Text (M2T)**: Describe hand motion in natural language

The system follows a two-stage pipeline:
1. **VQVAE** — Decomposed vector-quantization of hand trajectory and hand-pose independently
2. **LLM (T5-base)** — A language model trained on motion tokens with instruction tuning

---

## Setup

### 1. Clone

```bash
git clone https://github.com/[your-repo]/CLUTCH.git
cd CLUTCH
```

### 2. Create environment

**Option A — automated script (recommended):**
```bash
bash setup_env.sh
conda activate clutch
```

**Option B — from full environment file:**
```bash
conda env create -f environment.yml
conda activate work39_torch2
```

### 3. Download assets

Bootstrap external runtime dependencies first:

```bash
./download_deps.sh
```

This prepares:
- `deps/t2m/` for quantitative evaluator metrics
- `deps/glove/` for EgoVid5M dataloading

The T2M and M2T scripts can still run smoke tests without the evaluator bundle, but `deps/glove/` is required by the current datamodule.

### 4. Download assets

CLUTCH resolves runtime assets from `./assets` by default.

Install the uploaded asset bundle:

```bash
./download_assets.sh --url <clutch-assets-zip-url>
```

For local testing before upload:

```bash
python tools/download_bundle.py assets --archive /path/to/clutch-assets.zip --force
```

The same bundle system is intended to be extended to pretrained checkpoints later.

The asset bundle is expected to contain:
- MANO runtime assets
- the released normalization file under `assets/egovid5m_release/`

---

## Dataset Preparation

### EgoVid5M

Download and preprocess EgoVid5M following the dataset instructions.  
Set `DATASET.dataset_dir` in configs to `/path/to/EgoVid5M_dataset`.

The dataset directory should contain:
```
EgoVid5M_dataset/
├── _aux_exp/text_processing_valid_index_nset_72_tr0_013_rot0_2/
│   ├── split_dict_tr0_8_val0_1_test0_1.json
│   └── mean_std_w_s20.npy
├── mano_smoothened_w_gaus_tr_sig2_rotated_180.npy
├── naive_claude_summary.json
├── template_pretrain_w_hand_2_hand_w_text.json
└── template_instructions_hand_motion.json
```

For the released prompt-demo/checkpoint path, only the normalization file is currently
staged inside `assets/egovid5m_release/`. The rest of the dataset-side metadata remains
part of the separate dataset release.

---

### GRAB (optional)

Set `DATASET.Grab_cfg.dataset_dir` in configs to `/path/to/GRAB_dataset`.

---

## Training

Training follows **4 sequential stages**. Update `FOLDER` (output dir) and `DATASET.dataset_dir` in each config before running.

### Stage 1a — VQVAE Trajectory

Trains the trajectory (global hand movement) tokenizer.

```bash
python train.py --cfg configs/stage1a_vqvae_traj.yaml --nodebug
```

Output checkpoint: `<FOLDER>/mgpt_GRAB/<exp_name>/checkpoints/epoch=999.ckpt`

### Stage 1b — VQVAE Hand-Pose

Trains the hand-pose (finger articulation) tokenizer.

```bash
python train.py --cfg configs/stage1b_vqvae_hp.yaml --nodebug
```

Output checkpoint: `<FOLDER>/mgpt_GRAB/<exp_name>/checkpoints/epoch=999.ckpt`

### Stage 1c — Extract Motion Tokens

Before LLM training, convert all motion sequences to discrete tokens using the trained VQVAEs.  
Update `TRAIN.PRETRAINED_VAE.traj_model` and `TRAIN.PRETRAINED_VAE.hp_model` in `configs/stage1c_extract_tokens.yaml` with the checkpoints from stages 1a and 1b.

```bash
python extract_codes.py --cfg configs/stage1c_extract_tokens.yaml
```

Tokens are saved to `<dataset_dir>/<DATASET.CODE_PATH>/`.

### Stage 2 — LLM Pretraining

Pretrain the T5-base language model on motion tokens with text supervision.  
Update `TRAIN.PRETRAINED_VAE` paths in `configs/stage2_llm_pretrain.yaml`.

```bash
python train.py --cfg configs/stage2_llm_pretrain.yaml --nodebug
```

### Stage 3 — Geometry Alignment

Fine-tune with geometry alignment loss. Update `TRAIN.PRETRAINED` with the stage 2 checkpoint.

```bash
python train.py --cfg configs/stage3_llm_align.yaml --nodebug
```

### Stage 4 — Instruction Tuning

Final instruction-tuning stage across the released tasks (T2M and M2T).  
Update `TRAIN.PRETRAINED` with the stage 3 checkpoint.

```bash
python train.py --cfg configs/stage4_llm_instruct.yaml --nodebug
```

---

## Inference / Evaluation

All evaluation scripts take the path to the trained LLM checkpoint and a config file.

### Text-to-Motion

```bash
cd llm/MotionGPT  # or from repo root
python test/test_t2m.py --test_t2m_model /path/to/checkpoints/llm_instruct.ckpt \
    --device 0 --nodebug
```

### Motion-to-Text

```bash
python test/test_m2t.py --test_t2m_model /path/to/checkpoints/llm_instruct.ckpt \
    --device 0 --nodebug
```

> The TM2T evaluator checkpoint (`cfg.METRIC.TM2T["ckpt"]`) must also be set — see [Pretrained Models](#pretrained-models).

---

## Pretrained Models

| Model | Description | Download |
|---|---|---|
| `vqvae_traj.ckpt` | Trajectory VQVAE (Stage 1a) | [link pending] |
| `vqvae_hp.ckpt` | Hand-pose VQVAE (Stage 1b) | [link pending] |
| `llm_pretrain.ckpt` | LLM pretrained (Stage 2) | [link pending] |
| `llm_align.ckpt` | LLM after alignment (Stage 3) | [link pending] |
| `llm_instruct.ckpt` | LLM after instruction tuning (Stage 4) | [link pending] |
| `tm2t_evaluator.ckpt` | TM2T evaluation model | [link pending] |

---

## Release Notes

- The configs in `configs/` are kept as training templates and still use placeholder paths.
- The repo-local release status is tracked in [RELEASE_CHECKLIST.md](./RELEASE_CHECKLIST.md).
- The current release asset bundle includes only the normalization file kept for prompt
  inference. Other dataset-derived training/evaluation files are expected from the
  separate dataset release.
- The current prompt-demo path writes `.npy` and `.obj` reliably. Video rendering still depends
  on a host with a working OpenGL backend for `pyrender`.

---

## Repository Structure

```
CLUTCH/
├── train.py                   # Training entry point (all stages)
├── extract_codes.py           # Extract motion tokens (between Stage 1 and 2)
├── release_checkpoints/       # Staged VQ-VAE and LLM checkpoints
├── environment.yml            # Conda environment
├── RELEASE_CHECKLIST.md       # Remaining release blockers
├── configs/
│   ├── stage1a_vqvae_traj.yaml     # Stage 1a: Trajectory VQVAE
│   ├── stage1b_vqvae_hp.yaml       # Stage 1b: Hand-pose VQVAE
│   ├── stage1c_extract_tokens.yaml # Stage 1c: Token extraction
│   ├── stage2_llm_pretrain.yaml    # Stage 2: LLM pretraining
│   ├── stage3_llm_align.yaml       # Stage 3: Geometry alignment
│   └── stage4_llm_instruct.yaml    # Stage 4: Instruction tuning
├── test/
│   ├── test_t2m.py            # Text-to-Motion evaluation
│   └── test_m2t.py            # Motion-to-Text evaluation
├── mGPT/                      # Core model module
│   ├── archs/                 # Architecture definitions (VQVAE, LM)
│   ├── models/                # PyTorch Lightning model wrappers
│   ├── data/                  # Dataset classes (EgoVid5M, GRAB)
│   ├── losses/                # Loss functions
│   ├── metrics/               # Evaluation metrics
│   └── utils/                 # Utilities (logging, checkpointing)
└── assets/                    # Runtime assets and release-local EgoVid5M metadata
```

---

## Citation

```bibtex
@inproceedings{clutch2026,
  title={{CLUTCH}: Contextualized Language model for Unlocking Text-Conditioned Hand motion modelling in the wild},
  author={Balamurugan Thambiraja and Omid Taheri and Radek Danecek and Giorgio Becherini and Gerard Pons-Moll and Justus Thies},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=W7YRskO47j}
}
```

---

## Acknowledgements

This codebase builds upon [MotionGPT](https://github.com/OpenMotionLab/MotionGPT), [MoMask](https://github.com/EricGuo5513/momask-codes), and the [GRAB dataset](https://grab.is.tue.mpg.de/).
