# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Audio stem separation using a U-Net model trained on the MUSDB18 dataset. The model takes a mixture audio spectrogram as input and outputs an isolated vocals spectrogram via a learned soft mask.

## Environment Setup

```bash
conda env create -f env/environment.yml
conda activate hf-audio-stem
```

Requires CUDA 12.4 and the MUSDB18 dataset (placed in `data/dataset/`, in `.stem` format — `is_wav=False`).

## Running Training

```bash
# From the repo root
python ai_model/training.py
```

Saved model checkpoints go to `ai_model/unet_models/meilleur_unet_vocal.pth`.

## Architecture

**Data pipeline (`ai_model/data_loader.py`)**
- Loads MUSDB18 tracks, extracts random 3-second chunks
- Converts stereo to mono, computes STFT (n_fft=2048, hop_length=512)
- Returns magnitude spectrograms cropped to `[1, 1024, 256]` (freq × time)
- Input `X` = full mixture magnitude; target `y` = vocals magnitude

**Model (`ai_model/u_net.py` — `UnetAudioStemmer`)**
- Encoder: 3× `DownConvBlock` (Conv2d stride=2, BN, LeakyReLU 0.2), channels: 1→64→128→256
- Bottleneck: Conv2d stride=2, 256→512, BN, ReLU
- Decoder: 3× `UpConvBlock` (ConvTranspose2d + skip concat + Conv2d, BN, ReLU), channels: 512→256→128→64
- Final upsample (ConvTranspose2d 64→64) + `Conv2d(64,1,1)` + Sigmoid → soft mask
- Output = `input × mask` (masking in the spectrogram domain)

**Utilities (`tool_box/`)**
- `converter.py:audio_to_spectrogram_db` — stereo→mono, STFT, returns `(magnitude, phase, magnitude_db)`
- `pad_or_crop.py` — crops/pads tensors to a multiple of a given number (currently unused in training)

**Training (`ai_model/training.py`)**
- Loss: L1Loss on magnitude spectrograms
- Optimizer: Adam lr=0.001, batch size=16, up to 1000 epochs
- Best val-loss model is checkpointed automatically

## Key Constraints

- Spectrogram dimensions are hard-cropped to `[:, :1024, :256]` in the dataloader to ensure U-Net compatibility (must be multiples of 16 given 4 stride-2 down-samples).
- `musdb.DB` is loaded with `is_wav=False` (native `.stem.mp4` format).
- `sys.path` manipulation in `data_loader.py` and `training.py` assumes scripts are run from the repo root or `ai_model/` directory respectively — prefer running from repo root.
