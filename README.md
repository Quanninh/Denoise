# DeepFilterNet ERB Speech Denoising

This repository implements the **DeepFilterNet** in PyTorch, based on [`model.ipynb`](model.ipynb). It provides an end-to-end pipeline for speech enhancement and denoising, from acoustic feature extraction and sharded data caching to neural network training and spectral mask synthesis.

---

## Architecture Overview

The system operates in the frequency domain using a human auditory-inspired ERB filterbank to compress high-resolution STFT magnitudes into perceptually spaced frequency bands, estimate non-negative mask gains, and reconstruct enhanced complex spectra.

```
                  ┌──────────────────────┐
                  │ Noisy Audio (16 kHz) │
                  └──────────┬───────────┘
                             │ STFT (N_fft=512, Hop=128)
                             ▼
 ┌────────────────────── Complex STFT ─────────────────────┐
 │                       [B, T, 257]                       │
 └──────┬───────────────────────────────────────────┬──────┘
        │ Magnitude / Power                         │
        ▼                                           │
 ┌──────────────┐                                   │
 │ ERB Analysis ├──► Input ERB [B, 1, T, 32]        │
 └──────────────┘           │                       │
                            ▼                       │
               ┌─────────────────────────┐          │
               │    DeepFilterNetDNN     │          │
               │   (ERB Encoder-Decoder) │          │
               └────────────┬────────────┘          │
                            │ ERB Gains [B, 1, T, 32]
                            ▼                       │
               ┌─────────────────────────┐          │
               │      ERB Synthesis      │          │
               │   (Linear Band Weight)  │          │
               └────────────┬────────────┘          │
                            │ Frequency Gains [B, T, 257]
                            ▼                       │
               ┌─────────────────────────┐          │
               │     Spectral Masking    │◄─────────┘
               └────────────┬────────────┘
                            │
                            ▼
               Enhanced Complex Spectrum [B, T, 257]
```

### Key Modules

1. **Separable Convolutions & Grouped Layers**:
   - `SeparableConv2d` / `SeparableTConv2d`: Depthwise-separable convolutions with configurable causal lookahead padding for real-time, low-latency processing.
   - `GroupedLinear`: Multi-group linear projection with channel shuffling (`channel_shuffle`).
   - `GroupedGRU` & `GroupedGRUStack`: 3-layer grouped gated recurrent units providing efficient sequence modeling across temporal frames.

2. **`ERBEncoder`**:
   - Downsamples 32 ERB frequency bands across 4 convolutional stages with lookahead.
   - Flattened bottleneck features are projected via `GroupedLinear` into a recurrent embedding (`GroupedGRUStack`, hidden size 512, 8 groups).

3. **`ERBDecoder`**:
   - U-Net style decoder with skip connections (`PConv` 1x1 convs) from encoder stages.
   - Upsamples features back to the 32-band resolution using `SeparableTConv2d`.
   - Sigmoid output activation produces real-valued gain masks $G_{\text{erb}} \in [0, 1]$.

4. **`DeepFilterNetDNN`**:
   - Combines the encoder and decoder to map `input_erb` $[B, 1, T, E]$ directly to spectral gains $[B, 1, T, E]$.

---

## Audio Processing & Filterbank

- **Sample Rate**: 16,000 Hz
- **STFT Configuration**: Hann window, $N_{\text{fft}} = 512$, window length = 512, hop length = 128 (8 ms frame shift).
- **ERB Filterbank**:
  - `make_erb_filterbank`: Generates 32 triangular filterbank weights spanning from 0 Hz to Nyquist (8 kHz), converting linear FFT frequencies to human auditory ERB scale.
  - `erb_synthesis_matrix`: Computes normalized inverse ERB weights ($257 \times 32$).
  - `apply_erb_gains`: Multiplies predicted ERB gains with the synthesis matrix and applies the resulting full-resolution frequency gains $[B, T, 257]$ to the noisy complex STFT.

---

## Dataset Pipeline & Sharding

The notebook includes high-performance data preparation and loading designed for large-scale datasets (e.g., DNS Challenge, AEC Challenge):

- **Offline Shard Storage (`build_erb_store`)**:
  - Precomputes complex STFTs (`input_spec`, `target_spec`) and log-power ERB features (`input_erb`, `target_erb`).
  - Serializes features in batched chunk files (`shard_0000.pt`, 200 pairs per shard) alongside an `index.jsonl` manifest.
- **`ShardBatchSampler`**:
  - Groups sample batches strictly within the same physical `.pt` shard file, minimizing random disk seeks and enabling fast sequential loading.
- **`IndexedERBDataset`**:
  - Efficiently parses `index.jsonl` metadata and loads tensors on demand with support for temporal cropping or padding (`segment_frames=256`).

---

## Training & Loss Function

### Compressed Spectral Loss
Training optimizes a compressed, phase-aware spectral loss (`compressed_spectral_loss`) with power compression factor $\alpha = 0.6$:
$$\mathcal{L} = \left| |Y|^\alpha - |\hat{Y}|^\alpha \right|^2 + \left| |Y|^\alpha e^{j \angle Y} - |\hat{Y}|^\alpha e^{j \angle \hat{Y}} \right|^2$$
This balances magnitude restoration with complex phase alignment.

### Default Hyperparameters
- **Batch Size**: 8
- **Optimizer**: Adam ($\text{lr} = 10^{-3}$)
- **Epochs**: 25
- **Split**: 70% Train, 15% Validation, 15% Test
- **Checkpoints**: Saved to `erb_checkpoints/best_model.ckpt` on validation loss improvement.

---

## Quickstart

### 1. Minimal Model Usage

```python
import torch
from erb_training import make_erb_filterbank, erb_synthesis_matrix, apply_erb_gains

# 1. Setup filterbank and synthesis matrix
sample_rate = 16000
n_fft = 512
erb_bins = 32

filterbank = make_erb_filterbank(sample_rate, n_fft, erb_bins)     # [32, 257]
synthesis_matrix = erb_synthesis_matrix(filterbank)               # [257, 32]

# 2. Forward pass with ERB features
# model = DeepFilterNetDNN(erb_bins=erb_bins, channels=64, hidden_size=512, groups=8)
input_erb = torch.randn(1, 1, 100, erb_bins)                      # [B, 1, T, E]
noisy_stft = torch.randn(1, 100, n_fft // 2 + 1, dtype=torch.complex64)

# 3. Predict gains and apply to complex spectrum
# gains = model(input_erb)                                        # [1, 1, 100, 32]
# enhanced_stft = apply_erb_gains(noisy_stft, gains, synthesis_matrix)
```

### 2. Running the Notebook
Open [`model.ipynb`](model.ipynb) in Jupyter, Google Colab, or Kaggle:
1. Run feature extraction or point to synthesized dataset (`index.jsonl` and shards).
2. Train model with `train_model()`.
3. Inspect training/validation loss curves.
4. Evaluate test set spectrogram similarity with `spectrogram_similarity()`.

