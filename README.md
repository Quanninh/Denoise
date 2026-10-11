# DualFilter: Two-Stage Perceptual Speech Enhancement

**DualFilterNet** is a low-latency, two-stage neural speech enhancement framework implemented in PyTorch, designed for real-time speech denoising and dereverberation. 

> [!NOTE]
> **Reference & Foundational Idea**:
> DualFilter is based on and inspired by the architectural principles and loss formulations introduced in **DeepFilterNet** (*H. Schröter et al., "DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio using Deep Filtering", Interspeech 2022*).

The system divides the challenging speech enhancement task into two decoupled, computationally efficient stages:
1. **Stage 1 (ERB Filtering):** Coarse spectral envelope attenuation via human auditory Equivalent Rectangular Bandwidth (ERB) filterbanks.
2. **Stage 2 (Deep Filtering):** Fine-grained complex linear prediction filtering on low-frequency speech harmonics to restore phase and suppress inter-harmonic noise.

---

## Table of Contents
- [System Architecture Overview](#system-architecture-overview)
- [Stage 1 Architecture (ERB Envelope Filtering)](#stage-1-architecture-erb-envelope-filtering)
- [Full Model Architecture (Stage 1 + Stage 2 Deep Filter)](#full-model-architecture-stage-1--stage-2-deep-filter)
- [Dataset Generator (`DynamicMixingDataset`)](#dataset-generator-dynamicmixingdataset)
- [Data Sources & Provenance](#data-sources--provenance)
- [Loss Functions & Training Objectives](#loss-functions--training-objectives)
- [Numerical Stability & Mixed Precision (AMP)](#numerical-stability--mixed-precision-amp)
- [Project Structure & Quickstart](#project-structure--quickstart)
- [References](#references)

---

## System Architecture Overview

```
                        ┌──────────────────────────────┐
                        │   Noisy Input Waveform x(t)  │
                        └──────────────┬───────────────┘
                                       │ STFT (N_fft=512, Hop=128, 16 kHz)
                                       ▼
                     ┌───────────────────────────────────┐
                     │ Complex Spectrogram X ∈ C[B,T,257]│
                     └─────────┬───────────────────┬─────┘
                               │                   │
                 Magnitude /   │                   │ Real & Imag Low Freqs
                 Log Power     │                   │ F_df = 96 bins (~3 kHz)
                               ▼                   ▼
                     ┌──────────────────┐  ┌──────────────────────┐
                     │  ERB Filterbank  │  │ CausalEmaUnitNorm    │
                     │  (32 Bands)      │  └──────────┬───────────┘
                     └─────────┬────────┘             │
                               │ CausalEmaNorm        │
                               ▼                      ▼
                     ┌──────────────────┐  ┌──────────────────────┐
                     │    ERBEncoder    │  │        DFNet         │
                     │  (SeparableConv, ├─►│ (Complex Conv, GRU,  │
                     │   GroupedGRU)    │  │  Embedding Merge)    │
                     └─────────┬────────┘  └──────────┬───────────┘
                               │                      │
             Encoder Skip      │ GRU Embedding        │
             Connections       │                      │
                               ▼                      │
                     ┌──────────────────┐             │
                     │    ERBDecoder    │             │
                     │  (SeparableTConv,│             │
                     │   PConv, Sigmoid)│             │
                     └─────────┬────────┘             │
                               │                      │
                  ERB Gains    │                      │ Complex Filter Coeffs c
                  G_erb        ▼                      │ Mixing Alpha α
                     ┌──────────────────┐             │
                     │  Synthesis Matrix│             │
                     │  M ∈ R[257, 32]  │             │
                     └─────────┬────────┘             │
                               │                      │
             Full-Band Gain    ▼                      │
             G = G_erb · M^T ┌──────────────────┐     │
                             │ Spectral Masking │     │
                             │ Y^G = X ⊙ G      │     │
                             └─────────┬────────┘     │
                                       │              │
                    Stage 1 Enhanced   ├──────────────┘
                    Spectrogram Y^G    │
                                       ▼
                             ┌──────────────────┐
                             │ Apply Deep Filter│
                             │ (Order N=5, α)   │
                             └─────────┬────────┘
                                       │
                                       ▼
                     ┌───────────────────────────────────┐
                     │ Final Enhanced Spectrogram Y^DF   │
                     └─────────────────┬─────────────────┘
                                       │ iSTFT (Synthesis Window)
                                       ▼
                        ┌──────────────────────────────┐
                        │ Enhanced Output Waveform y(t)│
                        └──────────────────────────────┘
```

---

## Stage 1 Architecture (ERB Envelope Filtering)

Stage 1 estimates a smooth, real-valued spectral gain envelope to attenuate broadband background noise with low computational complexity.

### 1. Acoustic Representations
- **STFT Configuration**: Sampling rate $f_s = 16{,}000\text{ Hz}$, $N_{\text{fft}} = 512$, window length $= 512$ (Hann), hop size $= 128$ ($8\text{ ms}$ frame step, $50\%\times 2$ overlap). Yields $F = 257$ single-sided frequency bins.
- **ERB Filterbank Analysis**: Converts linear FFT bins into 32 triangular frequency bands spaced according to the human auditory Equivalent Rectangular Bandwidth (ERB) scale.
  $$\text{ERB}(f) = 24.7 \cdot \left(4.37 \cdot \frac{f}{1000} + 1\right)$$
  The filterbank matrix $W_{\text{erb}} \in \mathbb{R}^{32 \times 257}$ aggregates power spectrum $|X|^2$:
  $$P_{\text{erb}} = |X|^2 W_{\text{erb}}^T \in \mathbb{R}^{B \times T \times 32}$$
  Features are log-scaled: $X_{\text{erb}} = \log_{10}(P_{\text{erb}} + 10^{-7})$.

### 2. Causal Exponential Moving Average Normalization (`CausalEmaNorm`)
To ensure strict real-time causality without leaking future frames, features are normalized using an online exponential moving average:
$$\mu_t = \alpha \mu_{t-1} + (1 - \alpha) x_t$$
$$\sigma_t^2 = \left( \alpha \sigma_{t-1}^2 + (1 - \alpha)(x_t - \mu_{t-1})(x_t - \mu_t) \right)^+$$
$$\hat{x}_t = \left(\frac{x_t - \mu_t}{\sqrt{\sigma_t^2 + \epsilon}}\right) \cdot \gamma + \beta$$
- **Decay factor**: $\alpha = 0.99$ (~1 second temporal memory at $8\text{ ms}$ hop).
- **Affine scaling**: Initialized with $\sigma_0^2 = 1.0$ (avoiding division-by-zero on initial frames) and learnable scale $\gamma = 0.1$, preventing large activations from saturating recurrent gates.

### 3. Neural Architecture
- **`ERBEncoder`**:
  - 4 convolutional stages of `SeparableConv2d` (depthwise separable $3\times 3$ convs with causal time padding and lookahead of 2 frames).
  - Frequency strides downsample bands: $32 \to 16 \to 8 \to 4$.
  - Bottleneck is flattened and projected via `GroupedLinear` ($8$ groups) into a 3-layer `GroupedGRUStack` (hidden dimension $512$, $8$ groups, grouped hidden-to-hidden matrices).
- **`ERBDecoder`**:
  - Symmetrical U-Net architecture with skip connections from the encoder via $1\times 1$ pointwise convs (`PConv`).
  - Upsamples frequency resolution using transposed separable convolutions (`SeparableTConv2d`).
  - Final projection with `Sigmoid` output yields non-negative envelope gains:
    $$G_{\text{erb}} \in [0, 1]^{B \times 1 \times T \times 32}$$

### 4. ERB Synthesis & Masking
An inverse synthesis matrix $M \in \mathbb{R}^{257 \times 32}$ linearly interpolates 32 ERB gains back to 257 FFT bins:
$$M = W_{\text{erb}}^T \oslash \left(\sum W_{\text{erb}}^T\right)$$
$$Y^G(t, f) = X(t, f) \cdot \left(G_{\text{erb}}(t) \cdot M^T\right)_f$$

---

## Full Model Architecture (Stage 1 + Stage 2 Deep Filter)

While Stage 1 cleans up broadband spectral envelopes, it cannot adjust phase or recover fine periodic harmonics in low frequencies. Stage 2 applies a multi-frame complex linear filter to the lower frequencies ($0\text{ Hz}$ to $\sim 3{,}000\text{ Hz}$, $F_{\text{df}} = 96$ bins).

### 1. Complex Feature Processing
- Input features are the real and imaginary parts of the noisy STFT: $[B, 2, T, F_{\text{df}}]$.
- Normalized using `CausalEmaUnitNorm`, dividing complex frames by a smoothed magnitude while preserving phase angle:
  $$x_{\text{unit}} = \frac{x_t}{\text{EMA}(|x_t|) + \epsilon}$$

### 2. `DFNet` Architecture
- **Complex Convolutions**: Two separable convolutional layers with lookahead process the normalized complex features.
- **Frequency Projection & Recurrence**: Grouped linear projection and a 2-layer `GroupedGRUStack` ($512$ hidden size, $8$ groups).
- **Embedding Fusion**: The temporal representation from `DFNet` is concatenated with the deep recurrent embedding from `ERBEncoder` and merged via a linear layer.
- **Outputs**:
  1. **Complex Filter Coefficients**: $c \in \mathbb{C}^{B \times T \times F_{\text{df}} \times N}$ ($N=5$ taps, real and imaginary channels bounded by `Tanh`).
  2. **Adaptive Shrinkage Factor**: $\alpha \in [0, 1]^{B \times T \times 1}$ via `Sigmoid`, dynamically controlling filtering strength.

### 3. Deep Filtering Mechanism (`apply_deep_filter`)
The filter operates on the Stage 1 output $Y^G$, applying causal linear prediction across $N=5$ past and current frames:
$$Y^{\text{DF}0}(t, f) = \sum_{n=0}^{N-1} c_n^*(t, f) \cdot Y^G(t - n + \text{lookahead}, f)$$

The filtered result is blended with the Stage 1 envelope using the predicted shrinkage parameter $\alpha$:
$$Y^{\text{DF}}(t, f) = \alpha(t) \cdot Y^{\text{DF}0}(t, f) + (1 - \alpha(t)) \cdot Y^G(t, f)$$

For high frequencies ($f \ge F_{\text{df}}$), where harmonics are weak and filter estimation is noisy, the Stage 1 output $Y^G(t, f)$ passes through unchanged.

---

## Dataset Generator (`DynamicMixingDataset`)

To avoid training on static, memorized mixtures, `DynamicMixingDataset` dynamically synthesizes realistic, augmented noisy reverberant audio pairs on the fly for every sample.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Clean Speech │     │ Noise Pool   │     │  RIR Corpus  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │ (1-4 Files)        │ (50% Prob)
       │ Random Crop        │ Random Crop        │
       │ Resample 16 kHz    │ Resample 16 kHz    │
       ▼                    ▼                    ▼
   s_clean(t)           n_mix(t)              rir(t)
       │                    │                    │
       ├────────────────────┼────────────────────┤
       ▼                    │                    │
┌─────────────────────────┐ │                    │
│   RIR Split Analysis    │◄─────────────────────┘
│ - T_0 = 50ms early gate │
│ - Exponential decay Rd  │
└────────────┬────────────┘
             │
     ┌───────┴─────────────────────────┐
     ▼ Full Reverb                     ▼ Early Target Only
  s_rev(t) = s * rir               s_target(t) = s * rir_early
     │                                 │
     ▼ Normalize dBFS                  ▼
┌───────────────────────────────┐      │
│ SNR Mixing & Headroom Scaling │      │
│ - Target SNR ∈ {-5..20 dB}    │      │
│ - Final Target dBFS ∈ [-35..-15]     │
│ - Peak Anti-Clipping Rescaling│──────┼─────────────┐
└──────────────┬────────────────┘      │ Rescale     │
               ▼                       ▼             ▼
          noisy_wav               clean_target   (STFT / ERB)
```

### 1. On-The-Fly Mixing Pipeline
1. **Dynamic Audio Slicing**:
   - Randomly samples clean speech and cuts an exact window ($48{,}000\text{ samples} = 3.0\text{ s}$ or $160{,}000\text{ samples} = 10.0\text{ s}$).
   - Automatically handles multi-channel audio by downmixing to mono and resampling to $16\text{ kHz}$.
2. **Dense Multi-Source Noise Mixing**:
   - Instead of single-source noise, each sample randomly loads between $1$ and $4$ distinct noise recordings, uniformly summed to simulate realistic complex acoustic environments (e.g., street traffic + cafe babble + HVAC hum).
3. **Room Acoustic Simulation & Dereverberation**:
   - With a $50\%$ probability, a Room Impulse Response (RIR) is drawn.
   - **Noisy Input**: The clean speech is convolved with the full RIR: $s_{\text{rev}} = s_{\text{clean}} * \text{RIR}$.
   - **Target Audio**: The model is trained to remove late reverberation while retaining natural early reflections ($< 50\text{ ms}$). The target RIR is windowed with an exponential/cosine decay curve:
     $$w_{\text{early}}(t) = \begin{cases} 1, & t \le T_0 \\ 10^{-3(t - T_0)/R_d}, & t > T_0 \end{cases}$$
     where $T_0 = 50\text{ ms}$ and $R_d = 200\text{ ms}$. Clean speech is convolved with this shortened RIR, yielding a clean, dry target.
4. **Energy Normalization & SNR Scaling**:
   - Both speech and noise are normalized to a calibrated $-25\text{ dBFS}$ root-mean-square level.
   - The noise mixture is scaled to a randomly selected SNR:
     $$\text{SNR} \in \{-5\text{ dB}, 0\text{ dB}, 5\text{ dB}, 10\text{ dB}, 15\text{ dB}, 20\text{ dB}\}$$
   - Random microphone gain is simulated by shifting overall volume within $\pm 10\text{ dB}$ (target dBFS in $[-35, -15]$).
   - **Headroom Protection**: If the mixed signal peak exceeds $0.99$, both noisy input and clean target are synchronously scaled down to avoid digital clipping.
5. **Deterministic Validation/Test Sampling**:
   - For validation and test loaders, `flag=True` fixes pseudo-random seeds (`42 + idx`), guaranteeing identical test conditions across evaluation epochs.

---

## Data Sources & Provenance

The pipeline utilizes open audio corpora from international acoustic challenges:

| Data Modality | Original Source & Corpus | Description | Local / Kaggle Identifier |
| :--- | :--- | :--- | :--- |
| **Clean Speech** | **Microsoft DNS Challenge 4** (`read_speech`, LibriSpeech, VCTK, VoxCeleb) | High-quality clean speech recordings without background noise across thousands of speakers and acoustic environments. | `quanninhhoang/dns-cleanspeech-dataset` |
| **Noise Pool** | **Microsoft DNS Challenge 4** (`noise_fullband`, AudioSet, Freesound, DEMAND) | Diverse real-world non-stationary noises (domestic appliances, office chatter, vehicular noise, babble, nature, urban). | `quanninhhoang/dns-noise-dataset` |
| **Room Impulse Responses** | **DNS Challenge RIR Corpus** (Aachen AIR, BUT Reverb, OpenSLR28) | Real measured impulse responses from rooms, halls, and offices alongside synthetic image-source room simulations ($RT_{60} \in [0.1\text{ s}, 1.0\text{ s}]$). | `quanninhhoang/dns-impulse-dataset` |
| **Evaluation Benchmark** | **VoiceBank + DEMAND** (`clean_testset_wav`, `noisy_testset_wav`) | Standardized speech enhancement benchmark for reporting objective metrics (PESQ, CSIG, CBAK, COVL, STOI, SI-SDR). | `quanninhhoang/voicebank-demand` |

---

## Loss Functions & Training Objectives

### 1. Stage 1 Objective: Mask Loss (`ERBMaskLoss`)
Stage 1 exclusively predicts non-negative gains and is supervised with the Ideal Amplitude Mask (IAM):
$$\mathcal{M}_{\text{IAM}} = \min\left(1.0, \sqrt{\frac{|S|^2 W_{\text{erb}}^T}{|X|^2 W_{\text{erb}}^T + \epsilon}}\right)$$

Compressed power difference ($\gamma = 0.6$):
$$e = G_{\text{pred}}^\gamma - \mathcal{M}_{\text{IAM}}^\gamma$$

Asymmetric over-suppression and polynomial penalties ($L_2 + L_4$):
$$\mathcal{L}_{\text{mask}} = \mathbb{E}\left[ \left(e^2 + 10 \cdot e^4\right) \cdot \mathbf{1}_{G_{\text{pred}} < \mathcal{M}} \cdot f_{\text{under}} \right]$$
where $f_{\text{under}} = 2.0$ penalizes speech over-attenuation twice as heavily as remaining noise.

> [!NOTE]
> Stage 1 should **not** be supervised with complex phase losses ($\text{factor\_complex} = 0.0$), because real gains cannot adjust phase. Punishing Stage 1 for phase errors causes model collapse where gains saturate to $1.0$.

### 2. Full Model Objective: Decoupled Multi-Task Loss
When training the full model (Stage 1 + Stage 2), the loss separates envelope learning from phase reconstruction:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{mask}} \mathcal{L}_{\text{mask}}(G_{\text{erb}}, \mathcal{M}) + \lambda_{\text{spec}} \mathcal{L}_{\text{spectral}}(Y^{\text{DF}}, S)$$
- $\mathcal{L}_{\text{mask}}$ grades Stage 1 directly in the ERB domain.
- $\mathcal{L}_{\text{spectral}}$ grades the final complex output of Stage 2 ($Y^{\text{DF}}$) using both magnitude and phase:
  $$\mathcal{L}_{\text{spectral}} = \left\| |Y^{\text{DF}}|^\gamma - |S|^\gamma \right\|_2^2 + \left\| |Y^{\text{DF}}|^\gamma e^{j \angle Y^{\text{DF}}} - |S|^\gamma e^{j \angle S} \right\|_2^2$$

---

## Numerical Stability & Mixed Precision (AMP)

Training deep recurrent networks with fractional power spectral losses requires specific safeguards against gradient instability:

1. **Float16 & Complex64 Boundary in AMP**:
   Convolutions and GRUs execute in `float16` under `torch.amp.autocast`. Complex spectral math and loss calculations must execute in full `float32` / `complex64` to prevent `.square()` energy overflow beyond $65{,}504$.
2. **Epsilon Clamping Before Fractional Powers**:
   Derivatives of $x^\gamma$ ($\gamma = 0.6$) diverge as $x \to 0$ ($\frac{d}{dx} x^{0.6} = 0.6 x^{-0.4}$). All inputs to `.pow(0.6)` are clamped using `.clamp_min(1e-3)` or `.clamp_min(1e-12)` without energy denominators to keep backward gradients bounded.
3. **Causal Normalizer Variance Bounds**:
   In `CausalEmaNorm`, variance is clamped with `.clamp_min(0.0)` to eliminate floating-point negative artifacts from Welford updates, and initialized to $\sigma_0^2 = 1.0$ to prevent explosive scaling on early frames.
4. **GradScaler Lifecycle**:
   Gradients are unscaled with `scaler.unscale_(optimizer)` prior to norm clipping (`torch.nn.utils.clip_grad_norm_(..., max_norm=1.0)`). If non-finite gradients occur, `scaler.step()` safely skips parameter updates without corrupting weights.

---

## Project Structure & Quickstart

```
.
├── model.ipynb                 # Full model (Stage 1 + Stage 2) implementation
├── stage1.ipynb                # Stage 1 (ERB Envelope Filtering) isolated pipeline
├── stage1.py                   # Reusable DynamicMixingDataset & audio utilities
├── VoiceBank_Demand_Eval.xlsx  # Benchmark results on VoiceBank+DEMAND test set
└── README.md                   # System documentation
```

### Minimal Training Example (Stage 1)

```python
import torch
from stage1 import DynamicMixingDataset
from torch.utils.data import DataLoader

# 1. Initialize Filterbank & Model
filterbank = make_erb_filterbank(sr=16000, n_fft=512, n_erb=32).cuda()
synthesis_matrix = erb_synthesis_matrix(filterbank).cuda()
model = DualFilterDNN(erb_bins=32, channels=64, hidden_size=512).cuda()

# 2. Dataset & Loss Setup
train_dataset = DynamicMixingDataset(
    clean_dir="data/clean/train",
    noise_dir="data/noise/train",
    rir_dir="data/rir/train",
    filterbank=filterbank
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
criterion = CombinedLoss(filterbank, lambda_mask=1.0, lambda_stage1=0.0).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler('cuda')

# 3. Training Loop
model.train()
for batch in train_loader:
    noisy_wav = batch["noisy_wav"].cuda()
    clean_target = batch["clean_target"].cuda()
    
    # Compute STFT & ERB Features
    input_spec = torch.stft(noisy_wav, 512, 128, 512, return_complex=True).transpose(1, 2)
    target_spec = torch.stft(clean_target, 512, 128, 512, return_complex=True).transpose(1, 2)
    input_erb = torch.matmul(input_spec.abs().square(), filterbank.T)
    input_erb = torch.log10(input_erb + 1e-7).unsqueeze(1)

    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast('cuda'):
        gains = model(input_erb)
        stage1_spec = apply_erb_gains(input_spec, gains, synthesis_matrix)
    
    loss = criterion(
        stage1_spec=stage1_spec.to(torch.complex64),
        target_spec=target_spec.to(torch.complex64),
        input_spec=input_spec.to(torch.complex64),
        pred_gains=gains.to(torch.float32),
    )
    
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
```

---

## References

1. **DeepFilterNet (Foundational Idea & Architecture)**:
   * Hendrik Schröter, Alberto N. Escalante-B., Tobias Rosenkranz, Andreas Maier. *"DeepFilterNet: A Low Complexity Speech Enhancement Framework for Full-Band Audio using Deep Filtering"*, Interspeech 2022.
   * Hendrik Schröter, Tobias Rosenkranz, Alberto N. Escalante-B., Pascal Kirchhofer, Andreas Maier. *"DeepFilterNet2: Towards Real-Time Speech Enhancement on Embedded Devices for Full-Band Audio"*, IEEE International Workshop on Acoustic Signal Enhancement (IWAENC) 2022.
2. **Microsoft DNS Challenge**:
   * Chandan K. A. Reddy et al. *"The INTERSPEECH 2020 Deep Noise Suppression Challenge: Datasets, Subjective Testing Framework, and Challenge Results"*, Interspeech 2020.
