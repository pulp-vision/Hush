# Model Architecture

DeepFilterNet-SE is a real-time speech enhancement model based on [DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet), adapted for **16 kHz** with an additional **Separation Head** for background speaker suppression.

## Signal Flow

```
Input Waveform  [B, 1, T_samples]
       |
       v
  STFT (FFT=320, Hop=160, Hann window)   --> 10 ms frame rate, zero lookahead
       |
       v
Spectrum  [B, 1, T_frames, 161, 2]       --> 161 complex frequency bins
       |
  _____|_______________________________
  |                |                  |
  v                v                  |
ERB Features    DF Features        Spectrum
[B,1,T,32]     [B,2,T,64]         (stored for
                                    output)
  |                |
  '-------+--------'
          |
          v
    +-----------+
    |  ENCODER  |
    |           |
    | ERB branch:                         |
    |   Conv2dNormAct(1->16, k=3x3)       |
    |   Conv2dNormAct(16->16, k=1x3, s=2) | --> e0,e1
    |   Conv2dNormAct(16->16, k=1x3, s=2) | --> e2
    |   Conv2dNormAct(16->16, k=1x3)      | --> e3
    |                                     |
    | DF branch:                          |
    |   Conv2dNormAct(2->16, k=3x3)       | --> c0
    |   Conv2dNormAct(16->16, k=1x3, s=2) |
    |                                     |
    | Fusion: Add + GroupedLinearEinsum   |
    |                                     |
    | SqueezedGRU_S(256-dim, 1 layer)     |
    |    -> embeddings [B, T, 128]        |
    |    -> lSNR head  [B, T, 1]          |
    +-----------+
          |
    ______|______________________________
    |             |                    |
    v             v                    v
+----------+ +-----------+  +--------------------+
| ERB DEC  | |  DF DEC   |  |  SEPARATION HEAD   |
|          | |           |  |  (novel addition)  |
| SqzdGRU  | | SqzdGRU   |  |                    |
| 256-dim  | | 256-dim   |  | Linear(256->32)    |
| 1 layer  | | 3 layers  |  | + Sigmoid          |
|          | |           |  |                    |
| ConvT x4 | | Tanh      |  | Output: ERB mask   |
| + skip   | | DF coefs  |  | for bg speaker     |
| conns.   | | [B,T,64,  |  | [B, 1, T, 32]      |
|          | |  5,2]     |  |                    |
| Sigmoid  | |           |  | Used for:          |
|          | |           |  | SeparationLoss     |
| ERB mask | |           |  | (training only)    |
| [B,1,T,  | |           |  +--------------------+
|    32]   | |           |
+----+-----+ +-----+-----+
     |              |
     v              v
 Apply ERB mask  Apply DF filter
 to spectrum     to low-freq bins
     |              |
     '------+--------'
            |
            v
     Enhanced Spectrum
            |
            v
          ISTFT
            |
            v
   Enhanced Waveform  [B, 1, T_samples]
```

## Component Details

### STFT / Feature Extraction
- **FFT size**: 320 samples → 161 frequency bins at 16 kHz
- **Hop size**: 160 samples → 10 ms frame rate
- **Window**: Hann window
- **ERB features**: 32 ERB-spaced bands, log-power, exponential unit normalization
- **DF features**: complex spectrogram of lowest 64 bins, exponential unit normalization

### Encoder
The encoder processes two parallel feature streams:

**ERB Branch** (coarse spectral features):
- 4 depthwise-separable Conv2d blocks with BatchNorm + ReLU
- Two stride-2 layers reduce the frequency dimension from 32 → 8

**DF Branch** (fine complex features):
- 2 depthwise-separable Conv2d blocks on the raw complex spectrogram
- One stride-2 layer

**Fusion**: Element-wise addition of projected ERB and DF features, followed by a `SqueezedGRU_S` (256-dim hidden, 1 layer) that produces frame-wise embeddings and local SNR estimates.

### ERB Decoder
Reconstructs the full 32-band ERB gain mask using:
- A second `SqueezedGRU_S` (256-dim, 1 layer)
- 4 skip-connected ConvTranspose2d blocks (reversing the encoder strides)
- Final Sigmoid activation → mask values in [0, 1]

The mask is projected back to 161 frequency bins via the inverse ERB filterbank and multiplied with the input spectrum.

### DF Decoder (Deep Filtering)
Applies a complex-valued FIR filter to the lowest 64 frequency bins:
- `SqueezedGRU_S` (256-dim, 3 layers) processes the encoder embeddings
- Output: order-5 complex filter coefficients per time-frequency bin
- Applied via overlap-add convolution → high spectral resolution in low frequencies

### Separation Head (Novel Contribution)
A lightweight auxiliary head attached to the encoder embeddings:
```
emb [B, T, 256]
    |
    v
Linear(256 -> 32)
    |
    v
Sigmoid
    |
    v
sep_mask [B, 1, T, 32]   <- ERB-domain background speaker mask
```

During training, this head is supervised with `SeparationLoss` (L1, factor=0.1) to predict an ERB mask for the background speaker. This forces the encoder to learn speaker-discriminative representations without adding any inference-time cost (the head can be disabled post-training).

### Normalization
`ExponentialUnitNorm` is applied to both ERB and DF input features. It computes a running exponential average of the signal energy (decay constant `tau=1.0s`) and normalizes the features by the square root of this running average. This provides stable, speaker-independent input normalization.

## Key Hyperparameters

| Parameter | Value | Notes |
|---|---|---|
| Sample rate | 16,000 Hz | |
| FFT size | 320 | 20 ms window |
| Hop size | 160 | 10 ms frame rate |
| ERB bands | 32 | min 2 bins/band |
| DF bins | 64 | ~2 kHz coverage |
| DF order | 5 | FIR filter taps |
| Conv channels | 16 | depthwise separable |
| Embedding dim | 256 | GRU hidden size |
| DF GRU layers | 3 | deeper for fine filtering |
| ERB GRU layers | 2 (enc+dec) | |
| Total params | ~1.8M | |
| Lookahead | 0 | fully causal / real-time |

## Architecture Diagram

See `docs/architecture.png` for a visual representation.
