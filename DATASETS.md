# Datasets

The model was trained on standard speech enhancement benchmark datasets. The HDF5 shards in this project are derived from the following sources. You will need to download and prepare your own data using the instructions in [data/prepare_hdf5.md](data/prepare_hdf5.md).

## Primary Speech — `train_speech_*.hdf5`, `val_speech.hdf5`

These contain the clean speech targets (primary speaker).

| Dataset | URL | Notes |
|---|---|---|
| LibriSpeech train-clean-100 | https://www.openslr.org/12 | 100h read English speech |
| LibriSpeech train-clean-360 | https://www.openslr.org/12 | 360h read English speech |
| VCTK Corpus | https://datashare.ed.ac.uk/handle/10283/3443 | 110 speakers, British English |
| Common Voice (English) | https://commonvoice.mozilla.org | Crowd-sourced, diverse accents |

Validation set: held-out speakers from the above, not seen during training.

## Background Speech — `background_speech_*.hdf5`

The background speaker pool. These are human voices used as interferers. **Speakers in this set must be fully disjoint from the primary speech set.**

| Dataset | URL | Notes |
|---|---|---|
| LibriSpeech (disjoint split) | https://www.openslr.org/12 | Use a separate speaker split |
| VCTK (disjoint split) | https://datashare.ed.ac.uk/handle/10283/3443 | Separate speaker split |
| LibriTTS | https://www.openslr.org/60 | Higher quality, more speakers |

During training, background speakers are mixed at a **Signal-to-Interference Ratio (SIR)** uniformly sampled from `{24, 20, 18, 16, 14, 12}` dB below the primary speaker. This means the background is consistently quieter — a realistic scenario for voice calls.

## Noise — `noise_*.hdf5`

Environmental and background noise (non-speech).

| Dataset | URL | Notes |
|---|---|---|
| DNS Challenge Noise | https://github.com/microsoft/DNS-Challenge | Large-scale, diverse noise types |
| FreeSound | https://freesound.org | Community-licensed audio |
| ESC-50 | https://github.com/karolpiczak/ESC-50 | 50 environmental sound classes |
| AudioSet | https://research.google.com/audioset | YouTube-derived audio events |

SNR levels used during mixing: `{-100, -5, 0, 5, 10, 20, 40}` dB (randomly sampled per sample). SNR=-100 effectively means noise-only, forcing the model to handle severely degraded speech.

## Room Impulse Responses — `rir_*.hdf5`

Used for simulated reverberation (applied to 10% of training samples).

| Dataset | URL | Notes |
|---|---|---|
| MIT IR Survey | https://mcdermottlab.mit.edu/Reverb/IR_Survey.html | Real-room measurements |
| OpenAIR | https://www.openair.hosted.york.ac.uk | Acoustic impulse responses |
| BUT ReverbDB | https://speech.fit.vutbr.cz/software/but-speech-fit-reverb-database | FEKT Brno rooms |
| Simulated RIRs | (generated with `pyroomacoustics`) | Augment with synthetic RIRs |

## Data Mix Summary

| Category | Shards | Weight | Approx. Hours |
|---|---|---|---|
| Train speech | 24 | 10 | ~400h |
| Background speech | 24 | 1 | ~100h |
| Noise | 24 | 10 | ~200h |
| RIRs | 24 | 1 | N/A |
| Val speech | 1 | 10 | ~10h |

The sampling weights control how often each shard type is drawn. Higher weight = more frequent sampling. Speech and noise have weight 10 to ensure they dominate each batch, while background and RIRs (used for augmentation) have weight 1.

## Licensing Note

Each dataset has its own license. Please review the terms before use:
- LibriSpeech: CC BY 4.0
- VCTK: CC BY 4.0
- Common Voice: CC0 / CC BY 4.0
- DNS Challenge: Microsoft Research License
- ESC-50: CC BY-NC 3.0

This repository does not redistribute any raw audio data.
