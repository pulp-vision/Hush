# Manifest Format

JSON manifest files tell `prepare_hdf5.sh` which audio files to pack into HDF5 shards. Each manifest has a single `samples` array.

## Required Files

| Filename | HDF5 group | Purpose |
|---|---|---|
| `train.json` | `speech` | Primary speaker recordings (clean target) |
| `val.json` | `speech` | Validation speech (speaker-disjoint from train) |
| `noise.json` | `noise` | Environmental/background noise clips |
| `rir.json` | `rir` | Room impulse responses |
| `background.json` | `speech` | Background/interferer speakers (speaker-disjoint from train) |

`background.json` is optional. If absent, the script skips background speech packing.

## Sample Schema

### Speech manifests (`train.json`, `val.json`, `background.json`)

```json
{
  "samples": [
    {
      "path": "datasets/LibriSpeech/train-clean-100/1594/135914/1594-135914-0000.flac",
      "speaker_id": "libri_1594",
      "duration": 16.105,
      "dataset": "librispeech"
    }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `path` | yes | Path relative to `DATA_ROOT` (the `--working-dir` passed to `hdf5_prep`) |
| `speaker_id` | no | Speaker identifier (used for speaker-disjoint splitting) |
| `duration` | no | Duration in seconds (informational only) |
| `dataset` | no | Source dataset name (informational only) |

### Noise manifest (`noise.json`)

```json
{
  "samples": [
    {
      "path": "datasets/dns4/datasets_fullband/noise_fullband/1GS26LFpoJg.wav",
      "duration": 10.0,
      "category": "noise_fullband"
    }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `path` | yes | Path relative to `DATA_ROOT` |
| `duration` | no | Duration in seconds |
| `category` | no | Source category label |

### RIR manifest (`rir.json`)

```json
{
  "samples": [
    {
      "path": "datasets/dns4/datasets_fullband/impulse_responses/SLR26/simulated_rirs_48k/smallroom/Room093/Room093-00001.wav",
      "duration": 1.0
    }
  ]
}
```

| Field | Required | Description |
|---|---|---|
| `path` | yes | Path relative to `DATA_ROOT` |
| `duration` | no | Duration in seconds |

## Path Resolution

All `path` values are resolved relative to `DATA_ROOT`, which defaults to `../..` from the repo root and can be overridden with:

```bash
MANIFEST_DIR=/path/to/manifests DATA_ROOT=/path/to/datasets ./data/scripts/prepare_hdf5.sh
```

## Audio Requirements

All audio files must be pre-resampled to **16 kHz mono** before packing. The `hdf5_prep` binary will error on files with a different sample rate.

Use `sox` for batch resampling:

```bash
for f in /path/to/raw/*.wav; do
    sox "$f" -r 16000 -c 1 /path/to/resampled/$(basename "$f")
done
```

## Example Files

- `example_train.json` — speech manifest format
- `example_noise.json` — noise manifest format
- `example_rir.json` — RIR manifest format
- `example_background.json` — background speech manifest format
