# Face Recognition Pipeline

A face verification and identification pipeline built on a pretrained OpenFace
`nn4.small2` embedding network (Inception-style CNN, triplet-loss trained, 128-d
L2-normalized output), evaluated on the LFW (Labeled Faces in the Wild) dataset.

## Contents

| File / folder | Purpose |
|---|---|
| `face_recognition_pipeline.ipynb` | Main notebook — build the network, load weights, run verification/identification demos, measure accuracy |
| `facenet_arch.py` | Network architecture definition |
| `facenet_utils.py` | Weight loading (CSV → Keras) and image-encoding helpers |
| `weights/` | Pretrained CSV weight export (226 files, required) |
| `archive/` | LFW dataset, `lfw-deepfunneled/` images plus `pairs.csv` and related metadata (required) |
| `images/` | A handful of original assignment portraits (gitignored, not used by the notebook) |
| `old/` | Earlier, superseded version of the notebook/utils (gitignored) |

## Requirements

- **Python 3.11** (tested on 3.11.15)
- **TensorFlow 2.21.0** (brings in Keras 3.x as a dependency automatically — no separate Keras install needed)
- **NumPy 2.4.4**
- **Pillow 12.1.1**
- **Matplotlib 3.10.8**

Install with:

```bash
pip install tensorflow numpy pillow matplotlib
```

No GPU is required — the whole pipeline runs on CPU. On native Windows, TensorFlow
≥2.11 has no GPU backend anyway (WSL2 or the DirectML plugin would be needed for GPU
support); everything here was run and verified CPU-only.

## Data required

Both of these must be present alongside the notebook — neither is fetched
automatically:

- **`weights/`** — the 226 pretrained CSV files. Path resolution is anchored to
  `facenet_utils.py`'s own location, so this loads correctly regardless of the
  working directory the notebook is launched from.
- **`archive/lfw-deepfunneled/`** — the LFW dataset. Referenced with a path
  *relative to the working directory* (not anchored like `weights/`), so the notebook
  must be launched with `Project/` as the working directory.

## How to run

1. Install the packages above.
2. Launch Jupyter **from this folder** (`Project/`), so relative paths into
   `archive/` resolve correctly:
   ```bash
   jupyter notebook face_recognition_pipeline.ipynb
   ```
3. Run all cells top to bottom. Runtime is a few minutes on CPU, the accuracy
   section (7) encodes over a thousand LFW images.

## Notes

The notebook depends only on `weights/` and `archive/`. The `images/` folder holds
leftover portraits from the original course assignment and is no longer referenced by
any cell, so a fresh clone without it runs end to end without errors.
