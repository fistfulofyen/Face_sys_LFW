# Face Recognition Pipeline

A face verification and identification pipeline built on a pretrained OpenFace
`nn4.small2` embedding network (Inception-style CNN, triplet-loss trained, 128-d
L2-normalized output), evaluated on the LFW (Labeled Faces in the Wild) dataset.

## Contents

| File / folder | Purpose |
|---|---|
| `face_recognition_pipeline.ipynb` | Main notebook, build the network, load weights, run verification/identification demos, measure accuracy |
| `facenet_arch.py` | Network architecture definition |
| `facenet_utils.py` | Weight loading (CSV → Keras) and image-encoding helpers |
| `weights/` | Pretrained CSV weight export (226 files, required) |
| `archive/` | LFW dataset, `lfw-deepfunneled/` images plus `pairs.csv` and related metadata (required) |

## Requirements

- **Python 3.11** (tested on 3.11.15)
- **TensorFlow 2.21.0** (brings in Keras 3.x as a dependency automatically, no separate Keras install needed)
- **NumPy 2.4.4**
- **Pillow 12.1.1**
- **Matplotlib 3.10.8**

Install with:

```bash
pip install tensorflow numpy pillow matplotlib
```

No GPU is required, the whole pipeline runs on CPU. On native Windows, TensorFlow
≥2.11 has no GPU backend anyway (WSL2 or the DirectML plugin would be needed for GPU
support); everything here was run and verified CPU-only.

## Data required

Both of these must be present alongside the notebook, neither is fetched
automatically:

- **`weights/`**, the 226 pretrained CSV files. Path resolution is anchored to
  `facenet_utils.py`'s own location, so this loads correctly regardless of the
  working directory the notebook is launched from.
- **`archive/lfw-deepfunneled/`**, the LFW dataset. Referenced with a path
  *relative to the working directory* (not anchored like `weights/`), so the notebook
  must be launched with `Project/` as the working directory.

## Where the data came from

Neither dataset in this repository was created by this project.

**`archive/`, Labeled Faces in the Wild (LFW)**

13,233 photographs of 5,749 people, published by the University of Massachusetts,
Amherst. The images were collected from online news sources, and each photograph
remains under the copyright of its original creator. This project uses the
`lfw-deepfunneled` variant, in which the faces have already been detected and
aligned, plus the official `pairs.csv` benchmark list (3,000 same-person and
3,000 different-person pairs).

- Dataset: <https://vis-www.cs.umass.edu/lfw/>
- Citation: G. B. Huang, M. Ramesh, T. Berg, and E. Learned-Miller, *Labeled Faces
  in the Wild: A Database for Studying Face Recognition in Unconstrained
  Environments*, University of Massachusetts Amherst, Technical Report 07-49,
  October 2007.

**`weights/`, pretrained OpenFace model**

The 226 CSV files are the `nn4.small2.v1` weights from the OpenFace project at
Carnegie Mellon University, released under the Apache License 2.0 and trained on
CASIA-WebFace and FaceScrub (~500,000 images). The CSV export was produced by the
Keras-OpenFace project. No training is performed in this repository; the network is
used as a frozen feature extractor.

- OpenFace: <https://github.com/cmusatyalab/openface>
- CSV export: <https://github.com/iwantooxxoox/Keras-OpenFace>

See [`LICENSE`](LICENSE) for the full third-party attribution and terms.

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
