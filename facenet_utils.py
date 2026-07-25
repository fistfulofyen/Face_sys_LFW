"""Weight loading and image-encoding helpers for the OpenFace embedding network.

The pretrained parameters ship as flat CSV exports of the original Torch model,
one file per tensor, under ``weights/``. Naming convention:

    <layer_name>_w.csv   kernel (conv) / gamma (batch-norm) / kernel (dense)
    <layer_name>_b.csv   bias  (conv)  / beta  (batch-norm) / bias   (dense)
    <layer_name>_m.csv   moving mean       (batch-norm only)
    <layer_name>_v.csv   moving variance   (batch-norm only)

Convolution kernels are stored flat in Torch's ``(out, in, h, w)`` layout and
must be reshaped and transposed into the TensorFlow ``(h, w, in, out)`` layout
before being handed to Keras.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

import numpy as np
import PIL.Image
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Paths and constants
# --------------------------------------------------------------------------- #

# Resolved relative to this file rather than the working directory, so the
# helpers keep working regardless of where the kernel was launched from.
PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = PACKAGE_ROOT / "weights"

FACE_INPUT_SIZE = (96, 96)   # (height, width) expected by the network

# Centre crop for LFW's 250x250 portraits, as ``(box_size, offset_y)``. The
# positive offset shifts the box downwards, away from hair and towards the eyes/
# nose/mouth region the weights were trained on. Measured on 1,800 LFW pairs,
# this lifts verification accuracy from ~58% (no crop) to ~88%.
LFW_FACE_CROP = (92, 18)
DENSE_LAYER_NAME = "dense_layer"
# The embedding layer is the one place where the Keras layer name and the CSV
# filename stem disagree: the layer is 'dense_layer', the exports are 'dense_*'.
DENSE_FILE_STEM = "dense"
DENSE_KERNEL_SHAPE = (128, 736)

# Torch kernel layouts, keyed by Keras layer name: (out_ch, in_ch, kh, kw).
# Membership in this table is also what marks a layer as convolutional.
KERNEL_SHAPES = {
    "conv1": (64, 3, 7, 7),
    "conv2": (64, 64, 1, 1),
    "conv3": (192, 64, 3, 3),

    "inception_3a_1x1_conv":  (64, 192, 1, 1),
    "inception_3a_pool_conv": (32, 192, 1, 1),
    "inception_3a_5x5_conv1": (16, 192, 1, 1),
    "inception_3a_5x5_conv2": (32, 16, 5, 5),
    "inception_3a_3x3_conv1": (96, 192, 1, 1),
    "inception_3a_3x3_conv2": (128, 96, 3, 3),

    "inception_3b_3x3_conv1": (96, 256, 1, 1),
    "inception_3b_3x3_conv2": (128, 96, 3, 3),
    "inception_3b_5x5_conv1": (32, 256, 1, 1),
    "inception_3b_5x5_conv2": (64, 32, 5, 5),
    "inception_3b_pool_conv": (64, 256, 1, 1),
    "inception_3b_1x1_conv":  (64, 256, 1, 1),

    "inception_3c_3x3_conv1": (128, 320, 1, 1),
    "inception_3c_3x3_conv2": (256, 128, 3, 3),
    "inception_3c_5x5_conv1": (32, 320, 1, 1),
    "inception_3c_5x5_conv2": (64, 32, 5, 5),

    "inception_4a_3x3_conv1": (96, 640, 1, 1),
    "inception_4a_3x3_conv2": (192, 96, 3, 3),
    "inception_4a_5x5_conv1": (32, 640, 1, 1),
    "inception_4a_5x5_conv2": (64, 32, 5, 5),
    "inception_4a_pool_conv": (128, 640, 1, 1),
    "inception_4a_1x1_conv":  (256, 640, 1, 1),

    "inception_4e_3x3_conv1": (160, 640, 1, 1),
    "inception_4e_3x3_conv2": (256, 160, 3, 3),
    "inception_4e_5x5_conv1": (64, 640, 1, 1),
    "inception_4e_5x5_conv2": (128, 64, 5, 5),

    "inception_5a_3x3_conv1": (96, 1024, 1, 1),
    "inception_5a_3x3_conv2": (384, 96, 3, 3),
    "inception_5a_pool_conv": (96, 1024, 1, 1),
    "inception_5a_1x1_conv":  (256, 1024, 1, 1),

    "inception_5b_3x3_conv1": (96, 736, 1, 1),
    "inception_5b_3x3_conv2": (384, 96, 3, 3),
    "inception_5b_pool_conv": (96, 736, 1, 1),
    "inception_5b_1x1_conv":  (256, 736, 1, 1),
}

# Every layer carrying pretrained parameters, in export order.
LAYER_MANIFEST = (
    "conv1", "bn1", "conv2", "bn2", "conv3", "bn3",

    "inception_3a_1x1_conv", "inception_3a_1x1_bn",
    "inception_3a_pool_conv", "inception_3a_pool_bn",
    "inception_3a_5x5_conv1", "inception_3a_5x5_conv2",
    "inception_3a_5x5_bn1", "inception_3a_5x5_bn2",
    "inception_3a_3x3_conv1", "inception_3a_3x3_conv2",
    "inception_3a_3x3_bn1", "inception_3a_3x3_bn2",

    "inception_3b_3x3_conv1", "inception_3b_3x3_conv2",
    "inception_3b_3x3_bn1", "inception_3b_3x3_bn2",
    "inception_3b_5x5_conv1", "inception_3b_5x5_conv2",
    "inception_3b_5x5_bn1", "inception_3b_5x5_bn2",
    "inception_3b_pool_conv", "inception_3b_pool_bn",
    "inception_3b_1x1_conv", "inception_3b_1x1_bn",

    "inception_3c_3x3_conv1", "inception_3c_3x3_conv2",
    "inception_3c_3x3_bn1", "inception_3c_3x3_bn2",
    "inception_3c_5x5_conv1", "inception_3c_5x5_conv2",
    "inception_3c_5x5_bn1", "inception_3c_5x5_bn2",

    "inception_4a_3x3_conv1", "inception_4a_3x3_conv2",
    "inception_4a_3x3_bn1", "inception_4a_3x3_bn2",
    "inception_4a_5x5_conv1", "inception_4a_5x5_conv2",
    "inception_4a_5x5_bn1", "inception_4a_5x5_bn2",
    "inception_4a_pool_conv", "inception_4a_pool_bn",
    "inception_4a_1x1_conv", "inception_4a_1x1_bn",

    "inception_4e_3x3_conv1", "inception_4e_3x3_conv2",
    "inception_4e_3x3_bn1", "inception_4e_3x3_bn2",
    "inception_4e_5x5_conv1", "inception_4e_5x5_conv2",
    "inception_4e_5x5_bn1", "inception_4e_5x5_bn2",

    "inception_5a_3x3_conv1", "inception_5a_3x3_conv2",
    "inception_5a_3x3_bn1", "inception_5a_3x3_bn2",
    "inception_5a_pool_conv", "inception_5a_pool_bn",
    "inception_5a_1x1_conv", "inception_5a_1x1_bn",

    "inception_5b_3x3_conv1", "inception_5b_3x3_conv2",
    "inception_5b_3x3_bn1", "inception_5b_3x3_bn2",
    "inception_5b_pool_conv", "inception_5b_pool_bn",
    "inception_5b_1x1_conv", "inception_5b_1x1_bn",

    DENSE_LAYER_NAME,
)


# --------------------------------------------------------------------------- #
# Weight loading
# --------------------------------------------------------------------------- #

def _read_tensor(weights_dir: Path, layer: str, suffix: str) -> np.ndarray:
    """Read one flat CSV export as a 1-D float array."""
    return np.genfromtxt(weights_dir / f"{layer}_{suffix}.csv",
                         delimiter=",", dtype=None)


def _load_conv(weights_dir: Path, layer: str) -> list[np.ndarray]:
    """Kernel + bias, converted from Torch (out, in, h, w) to TF (h, w, in, out)."""
    kernel = _read_tensor(weights_dir, layer, "w").reshape(KERNEL_SHAPES[layer])
    kernel = kernel.transpose(2, 3, 1, 0)
    bias = _read_tensor(weights_dir, layer, "b")
    return [kernel, bias]


def _load_batchnorm(weights_dir: Path, layer: str) -> list[np.ndarray]:
    """Keras BatchNormalization expects [gamma, beta, moving_mean, moving_variance]."""
    return [_read_tensor(weights_dir, layer, suffix) for suffix in ("w", "b", "m", "v")]


def _load_dense(weights_dir: Path) -> list[np.ndarray]:
    """Kernel + bias, transposed from (units, features) to (features, units)."""
    kernel = _read_tensor(weights_dir, DENSE_FILE_STEM, "w").reshape(DENSE_KERNEL_SHAPE)
    kernel = kernel.transpose(1, 0)
    bias = _read_tensor(weights_dir, DENSE_FILE_STEM, "b")
    return [kernel, bias]


def read_weight_bundle(weights_dir=None) -> dict[str, list[np.ndarray]]:
    """Parse every CSV export into ``{layer_name: [arrays in Keras order]}``."""
    weights_dir = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
    if not weights_dir.is_dir():
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    bundle = {}
    for layer in LAYER_MANIFEST:
        if layer in KERNEL_SHAPES:
            bundle[layer] = _load_conv(weights_dir, layer)
        elif layer == DENSE_LAYER_NAME:
            bundle[layer] = _load_dense(weights_dir)
        else:
            bundle[layer] = _load_batchnorm(weights_dir, layer)
    return bundle


def apply_pretrained_weights(model, weights_dir=None) -> int:
    """Populate ``model`` in place from the CSV exports.

    Returns:
        The number of layers that received weights (expected: 75).
    """
    bundle = read_weight_bundle(weights_dir)
    for layer_name, arrays in bundle.items():
        model.get_layer(layer_name).set_weights(arrays)
    return len(bundle)


# --------------------------------------------------------------------------- #
# Image encoding
# --------------------------------------------------------------------------- #

def preprocess_image(image_path, target_size=FACE_INPUT_SIZE, crop=None) -> np.ndarray:
    """Load an image file as a batched, channels-first, [0, 1]-scaled array.

    Args:
        image_path: file to read.
        target_size: ``(height, width)`` fed to the network.
        crop: optional ``(box_size, offset_y)`` centre crop applied *before*
            resizing. Full-frame photographs such as LFW's 250x250 portraits
            spend most of their pixels on background and hair; cropping to the
            face region first is worth a large accuracy swing. See
            :data:`LFW_FACE_CROP`.

    Returns:
        Array of shape ``(1, 3, H, W)``.
    """
    if crop is None:
        image = tf.keras.preprocessing.image.load_img(image_path,
                                                      target_size=target_size)
    else:
        box_size, offset_y = crop
        image = PIL.Image.open(image_path).convert("RGB")
        left = (image.width - box_size) // 2
        top = (image.height - box_size) // 2 + offset_y
        image = image.crop((left, top, left + box_size, top + box_size))
        image = image.resize(target_size[::-1], PIL.Image.BILINEAR)

    scaled = np.around(np.transpose(image, (2, 0, 1)) / 255.0, decimals=12)
    return np.expand_dims(scaled, axis=0)


def encode_image(image_path, model, target_size=FACE_INPUT_SIZE, crop=None) -> np.ndarray:
    """Map a single image file to its 128-dimensional embedding, shape ``(1, 128)``."""
    return model.predict_on_batch(preprocess_image(image_path, target_size, crop))


def encode_many(image_paths, model, target_size=FACE_INPUT_SIZE, crop=None,
                batch_size=256, progress=False) -> np.ndarray:
    """Embed a list of images in batches. Returns shape ``(N, 128)``.

    Far faster than repeated :func:`encode_image` calls once you are working
    with thousands of files.
    """
    paths = list(image_paths)
    output = np.zeros((len(paths), 128), dtype=np.float32)

    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch = np.concatenate(
            [preprocess_image(p, target_size, crop) for p in chunk], axis=0
        )
        output[start:start + len(chunk)] = model.predict_on_batch(batch)
        if progress:
            print(f"\r  encoded {start + len(chunk)}/{len(paths)}", end="", flush=True)

    if progress:
        print()
    return output


def embedding_distance(left, right) -> float:
    """Euclidean (L2) distance between two embeddings."""
    return float(np.linalg.norm(np.asarray(left) - np.asarray(right)))


def build_identity_database(entries, model, target_size=FACE_INPUT_SIZE,
                            crop=None) -> dict:
    """Encode a ``{name: image_path}`` mapping into ``{name: embedding}``."""
    return {
        name: encode_image(path, model, target_size, crop)
        for name, path in entries.items()
    }


# --------------------------------------------------------------------------- #
# Visual comparison
# --------------------------------------------------------------------------- #

def show_face_pair(probe_path, match_path, distance=None, probe_label="query",
                   match_label="match", accepted=None, crop=None, figsize=(6, 3.2)):
    """Display a query face beside the face it was matched against.

    Args:
        probe_path: image being identified.
        match_path: image of the identity it was matched to.
        distance: L2 distance between the two embeddings, shown in the title.
        probe_label / match_label: captions under each panel.
        accepted: ``True``/``False`` to colour-code the verdict; ``None`` for neutral.
        crop: same ``(box_size, offset_y)`` form as :func:`preprocess_image`, so the
            panels show what the network actually saw.
        figsize: forwarded to matplotlib.
    """
    import matplotlib.pyplot as plt

    def _panel(path):
        image = PIL.Image.open(path).convert("RGB")
        if crop is not None:
            box_size, offset_y = crop
            left = (image.width - box_size) // 2
            top = (image.height - box_size) // 2 + offset_y
            image = image.crop((left, top, left + box_size, top + box_size))
        return image

    figure, axes = plt.subplots(1, 2, figsize=figsize)
    for axis, path, caption in ((axes[0], probe_path, probe_label),
                                (axes[1], match_path, match_label)):
        axis.imshow(_panel(path))
        axis.set_title(caption, fontsize=11)
        axis.axis("off")

    if distance is not None:
        colour = "black" if accepted is None else ("seagreen" if accepted else "firebrick")
        verdict = "" if accepted is None else ("   ACCEPTED" if accepted else "   REJECTED")
        figure.suptitle(f"L2 distance = {distance:.4f}{verdict}",
                        fontsize=12, color=colour)

    figure.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# Benchmarking against LFW
# --------------------------------------------------------------------------- #

LFW_IMAGE_ROOT = PACKAGE_ROOT / "archive" / "lfw-deepfunneled" / "lfw-deepfunneled"
LFW_PAIRS_CSV = PACKAGE_ROOT / "archive" / "pairs.csv"


def _lfw_path(person: str, index) -> Path:
    return LFW_IMAGE_ROOT / person / f"{person}_{int(index):04d}.jpg"


def load_lfw_pairs(n_pairs=600, seed=0, pairs_csv=None):
    """Sample a class-balanced subset of the official LFW benchmark pairs.

    The file holds 6,000 pairs -- 3,000 of the same person and 3,000 of different
    people. Rows with a fourth field name a second person (a mismatched pair);
    rows without one reuse the first name (a matched pair).

    Args:
        n_pairs: total pairs to return; half matched, half mismatched.
        seed: RNG seed, so a run is reproducible.
        pairs_csv: override the default ``archive/pairs.csv`` location.

    Returns:
        List of ``(path_a, path_b, label)`` with label 1 for "same person".
    """
    pairs_csv = Path(pairs_csv) if pairs_csv is not None else LFW_PAIRS_CSV
    with open(pairs_csv, newline="") as handle:
        rows = list(csv.reader(handle))[1:]

    matched, mismatched = [], []
    for row in rows:
        if len(row) > 3 and row[3].strip():
            mismatched.append((_lfw_path(row[0], row[1]), _lfw_path(row[2], row[3]), 0))
        else:
            matched.append((_lfw_path(row[0], row[1]), _lfw_path(row[0], row[2]), 1))

    rng = random.Random(seed)
    half = n_pairs // 2
    sample = rng.sample(matched, min(half, len(matched))) + \
             rng.sample(mismatched, min(half, len(mismatched)))
    rng.shuffle(sample)
    return sample


def pair_distances(pairs, model, crop=LFW_FACE_CROP, progress=True):
    """Embed every unique image once, then return ``(distances, labels)``."""
    unique = sorted({str(path) for a, b, _ in pairs for path in (a, b)})
    lookup = {path: i for i, path in enumerate(unique)}

    if progress:
        print(f"encoding {len(unique)} unique images from {len(pairs)} pairs...")
    embeddings = encode_many(unique, model, crop=crop, progress=progress)

    left = embeddings[[lookup[str(a)] for a, _, _ in pairs]]
    right = embeddings[[lookup[str(b)] for _, b, _ in pairs]]
    distances = np.linalg.norm(left - right, axis=1)
    labels = np.array([label for _, _, label in pairs])
    return distances, labels


def sweep_threshold(distances, labels, grid=None):
    """Accuracy at every candidate threshold. Returns ``(thresholds, accuracies)``."""
    grid = np.linspace(0.0, 2.0, 401) if grid is None else np.asarray(grid)
    accuracies = np.array([((distances < t) == labels).mean() for t in grid])
    return grid, accuracies


def roc_curve(distances, labels):
    """ROC for a distance-based classifier. Returns ``(fpr, tpr, auc)``."""
    order = np.argsort(distances)
    ranked = labels[order]
    positives, negatives = ranked.sum(), len(ranked) - ranked.sum()
    tpr = np.cumsum(ranked) / positives
    fpr = np.cumsum(1 - ranked) / negatives
    return fpr, tpr, float(np.trapezoid(tpr, fpr))


def evaluate_verification(distances, labels, folds=10, seed=0):
    """Cross-validated verification accuracy, the standard LFW protocol.

    The decision threshold is a fitted parameter, so quoting accuracy at the
    threshold that maximises accuracy on the *same* data overstates performance.
    Here each fold is scored using a threshold chosen on the other folds.

    Returns:
        Dict with per-fold accuracies, their mean and standard deviation, the
        optimistic single-threshold figure, ROC AUC, and the equal-error rate.
    """
    distances, labels = np.asarray(distances), np.asarray(labels)
    grid, accuracies = sweep_threshold(distances, labels)

    rng = np.random.default_rng(seed)
    assignment = rng.permutation(len(distances)) % folds

    fold_scores, fold_thresholds = [], []
    for fold in range(folds):
        test = assignment == fold
        train = ~test
        _, train_acc = sweep_threshold(distances[train], labels[train], grid)
        threshold = grid[int(np.argmax(train_acc))]
        fold_scores.append(((distances[test] < threshold) == labels[test]).mean())
        fold_thresholds.append(threshold)

    fpr, tpr, auc = roc_curve(distances, labels)
    eer = float(fpr[np.nanargmin(np.abs((1 - tpr) - fpr))])

    return {
        "fold_accuracies": np.array(fold_scores),
        "accuracy_mean": float(np.mean(fold_scores)),
        "accuracy_std": float(np.std(fold_scores)),
        "threshold_mean": float(np.mean(fold_thresholds)),
        "best_threshold": float(grid[int(np.argmax(accuracies))]),
        "best_accuracy_optimistic": float(accuracies.max()),
        "auc": auc,
        "eer": eer,
        "roc": (fpr, tpr),
        "sweep": (grid, accuracies),
        "n_pairs": len(distances),
    }


def distance_matrix(probe_embeddings, gallery_embeddings) -> np.ndarray:
    """All probe-to-gallery distances, shape ``(n_probes, n_gallery)``."""
    probe_embeddings = np.asarray(probe_embeddings)
    gallery_embeddings = np.asarray(gallery_embeddings)
    deltas = probe_embeddings[:, None, :] - gallery_embeddings[None, :, :]
    return np.linalg.norm(deltas, axis=2)


def identification_ranks(distances) -> np.ndarray:
    """Rank of the correct gallery entry for each probe, assuming probe *i*
    corresponds to gallery entry *i*.

    A rank of 0 means the correct identity was the nearest neighbour (a rank-1
    hit); a rank of 3 means three wrong people were closer.
    """
    distances = np.asarray(distances)
    n = len(distances)
    correct = distances[np.arange(n), np.arange(n)]
    return (distances < correct[:, None]).sum(axis=1)


def rank1_vs_gallery_size(distances, sizes, trials=20, seed=0):
    """Rank-1 accuracy as the gallery grows.

    Identification gets monotonically harder with more enrolled people, because
    every extra identity is another chance for a wrong match to be nearer. This
    resamples sub-galleries of each size so a single quoted number can be seen
    in context.

    Returns:
        ``(sizes, mean_accuracy, std_accuracy)``.
    """
    distances = np.asarray(distances)
    n = len(distances)
    rng = np.random.default_rng(seed)

    means, deviations = [], []
    for size in sizes:
        scores = []
        for _ in range(trials):
            picked = rng.choice(n, min(size, n), replace=False)
            block = distances[np.ix_(picked, picked)]
            scores.append((block.argmin(axis=1) == np.arange(len(picked))).mean())
        means.append(np.mean(scores))
        deviations.append(np.std(scores))

    return np.asarray(sizes), np.asarray(means), np.asarray(deviations)


def rank1_identification(gallery_names, gallery_embeddings, probe_names,
                         probe_embeddings):
    """Closed-set rank-1 identification accuracy.

    Every probe is assigned the identity of its nearest gallery entry; accuracy
    is the fraction assigned correctly. Harder than verification, because each
    probe competes against the whole gallery rather than one claimed identity.
    """
    gallery_embeddings = np.asarray(gallery_embeddings)
    probe_embeddings = np.asarray(probe_embeddings)

    # Pairwise distances via broadcasting: (n_probes, n_gallery)
    deltas = probe_embeddings[:, None, :] - gallery_embeddings[None, :, :]
    nearest = np.linalg.norm(deltas, axis=2).argmin(axis=1)

    predicted = [gallery_names[i] for i in nearest]
    correct = np.array([p == t for p, t in zip(predicted, probe_names)])
    return float(correct.mean()), predicted
