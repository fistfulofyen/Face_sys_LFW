"""OpenFace / FaceNet embedding network (nn4.small2 variant).

Builds the Inception-style convolutional network that maps a 96x96 RGB face
crop to a unit-norm 128-dimensional embedding.

The architecture is expressed declaratively: every Inception stage is assembled
from two reusable primitives -- a reduce/expand convolutional branch and a
pooling branch -- instead of being spelled out layer by layer.

NOTE ON LAYER NAMES
-------------------
The ``name=`` strings assigned to Conv2D / BatchNormalization / Dense layers are
NOT cosmetic. They are the lookup keys used by :mod:`facenet_utils` to match each
layer against its corresponding ``weights/<name>_[wbmv].csv`` file. Renaming any
of them will break weight loading.
"""

from __future__ import annotations

from typing import Callable, Sequence

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Activation,
    AveragePooling2D,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    Lambda,
    MaxPooling2D,
    ZeroPadding2D,
)
from tensorflow.keras.models import Model

# --------------------------------------------------------------------------- #
# Global configuration
# --------------------------------------------------------------------------- #

DATA_FORMAT = "channels_first"
CHANNEL_AXIS = 1
EMBEDDING_DIM = 128
DEFAULT_INPUT_SHAPE = (3, 96, 96)

# The pretrained Torch weights were exported with this epsilon everywhere...
BN_EPSILON = 1e-5
# ...except the very first stem block, which was originally instantiated without
# an explicit epsilon and therefore inherited the Keras default. Reproducing this
# inconsistency is required for numerically identical embeddings.
STEM_BN_EPSILON = 1e-3

K.set_image_data_format(DATA_FORMAT)


# --------------------------------------------------------------------------- #
# Primitives
# --------------------------------------------------------------------------- #

def _conv_bn_relu(tensor, conv_name, bn_name, filters, kernel,
                  strides=(1, 1), epsilon=BN_EPSILON):
    """Conv2D -> BatchNormalization -> ReLU, with explicitly named weight layers."""
    tensor = Conv2D(
        filters,
        kernel,
        strides=strides,
        data_format=DATA_FORMAT,
        name=conv_name,
    )(tensor)
    tensor = BatchNormalization(
        axis=CHANNEL_AXIS,
        epsilon=epsilon,
        name=bn_name,
    )(tensor)
    return Activation("relu")(tensor)


def _conv_branch(tensor, tag, *, reduce_filters, expand_filters=None,
                 expand_kernel=(3, 3), expand_strides=(1, 1), pad=None):
    """A bottleneck Inception branch.

    Always applies a 1x1 channel reduction. When ``expand_filters`` is given the
    branch continues with zero-padding and a second spatial convolution, and the
    two weight layers are suffixed ``1`` / ``2`` to match the exported filenames.
    """
    suffix = "" if expand_filters is None else "1"
    tensor = _conv_bn_relu(
        tensor,
        f"{tag}_conv{suffix}",
        f"{tag}_bn{suffix}",
        reduce_filters,
        (1, 1),
    )

    if pad is not None:
        tensor = ZeroPadding2D(padding=pad, data_format=DATA_FORMAT)(tensor)

    if expand_filters is None:
        return tensor

    return _conv_bn_relu(
        tensor,
        f"{tag}_conv2",
        f"{tag}_bn2",
        expand_filters,
        expand_kernel,
        strides=expand_strides,
    )


def _pool_branch(tensor, kind, pool_size, strides, *,
                 tag=None, filters=None, pad=None):
    """A pooling Inception branch, optionally projected by a 1x1 convolution."""
    pooling = MaxPooling2D if kind == "max" else AveragePooling2D
    tensor = pooling(pool_size=pool_size, strides=strides,
                     data_format=DATA_FORMAT)(tensor)

    if tag is not None:
        tensor = _conv_bn_relu(tensor, f"{tag}_conv", f"{tag}_bn", filters, (1, 1))

    if pad is not None:
        tensor = ZeroPadding2D(padding=pad, data_format=DATA_FORMAT)(tensor)

    return tensor


def _merge(branches: Sequence):
    """Depth-concatenate branch outputs. Order defines channel layout -- do not shuffle."""
    return Concatenate(axis=CHANNEL_AXIS)(list(branches))


# --------------------------------------------------------------------------- #
# Inception stages
# --------------------------------------------------------------------------- #

def stage_3a(x):
    return _merge([
        _conv_branch(x, "inception_3a_3x3", reduce_filters=96,
                     expand_filters=128, expand_kernel=(3, 3), pad=(1, 1)),
        _conv_branch(x, "inception_3a_5x5", reduce_filters=16,
                     expand_filters=32, expand_kernel=(5, 5), pad=(2, 2)),
        _pool_branch(x, "max", 3, 2, tag="inception_3a_pool",
                     filters=32, pad=((3, 4), (3, 4))),
        _conv_branch(x, "inception_3a_1x1", reduce_filters=64),
    ])


def stage_3b(x):
    return _merge([
        _conv_branch(x, "inception_3b_3x3", reduce_filters=96,
                     expand_filters=128, expand_kernel=(3, 3), pad=(1, 1)),
        _conv_branch(x, "inception_3b_5x5", reduce_filters=32,
                     expand_filters=64, expand_kernel=(5, 5), pad=(2, 2)),
        _pool_branch(x, "avg", (3, 3), (3, 3), tag="inception_3b_pool",
                     filters=64, pad=(4, 4)),
        _conv_branch(x, "inception_3b_1x1", reduce_filters=64),
    ])


def stage_3c(x):
    """Downsampling stage -- stride-2 expansions, no 1x1 passthrough branch."""
    return _merge([
        _conv_branch(x, "inception_3c_3x3", reduce_filters=128,
                     expand_filters=256, expand_kernel=(3, 3),
                     expand_strides=(2, 2), pad=(1, 1)),
        _conv_branch(x, "inception_3c_5x5", reduce_filters=32,
                     expand_filters=64, expand_kernel=(5, 5),
                     expand_strides=(2, 2), pad=(2, 2)),
        _pool_branch(x, "max", 3, 2, pad=((0, 1), (0, 1))),
    ])


def stage_4a(x):
    return _merge([
        _conv_branch(x, "inception_4a_3x3", reduce_filters=96,
                     expand_filters=192, expand_kernel=(3, 3), pad=(1, 1)),
        _conv_branch(x, "inception_4a_5x5", reduce_filters=32,
                     expand_filters=64, expand_kernel=(5, 5), pad=(2, 2)),
        _pool_branch(x, "avg", (3, 3), (3, 3), tag="inception_4a_pool",
                     filters=128, pad=(2, 2)),
        _conv_branch(x, "inception_4a_1x1", reduce_filters=256),
    ])


def stage_4e(x):
    """Second downsampling stage."""
    return _merge([
        _conv_branch(x, "inception_4e_3x3", reduce_filters=160,
                     expand_filters=256, expand_kernel=(3, 3),
                     expand_strides=(2, 2), pad=(1, 1)),
        _conv_branch(x, "inception_4e_5x5", reduce_filters=64,
                     expand_filters=128, expand_kernel=(5, 5),
                     expand_strides=(2, 2), pad=(2, 2)),
        _pool_branch(x, "max", 3, 2, pad=((0, 1), (0, 1))),
    ])


def stage_5a(x):
    return _merge([
        _conv_branch(x, "inception_5a_3x3", reduce_filters=96,
                     expand_filters=384, expand_kernel=(3, 3), pad=(1, 1)),
        _pool_branch(x, "avg", (3, 3), (3, 3), tag="inception_5a_pool",
                     filters=96, pad=(1, 1)),
        _conv_branch(x, "inception_5a_1x1", reduce_filters=256),
    ])


def stage_5b(x):
    return _merge([
        _conv_branch(x, "inception_5b_3x3", reduce_filters=96,
                     expand_filters=384, expand_kernel=(3, 3), pad=(1, 1)),
        _pool_branch(x, "max", 3, 2, tag="inception_5b_pool",
                     filters=96, pad=(1, 1)),
        _conv_branch(x, "inception_5b_1x1", reduce_filters=256),
    ])


INCEPTION_STAGES: tuple[Callable, ...] = (
    stage_3a, stage_3b, stage_3c,
    stage_4a, stage_4e,
    stage_5a, stage_5b,
)


# --------------------------------------------------------------------------- #
# Full network
# --------------------------------------------------------------------------- #

def _build_stem(tensor):
    """Three strided/padded conv blocks that take 96x96 down to the Inception trunk."""
    tensor = ZeroPadding2D(padding=(3, 3), data_format=DATA_FORMAT)(tensor)
    tensor = _conv_bn_relu(tensor, "conv1", "bn1", 64, (7, 7),
                           strides=(2, 2), epsilon=STEM_BN_EPSILON)

    tensor = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(tensor)
    tensor = MaxPooling2D(pool_size=(3, 3), strides=2, data_format=DATA_FORMAT)(tensor)

    tensor = _conv_bn_relu(tensor, "conv2", "bn2", 64, (1, 1))
    tensor = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(tensor)

    tensor = _conv_bn_relu(tensor, "conv3", "bn3", 192, (3, 3))
    tensor = ZeroPadding2D(padding=(1, 1), data_format=DATA_FORMAT)(tensor)
    tensor = MaxPooling2D(pool_size=3, strides=2, data_format=DATA_FORMAT)(tensor)

    return tensor


def build_embedding_network(input_shape=DEFAULT_INPUT_SHAPE,
                            model_name="OpenFaceEmbedder") -> Model:
    """Assemble the full face-embedding network.

    Args:
        input_shape: channels-first image shape, ``(channels, height, width)``.
        model_name: name attached to the returned Keras ``Model``.

    Returns:
        An uncompiled Keras ``Model`` mapping images to L2-normalized
        128-dimensional embeddings. Weights are randomly initialized -- call
        ``facenet_utils.apply_pretrained_weights`` to populate them.
    """
    image_input = Input(shape=input_shape)

    features = _build_stem(image_input)
    for stage in INCEPTION_STAGES:
        features = stage(features)

    features = AveragePooling2D(pool_size=(3, 3), strides=(1, 1),
                                data_format=DATA_FORMAT)(features)
    features = Flatten()(features)
    features = Dense(EMBEDDING_DIM, name="dense_layer")(features)

    # Projecting onto the unit hypersphere makes squared-L2 distance and cosine
    # similarity monotonically equivalent, which is what the triplet loss assumes.
    embedding = Lambda(
        lambda t: K.l2_normalize(t, axis=1),
        name="l2_normalize",
    )(features)

    return Model(inputs=image_input, outputs=embedding, name=model_name)
