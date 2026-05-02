"""Nahual server for StarDist (TensorFlow backend).

StarDist is a star-convex polygon segmentation model. This server wraps the
2D variant (``StarDist2D``) with a setup/process pair compatible with the
Nahual responder. The default model is ``2D_versatile_fluo`` — the standard
fluorescent-nuclei pretrained checkpoint shipped via
``StarDist2D.from_pretrained``. The first ``setup()`` call downloads the
weights to the user's cache (~50 MB) if not already present.

``process()`` accepts a 5-D NCZYX numpy array, squeezes the leading N and Z
axes (a single 2-D YX image with C channels), runs ``predict_instances``,
and returns the instance label map shape ``(N, H, W)`` as numpy.

Run with:
    nix run --impure . -- ipc:///tmp/stardist.ipc
or:
    python server.py ipc:///tmp/stardist.ipc
"""

import os
import sys
from functools import partial
from typing import Callable

# Reduce TF log noise before importing tensorflow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# TF 2.13 in nixos-24.11 looks up the standalone `keras` module at runtime;
# nixpkgs ships `tf-keras` (2.17) which matches the API. Set the legacy flag
# before importing tensorflow / csbdeep / stardist so `tf.keras` redirects to
# `tf_keras`.
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")

import numpy
import pynng
import tensorflow as tf
import trio
from nahual.preprocess import pad_channel_dim, validate_input_shape
from nahual.server import responder

from stardist.models import StarDist2D

address = sys.argv[1]


def setup(
    model_name: str = "2D_versatile_fluo",
    device: int | None = 0,
    expected_tile_size: int = 16,
    expected_channels: int = 1,
    prob_thresh: float | None = None,
    nms_thresh: float | None = None,
) -> tuple[Callable, dict]:
    """Load a pretrained StarDist2D model.

    Parameters
    ----------
    model_name : str
        One of the pretrained 2D StarDist models (e.g. ``2D_versatile_fluo``,
        ``2D_versatile_he``, ``2D_paper_dsb2018``, ``2D_demo``).
    device : int | None
        CUDA device index. None defaults to 0. TensorFlow will fall back to
        CPU if no GPU is visible.
    expected_tile_size : int
        Required divisor for the trailing spatial dims of incoming arrays.
    expected_channels : int
        Number of channels the model was trained on. ``2D_versatile_fluo``
        expects 1 (grayscale fluorescence); ``2D_versatile_he`` expects 3.
    prob_thresh, nms_thresh : float | None
        Optional overrides for ``predict_instances`` thresholds. ``None``
        means use the model's calibrated defaults.
    """
    # Bind to a specific GPU when requested. Must happen before any TF op
    # actually allocates memory.
    gpus = tf.config.list_physical_devices("GPU")
    if gpus and device is not None:
        try:
            tf.config.set_visible_devices([gpus[int(device)]], "GPU")
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_memory_growth(gpu, True)
        except (RuntimeError, IndexError):
            # set_visible_devices fails if TF is already initialized; ignore.
            pass

    model = StarDist2D.from_pretrained(model_name)

    # Re-query after possibly restricting visibility.
    visible_gpus = tf.config.get_visible_devices("GPU")
    device_str = f"GPU:{device}" if visible_gpus else "CPU:0"

    info = {
        "device": device_str,
        "model_name": model_name,
        "expected_tile_size": expected_tile_size,
        "expected_channels": expected_channels,
        "prob_thresh": prob_thresh,
        "nms_thresh": nms_thresh,
    }

    processor = partial(
        process,
        model=model,
        expected_tile_size=expected_tile_size,
        expected_channels=expected_channels,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )
    return processor, info


def process(
    pixels: numpy.ndarray,
    model: StarDist2D,
    expected_tile_size: int,
    expected_channels: int,
    prob_thresh: float | None,
    nms_thresh: float | None,
) -> numpy.ndarray:
    """Run StarDist2D on an NCZYX numpy array.

    Squeezes the leading N and Z axes (StarDist2D operates on a single 2-D
    image at a time), pads channels up to ``expected_channels``, and returns
    an instance label map with a leading batch axis: shape ``(N, H, W)``.
    """
    if pixels.ndim != 5:
        raise ValueError(
            f"Expected NCZYX (5D) array, got shape {pixels.shape}"
        )
    n, _, z, *input_yx = pixels.shape
    validate_input_shape(input_yx, expected_tile_size)

    # pad_channel_dim drops Z (axis 2) and pads channels (axis 1) → NCYX.
    pixels = pad_channel_dim(pixels, expected_channels).astype(numpy.float32)

    predict_kwargs = {}
    if prob_thresh is not None:
        predict_kwargs["prob_thresh"] = prob_thresh
    if nms_thresh is not None:
        predict_kwargs["nms_thresh"] = nms_thresh

    labels_per_image = []
    for i in range(n):
        img = pixels[i]  # CYX
        if expected_channels == 1:
            # StarDist2D for grayscale models expects a 2-D YX image.
            img2d = img[0]
            axes = "YX"
        else:
            # Multi-channel: feed YXC.
            img2d = numpy.transpose(img, (1, 2, 0))
            axes = "YXC"

        labels, _details = model.predict_instances(
            img2d, axes=axes, **predict_kwargs
        )
        labels_per_image.append(labels.astype(numpy.int32))

    return numpy.stack(labels_per_image, axis=0)


async def main():
    with pynng.Rep0(listen=address, recv_timeout=300) as sock:
        print(f"StarDist server listening on {address}", flush=True)
        async with trio.open_nursery() as nursery:
            responder_curried = partial(responder, setup=setup)
            nursery.start_soon(responder_curried, sock)


if __name__ == "__main__":
    try:
        trio.run(main)
    except KeyboardInterrupt:
        pass
