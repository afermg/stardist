#!/usr/bin/env python3
"""End-to-end pretrained inference against the StarDist OCI container."""

import json
import os

os.environ.setdefault("NAHUAL_IPC_TIMEOUT_MS", "1800000")

import numpy as np
from nahual.process import dispatch_setup_process


def main() -> None:
    address = os.environ.get("NAHUAL_ADDRESS", "tcp://127.0.0.1:5555")
    device = os.environ.get("NAHUAL_DEVICE", "cpu")
    setup, process = dispatch_setup_process("stardist")
    info = setup(
        {
            "model_name": "2D_versatile_fluo",
            "device": device,
            "prob_thresh": 0.9,
        },
        address=address,
    )
    pixels = np.random.default_rng(42).random((1, 1, 1, 256, 256), dtype=np.float32)
    result = process(pixels, address=address)
    expected_device = (
        "CPU:0" if device.lower().startswith("cpu") else f"GPU:{device.split(':')[-1]}"
    )
    assert info["device"] == expected_device, info
    assert info["model_name"] == "2D_versatile_fluo", info
    assert result.shape == (1, 256, 256), result.shape
    assert np.issubdtype(result.dtype, np.integer), result.dtype
    assert (result >= 0).all()
    print(
        json.dumps(
            {
                "setup": info,
                "shape": list(result.shape),
                "instances": int(result.max()),
            }
        )
    )


if __name__ == "__main__":
    main()
