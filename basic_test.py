"""Standalone smoke test for StarDist.

Loads StarDist2D the same way ``server.py`` does and runs a forward pass on a
small synthetic input. Does NOT spin up the IPC server — the goal here is to
verify the dev shell + model assembly without the network plumbing.

Run from the repo root:
    nix develop --impure --command python basic_test.py

Expected output:
    setup info dict (with device == GPU:0)
    type and shape of process() result.
"""

import os
import sys

# server.py reads sys.argv[1] at import time; inject a placeholder so
# importing it from this file doesn't crash.
if len(sys.argv) < 2:
    sys.argv.append("ipc:///tmp/stardist_basic_test.ipc")

# PYTHONSAFEPATH=1 (set in flake's runServer/devShell) keeps Python from
# prepending the script's directory to sys.path (which would let the
# in-tree `stardist/` source tree shadow the nix-built compiled package).
# We still want `server.py` to be importable, so APPEND the script dir
# at the tail — nix-store packages keep priority.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy  # noqa: E402
from server import setup  # noqa: E402


def main() -> None:
    processor, info = setup()
    print(f"setup: {info}")
    assert "GPU" in info["device"] or "cuda" in info["device"], (
        f"Not on GPU! info['device']={info['device']!r}"
    )

    # 5-D NCZYX. Single-channel, single-Z, 256x256.
    numpy.random.seed(0)
    data = numpy.random.random_sample((1, 1, 1, 256, 256)).astype(numpy.float32)
    out = processor(data)

    arr = out.cpu().numpy() if hasattr(out, "cpu") else out
    print(f"process: {type(arr).__name__} shape={arr.shape} dtype={arr.dtype} max_label={int(arr.max())}")


if __name__ == "__main__":
    main()
