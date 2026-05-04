# /usr/bin/env python
"""
This example uses a server within the environment defined on `https://github.com/afermg/stardist.git`.

Run `nix run github:afermg/stardist/nahual-wrap -- ipc:///tmp/stardist.ipc` from any
directory, or `nix develop --command bash -c "python server.py ipc:///tmp/stardist.ipc"`
from the root of that repository.
"""

import numpy

from nahual.process import dispatch_setup_process

# StarDist isn't in nahual's built-in registry; pass signature explicitly.
setup, process = dispatch_setup_process("stardist", signature=("dict", "numpy"))
address = "ipc:///tmp/stardist.ipc"

# %% Load model server-side
parameters = {
    "model_name": "2D_versatile_fluo",
    "device": 0,
}
response = setup(parameters, address=address)
print(response)
# Expected: {'device': 'GPU:0', 'model_name': '2D_versatile_fluo', ...}

# %% Define custom data
# Image models: 5-D NCZYX. StarDist2D squeezes the Z axis server-side.
tile_size = 256  # multiples of 16
numpy.random.seed(seed=42)
data = numpy.random.random_sample((1, 1, 1, tile_size, tile_size)).astype(numpy.float32)
result = process(data, address=address)
print(f"Shape: {result.shape}, dtype: {result.dtype}, max_label: {result.max()}")
# Expected: Shape: (1, 256, 256), dtype: int32, max_label: <number of detected nuclei>
