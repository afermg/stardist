# StarDist Nahual OCI image

Build the reproducible archive and load it into Podman or Docker:

```console
nix build .#oci-image
podman load < result                         # or: docker load < result
```

The image is tagged `nahual/stardist:local` and listens on TCP port 5555. The
pretrained model is downloaded on first setup, so persist `/tmp/nahual` as a
model cache:

```console
podman run --rm --device nvidia.com/gpu=all -p 5555:5555 \
  -v nahual-stardist-cache:/tmp/nahual nahual/stardist:local
```

For Docker, replace the CDI device option with `--gpus all`. CPU operation is
supported. With Nahual and NumPy installed on the host, run pretrained
end-to-end segmentation with:

```console
NAHUAL_DEVICE=cpu python oci/smoke_test.py
```

The default is the `2D_versatile_fluo` model. Other pretrained StarDist2D
models and calibrated threshold overrides can be selected in the setup request.
