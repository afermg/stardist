{
  description = "StarDist Nahual server (TensorFlow backend, GPU-enabled)";

  inputs = {
    # Pinned to nixos-24.11 because TF 2.13 / tf-keras isn't packaged on
    # unstable, and csbdeep needs a TF-2.x runtime that nix has actually
    # built (see DeepProfiler precedent).
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      systems,
      ...
    }@inputs:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          system = system;
          config = {
            allowUnfree = true;
            # cudaSupport=true selects nixpkgs' python311-tensorflow-gpu-2.13.0
            # derivation (a separate, CUDA-linked TF wheel — not the same store
            # path as the CPU-only python311-tensorflow-2.13.0). This pulls in
            # cuda-merged-11.8 + cudnn-merged + UCC/NCCL/NVSHMEM, which on a
            # cold cache compiles UCC/NCCL from source for ~30+ min. Required
            # for GPU runtime, so we keep it on.
            cudaSupport = true;
          };
        };
      in
      with pkgs;
      rec {
        # tensorflow 2.13 in nixos-24.11 only supports up to python3.11.
        pythonForServer = pkgs.python311;

        apps.default =
          let
            python_with_pkgs = pythonForServer.withPackages (pp: [
              packages.nahual
              packages.stardist
              packages.csbdeep
              pp.tensorflow
              # tf-keras' wheel METADATA lists `tensorflow` as a runtime dep,
              # but with cudaSupport=true we ship `tensorflow-gpu` (different
              # package name, same `tensorflow` import path). Skip the
              # runtime-deps check to avoid the spurious failure.
              (pp.tf-keras.overridePythonAttrs (_: { dontCheckRuntimeDeps = true; }))
              # csbdeep does a raw `from keras import __version__` (does NOT
              # honor TF_USE_LEGACY_KERAS for that import). nixpkgs 24.11's
              # python311Packages.keras conflicts with tf-keras at install
              # time, so instead csbdeep itself is patched to look at tf_keras
              # — see nix/csbdeep.nix postPatch.
              pp.numpy
              pp.scikit-image
              pp.numba
              pp.imageio
              pp.tifffile
              pp.trio
            ]);
            runServer = pkgs.writeScriptBin "runserver.sh" ''
              #!${pkgs.bash}/bin/bash
              # TF 2.13 in nixos-24.11 expects standalone keras at runtime;
              # tf-keras 2.17 in legacy mode satisfies the API. csbdeep also
              # honours TF_USE_LEGACY_KERAS when picking its keras backend.
              export TF_USE_LEGACY_KERAS=1
              # PYTHONSAFEPATH=1 (Python 3.11+) prevents Python from prepending
              # the script's directory to sys.path. Without it, the upstream
              # `stardist/` source tree at ${self}/stardist shadows the nix-
              # built compiled package and `stardist.lib.stardist2d` (the C
              # extension) is reported missing — same source-shadow trap that
              # the importlib guard handles in basic_test.py.
              export PYTHONSAFEPATH=1
              ${python_with_pkgs}/bin/python ${self}/server.py ''${@:-"ipc:///tmp/stardist.ipc"}
            '';
          in
          {
            type = "app";
            program = "${runServer}/bin/runserver.sh";
          };

        packages = {
          # Build pynng locally for python3.11 (tensorflow 2.13's interpreter).
          pynng = pythonForServer.pkgs.callPackage ./nix/pynng.nix { };
          # nahual recipe sourced from upstream flake; built against our
          # local python3.11 since the upstream-built python3.13 wheel
          # would ABI-clash with TF 2.13.
          nahual = pythonForServer.pkgs.callPackage (inputs.nahual-flake + "/nix/nahual.nix") {
            pynng = packages.pynng;
          };
          csbdeep = pythonForServer.pkgs.callPackage ./nix/csbdeep.nix {
            tf-keras = pythonForServer.pkgs.tf-keras.overridePythonAttrs (_: {
              dontCheckRuntimeDeps = true;
            });
          };
          stardist = pythonForServer.pkgs.callPackage ./nix/stardist.nix {
            csbdeep = packages.csbdeep;
          };
        };

        devShells = {
          default =
            let
              python_with_pkgs = pythonForServer.withPackages (pp: [
                packages.nahual
                packages.stardist
                packages.csbdeep
                pp.tensorflow
                (pp.tf-keras.overridePythonAttrs (_: { dontCheckRuntimeDeps = true; }))
                pp.numpy
                pp.scikit-image
                pp.numba
                pp.imageio
                pp.tifffile
                pp.trio
                pp.scikit-learn
                pp.pyyaml
              ]);
            in
            mkShell {
              packages = [
                python_with_pkgs
                pkgs.cudaPackages.cudatoolkit
              ];
              shellHook = ''
                export TF_USE_LEGACY_KERAS=1
                export PYTHONPATH=${python_with_pkgs}/${python_with_pkgs.sitePackages}:$PYTHONPATH
              '';
            };
        };
      }
    );
}
