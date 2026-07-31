{
  description = "StarDist Nahual server (TensorFlow backend, GPU-enabled)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";
    systems.url = "github:nix-systems/default";
    flake-utils.url = "github:numtide/flake-utils";
    flake-utils.inputs.systems.follows = "systems";
    nahual-flake.url = "github:afermg/nahual";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
        pythonForServer = pkgs.python311;
        modelPackages = rec {
          pynng = pythonForServer.pkgs.callPackage ./nix/pynng.nix {};
          nahual = pythonForServer.pkgs.callPackage (inputs.nahual-flake + "/nix/nahual.nix") {
            inherit pynng;
          };
          csbdeep = pythonForServer.pkgs.callPackage ./nix/csbdeep.nix {
            tf-keras = pythonForServer.pkgs.tf-keras.overridePythonAttrs (_: {
              dontCheckRuntimeDeps = true;
            });
          };
          stardist = pythonForServer.pkgs.callPackage ./nix/stardist.nix {
            inherit csbdeep;
          };
        };
        python_with_pkgs = pythonForServer.withPackages (pp: [
          modelPackages.nahual
          modelPackages.stardist
          modelPackages.csbdeep
          pp.tensorflow
          (pp.tf-keras.overridePythonAttrs (_: {
            dontCheckRuntimeDeps = true;
          }))
          pp.numpy
          pp.scikit-image
          pp.numba
          pp.imageio
          pp.tifffile
          pp.trio
        ]);
        runServer = pkgs.writeScriptBin "nahual-stardist" ''
          #!${pkgs.bash}/bin/bash
          export TF_USE_LEGACY_KERAS=1
          export TF_FORCE_GPU_ALLOW_GROWTH=true
          export PYTHONSAFEPATH=1
          exec ${python_with_pkgs}/bin/python ${self}/server.py \
            "''${1:-tcp://0.0.0.0:5555}"
        '';
        stardistApp = {
          type = "app";
          program = "${runServer}/bin/nahual-stardist";
        };
      in
        with pkgs; rec {
          packages =
            modelPackages
            // pkgs.lib.optionalAttrs pkgs.stdenv.hostPlatform.isLinux {
              oci-image = import ./nix/oci-image.nix {
                inherit pkgs;
                name = "stardist";
                title = "Nahual StarDist";
                description = "StarDist instance segmentation served through Nahual";
                source = "https://github.com/afermg/stardist";
                revision = self.rev or self.dirtyRev or "unknown";
                server = runServer;
                entrypoint = stardistApp.program;
                extraEnv = [
                  "TF_USE_LEGACY_KERAS=1"
                  "TF_FORCE_GPU_ALLOW_GROWTH=true"
                  "TF_CPP_MIN_LOG_LEVEL=2"
                ];
              };
            };
          inherit pythonForServer python_with_pkgs;
          scripts.runServer = runServer;
          apps = rec {
            stardist = stardistApp;
            default = stardist;
          };
          devShells.default = mkShell {
            packages = [
              python_with_pkgs
              pkgs.cudaPackages.cudatoolkit
              pythonForServer.pkgs.scikit-learn
              pythonForServer.pkgs.pyyaml
            ];
            shellHook = ''
              export TF_USE_LEGACY_KERAS=1
              export TF_FORCE_GPU_ALLOW_GROWTH=true
              export PYTHONSAFEPATH=1
            '';
          };
        }
    );
}
