{
  lib,
  buildPythonPackage,
  setuptools,
  wheel,
  numpy,
  scikit-image,
  numba,
  imageio,
  csbdeep,
  llvmPackages,
}:
buildPythonPackage {
  pname = "stardist";
  # Match upstream version.py.
  version = "0.9.2";
  format = "setuptools";

  # Local source — afermg/stardist nahual-wrap branch.
  src = ./..;

  # OpenMP support requires libomp (the setup.py probes for -fopenmp).
  nativeBuildInputs = [
    llvmPackages.openmp
  ];

  build-system = [
    setuptools
    wheel
  ];

  propagatedBuildInputs = [
    csbdeep
    numpy
    scikit-image
    numba
    imageio
  ];

  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  pythonImportsCheck = [
    # Skip — importing stardist drags in tensorflow/csbdeep at module level.
  ];

  meta = {
    description = "StarDist - Object Detection with Star-convex Shapes";
    homepage = "https://github.com/stardist/stardist";
    license = lib.licenses.bsd3;
  };
}
