{
  lib,
  buildPythonPackage,
  fetchPypi,
  setuptools,
  numpy,
  scipy,
  matplotlib,
  six,
  tifffile,
  tqdm,
  packaging,
  tensorflow,
  tf-keras,
}:
buildPythonPackage rec {
  pname = "csbdeep";
  version = "0.8.2";
  format = "setuptools";

  src = fetchPypi {
    inherit pname version;
    sha256 = "sha256-3gGI051erq/gLTTFD8HNZ9kiuSBihyb73M3z/0ePXoY=";
  };

  # csbdeep does a raw `from keras import __version__` to detect Keras 2 vs 3.
  # We ship tf-keras 2.17 (the standalone Keras 2 maintained by Google for TF
  # >=2.16 compat) — but tf-keras installs as `tf_keras`, not `keras`.
  # Redirect csbdeep's keras-detection import to tf_keras so its TF 2.6+ code
  # path picks up our installed Keras 2 rather than blowing up.
  postPatch = ''
    substituteInPlace csbdeep/utils/tf.py \
      --replace-fail "from keras import __version__ as _v_keras" \
        "from tf_keras import __version__ as _v_keras" \
      --replace-fail "_KERAS = 'keras' if (IS_TF_1 or IS_KERAS_3_PLUS) else 'tensorflow.keras'" \
        "_KERAS = 'tf_keras'"
  '';

  build-system = [
    setuptools
  ];

  propagatedBuildInputs = [
    numpy
    scipy
    matplotlib
    six
    tifffile
    tqdm
    packaging
    tensorflow
    tf-keras
  ];

  # Tests need GPU + datasets + a configured TF env; skip on the build host.
  doCheck = false;
  pythonRuntimeDepsCheck = false;
  dontCheckRuntimeDeps = true;

  pythonImportsCheck = [
    # Don't import csbdeep here — it pulls in tensorflow at import time, which
    # tries to allocate CUDA in the sandbox. The runtime smoke test covers it.
  ];

  meta = {
    description = "Toolbox for content-aware image restoration (CARE)";
    homepage = "https://github.com/CSBDeep/CSBDeep";
    license = lib.licenses.bsd3;
  };
}
