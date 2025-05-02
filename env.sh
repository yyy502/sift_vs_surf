#!/usr/bin/env bash
set -euo pipefail

# --- enable `conda activate` inside scripts ---
source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -n cvbuild python=3.10 -y
conda activate cvbuild
# --- core build tools and runtime libs ---
conda install -c conda-forge rhash ninja -y               # install rhash and Ninja
# Install pip-provided CMake binary (bundled, avoids dylib deps)
pip install cmake
# macOS: make sure loader can see conda libs at runtime
export DYLD_FALLBACK_LIBRARY_PATH="${CONDA_PREFIX}/lib:${DYLD_FALLBACK_LIBRARY_PATH:-}"
# Ensure cmake sees librhash at runtime on macOS
export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${DYLD_LIBRARY_PATH:-}"
# Also include Homebrew library paths for librhash
export DYLD_LIBRARY_PATH="/usr/local/lib:/opt/homebrew/lib:${DYLD_LIBRARY_PATH}"
export DYLD_FALLBACK_LIBRARY_PATH="/usr/local/lib:/opt/homebrew/lib:${DYLD_FALLBACK_LIBRARY_PATH}"

conda install -c conda-forge numpy matplotlib libjpeg-turbo eigen ffmpeg -y

git clone --depth 1 https://github.com/opencv/opencv.git
git clone --depth 1 https://github.com/opencv/opencv_contrib.git

 # Build inside the cloned opencv directory
cd opencv
mkdir -p build && cd build
cmake -GNinja \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
  -D OPENCV_ENABLE_NONFREE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -D BUILD_opencv_python3=ON \
  -D PYTHON_EXECUTABLE=$(which python) \
  ../

ninja
ninja install    # copies cv2 .so / .pyd into the env’s site‑packages

python -c "import cv2; print(cv2.__version__); cv2.xfeatures2d.SURF_create(); print('SURF OK')"