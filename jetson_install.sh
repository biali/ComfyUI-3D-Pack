#!/usr/bin/env bash
# ============================================================
# jetson_install.sh
# Full install script for ComfyUI-3D-Pack on
#   NVIDIA Jetson Orin Nano – JetPack 6.x
#   aarch64 / Python 3.12 / PyTorch 2.8 / CUDA 12.8
#
# Usage (from the ComfyUI-3D-Pack repo root):
#   chmod +x jetson_install.sh
#   ./jetson_install.sh
#
# Environment variables (optional overrides):
#   PYTHON        path to python binary  (default: python3)
#   COMFY3D_DIR   path to this repo      (default: auto-detected)
#   JOBS          parallel make jobs     (default: 4)
# ============================================================
set -euo pipefail

PYTHON="${PYTHON:-python3}"
COMFY3D_DIR="${COMFY3D_DIR:-$(cd "$(dirname "$0")" && pwd)}"
JOBS="${JOBS:-4}"

# Jetson Orin Nano – Ampere SM87
export TORCH_CUDA_ARCH_LIST="8.7"
export MAX_JOBS="${JOBS}"
export FORCE_CUDA=1

# Headless OpenGL via EGL (for nvdiffrast etc.)
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

# CUDA 12.8 from JetPack 6
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

echo "================================================================"
echo " ComfyUI-3D-Pack – Jetson Orin Nano install"
echo "  Python : $("${PYTHON}" --version 2>&1)"
echo "  CUDA   : $(nvcc --version 2>/dev/null | grep release || echo 'nvcc not found – using CUDA_HOME')"
echo "  Arch   : $(uname -m)"
echo "================================================================"

# ── 0. System prerequisites ─────────────────────────────────────────────
echo "[1/9] Checking system packages…"
PKGS_NEEDED=()
for pkg in build-essential git cmake ninja-build libgl1 libegl1 libglib2.0-0; do
    dpkg -s "$pkg" &>/dev/null || PKGS_NEEDED+=("$pkg")
done
if [ ${#PKGS_NEEDED[@]} -gt 0 ]; then
    echo "  Installing: ${PKGS_NEEDED[*]}"
    apt-get install -y "${PKGS_NEEDED[@]}"
fi

# ── 1. PyTorch stack (from NGC) ─────────────────────────────────────────
echo "[2/9] Installing PyTorch 2.8 / CUDA 12.8 for Jetson…"
NGC_INDEX="https://pypi.ngc.nvidia.com"

# Check if already installed at the right version
TORCH_OK=false
"${PYTHON}" -c "import torch; assert '2.8' in torch.__version__, torch.__version__" 2>/dev/null && TORCH_OK=true

if [ "$TORCH_OK" = "false" ]; then
    "${PYTHON}" -m pip install \
        --extra-index-url "${NGC_INDEX}" \
        "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
        || {
            echo "  NGC install failed – trying dusty-nv find-links…"
            "${PYTHON}" -m pip install \
                --find-links "https://github.com/dusty-nv/jetson-containers/raw/master/packages/pytorch/dist/" \
                "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0"
        }
else
    echo "  PyTorch 2.8 already installed – skipping."
fi

# ── 2. Base pip packages from requirements.txt ──────────────────────────
echo "[3/9] Installing base pip requirements…"
cd "${COMFY3D_DIR}"
"${PYTHON}" -m pip install \
    --extra-index-url "${NGC_INDEX}" \
    -r requirements.txt \
    || true   # non-fatal; some packages may not have aarch64 wheels

# ── 3. xformers (build from source) ─────────────────────────────────────
echo "[4/9] Building xformers from source (SM87)…"
XFORMERS_OK=false
"${PYTHON}" -c "import xformers" 2>/dev/null && XFORMERS_OK=true
if [ "$XFORMERS_OK" = "false" ]; then
    TMP_XF=$(mktemp -d)
    git clone --depth 1 --recurse-submodules \
        https://github.com/facebookresearch/xformers.git "${TMP_XF}/xformers"
    cd "${TMP_XF}/xformers"
    "${PYTHON}" -m pip install --no-build-isolation -e .
    cd "${COMFY3D_DIR}"
    rm -rf "${TMP_XF}"
else
    echo "  xformers already installed – skipping."
fi

# ── 4. cumm (dependency of spconv) ──────────────────────────────────────
echo "[5/9] Building cumm from source…"
CUMM_OK=false
"${PYTHON}" -c "import cumm" 2>/dev/null && CUMM_OK=true
if [ "$CUMM_OK" = "false" ]; then
    TMP_CUMM=$(mktemp -d)
    git clone --depth 1 https://github.com/FindDefinition/cumm.git "${TMP_CUMM}/cumm"
    cd "${TMP_CUMM}/cumm"
    # cumm picks up CUDA arch from TORCH_CUDA_ARCH_LIST
    "${PYTHON}" -m pip install --no-build-isolation .
    cd "${COMFY3D_DIR}"
    rm -rf "${TMP_CUMM}"
else
    echo "  cumm already installed – skipping."
fi

# ── 5. spconv (CUDA 12.8) ───────────────────────────────────────────────
echo "[6/9] Building spconv from source (cu128, SM87)…"
SPCONV_OK=false
"${PYTHON}" -c "import spconv" 2>/dev/null && SPCONV_OK=true
if [ "$SPCONV_OK" = "false" ]; then
    TMP_SP=$(mktemp -d)
    git clone --depth 1 --recurse-submodules \
        https://github.com/traveller59/spconv.git "${TMP_SP}/spconv"
    cd "${TMP_SP}/spconv"
    # Force spconv to treat this as cu128
    export SPCONV_DISABLE_JIT=1
    "${PYTHON}" -m pip install --no-build-isolation .
    cd "${COMFY3D_DIR}"
    rm -rf "${TMP_SP}"
else
    echo "  spconv already installed – skipping."
fi

# ── 6. open3d ───────────────────────────────────────────────────────────
echo "[7/9] Installing open3d…"
OPEN3D_OK=false
"${PYTHON}" -c "import open3d" 2>/dev/null && OPEN3D_OK=true
if [ "$OPEN3D_OK" = "false" ]; then
    # Try ARM64 wheel from open3d-arm project / pypi first
    "${PYTHON}" -m pip install open3d --find-links \
        "https://github.com/isl-org/Open3D/releases/" \
        || {
            echo "  Trying open3d-cpu as fallback…"
            "${PYTHON}" -m pip install open3d-cpu || true
        }
else
    echo "  open3d already installed – skipping."
fi

# ── 7. CUDA extensions used by 3DGS / NeRF nodes ────────────────────────
echo "[8/9] Building CUDA extensions from source…"

build_cuda_ext() {
    local NAME="$1"
    local REPO="$2"
    local EXTRA_ARGS="${3:-}"

    INSTALLED=false
    "${PYTHON}" -c "import ${NAME//-/_}" 2>/dev/null && INSTALLED=true
    if [ "$INSTALLED" = "false" ]; then
        echo "  Building ${NAME}…"
        TMP=$(mktemp -d)
        git clone --depth 1 --recurse-submodules "${REPO}" "${TMP}/${NAME}"
        cd "${TMP}/${NAME}"
        # shellcheck disable=SC2086
        "${PYTHON}" -m pip install --no-build-isolation ${EXTRA_ARGS} .
        cd "${COMFY3D_DIR}"
        rm -rf "${TMP}"
    else
        echo "  ${NAME} already installed – skipping."
    fi
}

# diff-gaussian-rasterization
build_cuda_ext "diff_gaussian_rasterization" \
    "https://github.com/graphdeco-inria/diff-gaussian-rasterization.git"

# simple-knn
build_cuda_ext "simple_knn" \
    "https://github.com/camenduru/simple-knn.git"

# pointnet2_ops
build_cuda_ext "pointnet2_ops" \
    "https://github.com/erikwijmans/Pointnet2_PyTorch.git"

# nvdiffrast (CUDA JIT; the pip package itself is pure-Python)
NVDIFF_OK=false
"${PYTHON}" -c "import nvdiffrast" 2>/dev/null && NVDIFF_OK=true
if [ "$NVDIFF_OK" = "false" ]; then
    "${PYTHON}" -m pip install --no-build-isolation \
        "git+https://github.com/NVlabs/nvdiffrast.git"
fi

# diso
DISO_OK=false
"${PYTHON}" -c "import diso" 2>/dev/null && DISO_OK=true
if [ "$DISO_OK" = "false" ]; then
    TMP_DISO=$(mktemp -d)
    git clone --depth 1 https://github.com/SarahWeiii/diso.git "${TMP_DISO}/diso"
    cd "${TMP_DISO}/diso"
    "${PYTHON}" -m pip install --no-build-isolation .
    cd "${COMFY3D_DIR}"
    rm -rf "${TMP_DISO}"
fi

# pytorch3d (optional but used by several nodes)
P3D_OK=false
"${PYTHON}" -c "import pytorch3d" 2>/dev/null && P3D_OK=true
if [ "$P3D_OK" = "false" ]; then
    echo "  Building pytorch3d (this takes a while)…"
    TMP_P3D=$(mktemp -d)
    git clone --depth 1 https://github.com/facebookresearch/pytorch3d.git "${TMP_P3D}/pytorch3d"
    cd "${TMP_P3D}/pytorch3d"
    "${PYTHON}" -m pip install --no-build-isolation .
    cd "${COMFY3D_DIR}"
    rm -rf "${TMP_P3D}"
fi

# ── 8. Run install.py (finalise, download pycpp stubs, etc.) ─────────────
echo "[9/9] Running install.py finalisation…"
"${PYTHON}" "${COMFY3D_DIR}/install.py"

echo ""
echo "================================================================"
echo " ComfyUI-3D-Pack Jetson install complete!"
echo " Restart ComfyUI and enjoy."
echo "================================================================"
