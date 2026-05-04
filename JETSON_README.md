# ComfyUI-3D-Pack – Jetson Orin Nano Port

Patches and helper scripts to make **ComfyUI-3D-Pack** work on:

| Target | Value |
|--------|-------|
| Board | NVIDIA Jetson Orin Nano |
| OS / arch | Ubuntu 22.04 aarch64 (JetPack 6.x) |
| Python | 3.12 |
| PyTorch | 2.8.0 |
| CUDA | 12.8 |
| CUDA Arch | SM 8.7 (Ampere) |

---

## Files in this patch

| File | Purpose |
|------|---------|
| `requirements.txt` | Base pip requirements – drops x86-only binary deps, adds git-source installs |
| `install.py` | Patched main install script – detects aarch64, sets SM87, routes to Jetson NGC index |
| `jetson_install.sh` | **One-shot install helper** – runs everything in the right order |
| `jetson_aarch64/_Pre_Builds/_Build_Scripts/build_config.yaml` | Adds `linux_aarch64_py312_cu128` platform block |
| `jetson_aarch64/_Pre_Builds/_Build_Scripts/build_utils.py` | Patched build utils – multi-path CUDA detection, aarch64 platform naming |

---

## Quick Start

### 1. Clone the original repo

```bash
cd <your-ComfyUI>/custom_nodes/
git clone https://github.com/MrForExample/ComfyUI-3D-Pack.git
cd ComfyUI-3D-Pack
```

### 2. Apply the patch files

Copy all files from this patch directory **over** the cloned repo:

```bash
# From inside your ComfyUI-3D-Pack directory:
cp /path/to/patch/requirements.txt  .
cp /path/to/patch/install.py        .
cp /path/to/patch/jetson_install.sh .
cp /path/to/patch/jetson_aarch64/_Pre_Builds/_Build_Scripts/build_config.yaml \
       _Pre_Builds/_Build_Scripts/build_config.yaml
cp /path/to/patch/jetson_aarch64/_Pre_Builds/_Build_Scripts/build_utils.py \
       _Pre_Builds/_Build_Scripts/build_utils.py
```

### 3. Run the installer

```bash
chmod +x jetson_install.sh
./jetson_install.sh
```

The script will take **30–90 minutes** on first run because it compiles
CUDA extensions (xformers, spconv, diff-gaussian-rasterization, pytorch3d, …)
from source.

---

## Manual step-by-step (if the script fails at a specific stage)

### System prereqs

```bash
sudo apt-get install -y \
    build-essential git cmake ninja-build \
    libgl1 libegl1 libglib2.0-0 \
    python3-dev
```

### Environment variables (set before every build)

```bash
export TORCH_CUDA_ARCH_LIST="8.7"   # Orin Nano SM87
export MAX_JOBS=4                    # keep RAM in check
export FORCE_CUDA=1
export PYOPENGL_PLATFORM=egl        # headless rendering
export CUDA_HOME=/usr/local/cuda
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
```

### PyTorch 2.8 (from NGC)

```bash
pip install \
    --extra-index-url https://pypi.ngc.nvidia.com \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
```

> **Tip:** If the NGC wheel download is slow, the
> [dusty-nv / jetson-containers](https://github.com/dusty-nv/jetson-containers)
> project also provides PyTorch wheels via `--find-links`.

### xformers (build from source)

```bash
git clone --recurse-submodules https://github.com/facebookresearch/xformers.git
cd xformers
pip install --no-build-isolation -e .
cd ..
```

### cumm + spconv

```bash
# cumm first (spconv depends on it)
git clone https://github.com/FindDefinition/cumm.git
pip install --no-build-isolation ./cumm

# spconv
git clone --recurse-submodules https://github.com/traveller59/spconv.git
cd spconv
SPCONV_DISABLE_JIT=1 pip install --no-build-isolation .
cd ..
```

### diso

```bash
git clone https://github.com/SarahWeiii/diso.git
pip install --no-build-isolation ./diso
```

### open3d

```bash
# Attempt binary first:
pip install open3d
# If that fails (no aarch64 wheel), use CPU-only variant:
pip install open3d-cpu
```

### nvdiffrast

```bash
pip install --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git
```

### Gaussian-splatting CUDA extensions

```bash
# diff-gaussian-rasterization
git clone --recurse-submodules \
    https://github.com/graphdeco-inria/diff-gaussian-rasterization.git
pip install --no-build-isolation ./diff-gaussian-rasterization

# simple-knn
git clone https://github.com/camenduru/simple-knn.git
pip install --no-build-isolation ./simple-knn

# pointnet2_ops
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
pip install --no-build-isolation ./Pointnet2_PyTorch
```

### pytorch3d (optional, large build)

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation ./pytorch3d
```

### Remaining pip deps

```bash
pip install -r requirements.txt
```

### Finalise

```bash
python install.py
```

---

## Known limitations on Jetson Orin Nano

| Limitation | Notes |
|------------|-------|
| **VRAM** | Orin Nano has 8 GB unified LPDDR5 (shared CPU+GPU). Large models (Era3D, CRM, etc.) may OOM. Use `--lowvram` and avoid running other apps. |
| **No Flash Attention** | xformers Flash Attention requires Hopper (SM90+). Falls back to standard attention automatically. |
| **open3d headless** | Requires EGL. Set `PYOPENGL_PLATFORM=egl` before starting ComfyUI. |
| **Build time** | Full source build ~60–90 min. Pre-built wheels are not yet available for `linux_aarch64_py312_cu128`. |
| **xformers** | Memory-efficient attention still works on SM87; Flash Attention ops are silently disabled. |

---

## Verifying the install

After installing, run:

```bash
python - <<'EOF'
import torch, sys, platform
print(f"Python  : {sys.version}")
print(f"Arch    : {platform.machine()}")
print(f"Torch   : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"Device  : {torch.cuda.get_device_name(0)}")

for pkg in ["spconv", "nvdiffrast", "diso", "open3d",
            "diff_gaussian_rasterization", "simple_knn",
            "pointnet2_ops", "xformers"]:
    try:
        __import__(pkg)
        print(f"  OK : {pkg}")
    except ImportError as e:
        print(f"  MISSING : {pkg} – {e}")
EOF
```

---

## Contributing

If you build working aarch64 wheels, please consider uploading them to the
`MrForExample/Comfy3D_Pre_Builds` repository under the path:

```
_Build_Wheels/linux_aarch64_py312_cu128/<package>.whl
```

That will let future Jetson users skip the long source compilation.
