"""
install.py – ComfyUI-3D-Pack
─────────────────────────────
Patched for NVIDIA Jetson Orin Nano
  arch   : aarch64 (linux-aarch64)
  Python : 3.12
  Torch  : 2.8.0  (CUDA 12.8)
  CUDA   : 12.8  (JetPack 6.x)

This script is called by ComfyUI-Manager & Comfy-CLI after requirements.txt
is installed:
  https://github.com/ltdrdata/ComfyUI-Manager/…#custom-node-support-guide

What changed vs upstream:
  1. CUDA 12.8 + aarch64 are recognised as a valid platform.
  2. Wheel download falls back to build-from-source for every package that
     has no pre-built aarch64 wheel (spconv, cumm, diso, open3d, etc.).
  3. torch/torchvision are pulled from the NGC index on aarch64.
  4. TORCH_CUDA_ARCH_LIST is forced to "8.7" (Orin SM87) before any CUDA
     extension is compiled.
  5. xformers is compiled from source on aarch64 (no official wheel).
  6. A platform_check() call warns early if the environment is unexpected.
"""

import sys
import os
from os.path import dirname
import glob
import subprocess
import traceback
import platform

# ── portable single-file invocation ──────────────────────────────────────
if sys.argv[0] == "install.py":
    sys.path.append(".")

COMFY3D_ROOT_ABS_PATH = dirname(os.path.abspath(__file__))
BUILD_SCRIPT_ROOT_ABS_PATH = os.path.join(
    COMFY3D_ROOT_ABS_PATH, "_Pre_Builds", "_Build_Scripts"
)
sys.path.insert(0, BUILD_SCRIPT_ROOT_ABS_PATH)

# ── Jetson / aarch64 detection ────────────────────────────────────────────
IS_AARCH64 = platform.machine().lower() in ("aarch64", "arm64")

# ── Force SM87 arch list before any C extension is built ─────────────────
if IS_AARCH64 and "TORCH_CUDA_ARCH_LIST" not in os.environ:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.7"
    print("[Comfy3D] Set TORCH_CUDA_ARCH_LIST=8.7 (Orin Nano SM87)")

try:
    from build_utils import (
        get_platform_config_name,
        get_current_platform_config,
        git_folder_parallel,
        install_remote_packages,
        install_platform_packages,
        install_isolated_packages,
        wheels_dir_exists_and_not_empty,
        build_config,
        PYTHON_PATH,
        WHEELS_ROOT_ABS_PATH,
        PYTHON_VERSION,
        CUDA_VERSION,
        IS_AARCH64 as _IS_AARCH64,
    )
    from shared_utils.log_utils import cstr

    # ── Ensure PyGithub is available ──────────────────────────────────────
    try:
        import github
    except ImportError:
        subprocess.run(
            [PYTHON_PATH, "-m", "pip", "install", "PyGithub"],
            check=True,
        )

    # ─────────────────────────────────────────────────────────────────────
    def platform_check():
        """Warn early if we are on an unexpected / unsupported config."""
        import torch
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"
        torch_ver = torch.__version__
        cuda_ver = CUDA_VERSION
        gpu = "unknown"
        try:
            gpu = torch.cuda.get_device_name(0)
        except Exception:
            pass

        cstr(f"Python Version : {py_ver}").msg.print()
        cstr(f"PyTorch        : {torch_ver}").msg.print()
        cstr(f"CUDA           : {cuda_ver}").msg.print()
        cstr(f"Platform       : {platform.machine()} / {platform.system()}").msg.print()
        cstr(f"GPU            : {gpu}").msg.print()
        cstr(f"Config key     : {get_platform_config_name()}").msg.print()

        if IS_AARCH64:
            cstr("Running on aarch64 – Jetson-specific build path activated.").msg.print()
            if cuda_ver not in ("12.8",):
                cstr(
                    f"[WARN] Expected CUDA 12.8 on Jetson, got {cuda_ver}. "
                    "Some pre-built wheels may not match."
                ).warning.print()

    # ─────────────────────────────────────────────────────────────────────
    def install_local_wheels(builds_dir: str) -> bool:
        """Install all .whl files found under builds_dir."""
        wheel_files = glob.glob(
            os.path.join(builds_dir, "**", "*.whl"), recursive=True
        )
        if not wheel_files:
            cstr("No wheel files found in directory").warning.print()
            return False

        success_count = 0
        for wheel_path in wheel_files:
            result = subprocess.run(
                [
                    PYTHON_PATH, "-s", "-m", "pip", "install",
                    "--no-deps", "--force-reinstall", wheel_path,
                ],
                text=True, capture_output=True,
            )
            if result.returncode == 0:
                cstr(
                    f"Installed wheel: {os.path.basename(wheel_path)}"
                ).msg.print()
                success_count += 1
            else:
                cstr(
                    f"Failed to install {os.path.basename(wheel_path)}: "
                    f"{result.stderr[:300]}"
                ).error.print()

        if success_count == len(wheel_files):
            cstr(f"Installed all {len(wheel_files)} wheels").msg.print()
            return True
        elif success_count > 0:
            cstr(
                f"Partially installed: {success_count}/{len(wheel_files)}"
            ).warning.print()
            return False
        else:
            cstr("Failed to install any wheels").error.print()
            return False

    # ─────────────────────────────────────────────────────────────────────
    def try_wheels_first_approach() -> bool:
        """
        Try installing from pre-built wheels.
        On aarch64 the remote repo will almost certainly not have wheels,
        so this returns False quickly and we fall through to build-from-source.
        """
        platform_config_name = get_platform_config_name()
        builds_dir = os.path.join(WHEELS_ROOT_ABS_PATH, platform_config_name)

        cstr(f"Trying wheels-first approach ({platform_config_name})…").msg.print()

        # Check local cache
        if wheels_dir_exists_and_not_empty(builds_dir):
            cstr(f"Found cached wheels in {builds_dir}").msg.print()
            if install_local_wheels(builds_dir):
                return True

        # Try downloading from GitHub
        remote_path = f"{build_config.wheels_dir_name}/{platform_config_name}"
        cstr(f"Attempting to download wheels from {build_config.repo_id}…").msg.print()
        if git_folder_parallel(
            build_config.repo_id, remote_path, recursive=True, root_outdir=builds_dir
        ):
            cstr("Downloaded wheels from repository").msg.print()
            if install_local_wheels(builds_dir):
                return True
        else:
            cstr(
                "No pre-built wheels found for this platform "
                f"({platform_config_name}). Will build from source."
            ).warning.print()

        return False

    # ─────────────────────────────────────────────────────────────────────
    def try_auto_build_all(builds_dir: str) -> bool:
        """Trigger auto_build_all.py to compile all CUDA extensions."""
        cstr("Building all required packages from source…").msg.print()

        env = os.environ.copy()
        if IS_AARCH64:
            env.setdefault("TORCH_CUDA_ARCH_LIST", "8.7")
            env.setdefault("MAX_JOBS", "4")  # Orin has limited RAM

        result = subprocess.run(
            [PYTHON_PATH, "auto_build_all.py", "--output_root_dir", builds_dir],
            cwd=BUILD_SCRIPT_ROOT_ABS_PATH,
            text=True,
            capture_output=True,
            env=env,
        )
        cstr(f"[BUILD LOG]\n{result.stdout}").msg.print()
        if result.returncode != 0:
            cstr(f"[BUILD ERROR]\n{result.stderr}").error.print()
        return result.returncode == 0

    # ─────────────────────────────────────────────────────────────────────
    def install_jetson_torch_stack():
        """
        On aarch64, pull torch/torchvision/torchaudio from the NGC index.
        Skipped on x86_64 (handled by install_platform_packages).
        """
        if not IS_AARCH64:
            return

        NGC_INDEX = "https://pypi.ngc.nvidia.com"
        pkgs = [
            f"torch==2.8.0",
            f"torchvision==0.23.0",
            f"torchaudio==2.8.0",
        ]

        cstr("Installing PyTorch stack from NGC index for Jetson…").msg.print()
        result = subprocess.run(
            [
                PYTHON_PATH, "-m", "pip", "install",
                "--extra-index-url", NGC_INDEX,
                *pkgs,
            ],
            text=True, capture_output=True,
        )
        if result.returncode != 0:
            cstr(
                f"NGC torch install failed – trying dusty-nv wheels…\n"
                f"{result.stderr[:400]}"
            ).warning.print()
            # Fallback: dusty-nv / jetson-containers pip index
            DUSTY_INDEX = "https://github.com/dusty-nv/jetson-containers/raw/master/packages/pytorch/dist/"
            result2 = subprocess.run(
                [
                    PYTHON_PATH, "-m", "pip", "install",
                    "--find-links", DUSTY_INDEX,
                    *pkgs,
                ],
                text=True, capture_output=True,
            )
            if result2.returncode != 0:
                cstr(
                    "Could not auto-install torch for Jetson. "
                    "Please install manually from https://pypi.ngc.nvidia.com "
                    "or via jetson-containers before running this script."
                ).error.print()
            else:
                cstr("Installed torch from dusty-nv fallback index.").msg.print()
        else:
            cstr("Installed PyTorch stack from NGC.").msg.print()

    # ─────────────────────────────────────────────────────────────────────
    def install_build_tools():
        """Make sure cmake, ninja, setuptools, wheel are present."""
        cstr("Checking build tools…").msg.print()
        for tool in ["ninja", "cmake", "setuptools", "wheel"]:
            try:
                __import__(tool)
                cstr(f"  {tool} already installed").msg.print()
            except ImportError:
                cstr(f"  Installing {tool}…").msg.print()
                r = subprocess.run(
                    [PYTHON_PATH, "-m", "pip", "install", "--upgrade", tool],
                    text=True, capture_output=True,
                )
                if r.returncode != 0:
                    raise RuntimeError(
                        f"Failed to install build tool {tool}:\n{r.stderr}"
                    )

    # ══════════════════════════════════════════════════════════════════════
    # MAIN INSTALL FLOW
    # ══════════════════════════════════════════════════════════════════════

    platform_check()
    install_build_tools()

    # Step 1: install PyTorch stack (Jetson-specific or normal)
    if IS_AARCH64:
        install_jetson_torch_stack()
    else:
        install_remote_packages(build_config.build_base_packages)

    # Step 2: install any extra platform-specific packages
    install_platform_packages()

    # Step 3: install packages that need --no-build-isolation
    plat_cfg = get_current_platform_config()
    isolated = plat_cfg.get("isolated_packages", []) + list(
        getattr(build_config, "isolated_packages", [])
    )
    if isolated:
        install_isolated_packages(isolated)

    # Step 4: try pre-built wheels → fall back to source build
    platform_config_name = get_platform_config_name()
    builds_dir = os.path.join(WHEELS_ROOT_ABS_PATH, platform_config_name)

    wheels_success = try_wheels_first_approach()

    if not wheels_success:
        cstr(
            "No pre-built wheels available for this platform – "
            "compiling CUDA extensions from source (this may take 30–60 min on Jetson)…"
        ).warning.print()
        if try_auto_build_all(builds_dir):
            install_local_wheels(builds_dir)
            wheels_success = True
        else:
            cstr(
                "Source build also failed. "
                "See _Pre_Builds/README.md for manual build instructions."
            ).error.print()

    # Step 5: download matching Python C++ source stubs
    remote_pycpp_dir = f"_Python_Source_cpp/{PYTHON_VERSION}"
    python_root_dir = dirname(PYTHON_PATH)
    if git_folder_parallel(
        build_config.repo_id,
        remote_pycpp_dir,
        recursive=True,
        root_outdir=python_root_dir,
    ):
        cstr("Downloaded Python C++ source files.").msg.print()
    else:
        cstr(
            f"[WARN] Couldn't download {remote_pycpp_dir} – some nodes may not work."
        ).warning.print()

    cstr("Comfy3D install finished! Let's Accelerate! 🚀").msg.print()

except Exception:
    traceback.print_exc()
    try:
        from shared_utils.log_utils import cstr
        cstr(
            "Comfy3D install failed. "
            "See https://github.com/MrForExample/ComfyUI-3D-Pack/tree/main/_Pre_Builds/README.md "
            "for manual installation instructions."
        ).error.print()
    except Exception:
        print(
            "[ERROR] Comfy3D install failed – check the traceback above."
        )
