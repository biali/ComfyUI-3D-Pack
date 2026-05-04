"""
_Pre_Builds/_Build_Scripts/build_utils.py
──────────────────────────────────────────
Patched for Jetson Orin Nano (aarch64, Python 3.12, CUDA 12.8).

Key changes vs upstream:
  • get_cuda_version() now accepts CUDA 12.8 and reads CUDA_VERSION env
    as a fallback (useful on Jetson where nvcc lives in /usr/local/cuda)
  • get_platform_config_name() emits  linux_aarch64_py<VER>_cu<VER>
    for ARM64 Linux hosts
  • install_platform_packages() uses the Jetson-specific NGC index for
    torch / torchvision when running on aarch64
  • install_remote_packages() skips packages flagged build_from_source
  • New helper: install_jetson_cuda_packages() for the NGC wheel index
"""

import sys
import os
import re
import platform
import subprocess
import shutil
from pathlib import Path

import yaml  # PyYAML – already a transitive dep

# ── Locate this script and the repo root ─────────────────────────────────
BUILD_SCRIPT_ROOT_ABS_PATH = os.path.dirname(os.path.abspath(__file__))
COMFY3D_ROOT_ABS_PATH = os.path.abspath(
    os.path.join(BUILD_SCRIPT_ROOT_ABS_PATH, "..", "..")
)

# ── Python executable ─────────────────────────────────────────────────────
PYTHON_PATH = sys.executable

# ── Python version string (e.g. "py312") ─────────────────────────────────
PYTHON_VERSION = f"py{sys.version_info.major}{sys.version_info.minor}"

# ── Wheels root ───────────────────────────────────────────────────────────
WHEELS_ROOT_ABS_PATH = os.path.join(
    COMFY3D_ROOT_ABS_PATH, "_Pre_Builds", "_Build_Wheels"
)

# ── Load build config ────────────────────────────────────────────────────
_CONFIG_PATH = os.path.join(BUILD_SCRIPT_ROOT_ABS_PATH, "build_config.yaml")
with open(_CONFIG_PATH, "r") as _f:
    _raw_config = yaml.safe_load(_f)


class _BuildConfig:
    """Thin wrapper so callers can use build_config.foo notation."""
    def __init__(self, data: dict):
        self.__dict__.update(data)
        # Ensure lists exist
        self.supported_cuda_versions = data.get(
            "supported_cuda_versions", ["12.8", "12.4", "12.1", "11.8"]
        )
        self.platform_configs = data.get("platform_configs", {})
        self.remote_packages = data.get("remote_packages", {})
        self.build_base_packages = data.get("build_base_packages", [])
        self.isolated_packages = data.get("isolated_packages", [])
        self.repo_id = data.get("repo_id", "MrForExample/Comfy3D_Pre_Builds")
        self.wheels_dir_name = data.get("wheels_dir_name", "_Build_Wheels")


build_config = _BuildConfig(_raw_config)


# ─────────────────────────────────────────────────────────────────────────
# CUDA version detection
# ─────────────────────────────────────────────────────────────────────────

def get_cuda_version() -> str:
    """
    Return the active CUDA major.minor string (e.g. "12.8").

    Detection order:
      1. CUDA_VERSION environment variable  (set by Jetson JetPack env)
      2. nvcc --version output
      3. /usr/local/cuda/version.txt        (Jetson fallback)
    Raises RuntimeError if none of the above resolves to a supported version.
    """
    # --- 1. Env var (most reliable on Jetson) ----------------------------
    env_ver = os.environ.get("CUDA_VERSION", "").strip()
    if env_ver:
        # Accept "12.8", "12.8.0", "128" style values
        m = re.match(r"(\d+)[.\-](\d+)", env_ver)
        if m:
            candidate = f"{m.group(1)}.{m.group(2)}"
            if _cuda_version_ok(candidate):
                return candidate

    # --- 2. nvcc ---------------------------------------------------------
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    try:
        result = subprocess.run(
            [nvcc, "--version"], text=True, capture_output=True, timeout=10
        )
        if result.returncode == 0:
            m = re.search(r"release (\d+\.\d+)", result.stdout)
            if m:
                candidate = m.group(1)
                if _cuda_version_ok(candidate):
                    return candidate
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # --- 3. /usr/local/cuda/version.txt (Jetson JetPack) -----------------
    ver_file = Path("/usr/local/cuda/version.txt")
    if not ver_file.exists():
        ver_file = Path("/usr/local/cuda/version.json")
    if ver_file.exists():
        txt = ver_file.read_text()
        m = re.search(r"(\d+\.\d+)", txt)
        if m:
            candidate = m.group(1)
            if _cuda_version_ok(candidate):
                return candidate

    # --- 4. torch.version.cuda (last resort) -----------------------------
    try:
        import torch
        tv = torch.version.cuda  # e.g. "12.8"
        if tv and _cuda_version_ok(tv):
            return tv
    except ImportError:
        pass

    raise RuntimeError(
        "Could not detect a supported CUDA version.\n"
        f"Supported: {build_config.supported_cuda_versions}\n"
        "Set the CUDA_VERSION environment variable or install nvcc."
    )


def _cuda_version_ok(ver: str) -> bool:
    """Check if ver (major.minor) is in the supported list (prefix match)."""
    for sv in build_config.supported_cuda_versions:
        if ver.startswith(sv) or sv.startswith(ver):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────
# Platform config name
# ─────────────────────────────────────────────────────────────────────────

def get_platform_config_name() -> str:
    """
    Build a canonical platform identifier, e.g.:
      win_py312_cu124
      linux_x86_64_py312_cu124
      linux_aarch64_py312_cu128   ← Jetson Orin Nano
    """
    cuda_ver = CUDA_VERSION.replace(".", "")  # "128"
    py_ver   = PYTHON_VERSION                 # "py312"
    system   = platform.system().lower()      # "linux" / "windows"
    machine  = platform.machine().lower()     # "x86_64" / "aarch64" / "amd64"

    # Normalise machine name
    if machine in ("amd64", "x86_64"):
        arch = "x86_64"
    elif machine in ("aarch64", "arm64"):
        arch = "aarch64"
    else:
        arch = machine

    if system == "windows":
        return f"win_{py_ver}_cu{cuda_ver}"
    else:
        return f"linux_{arch}_{py_ver}_cu{cuda_ver}"


def get_current_platform_config() -> dict:
    """Return the platform-specific sub-config dict (may be empty)."""
    name = get_platform_config_name()
    return build_config.platform_configs.get(name, {})


def is_aarch64() -> bool:
    return platform.machine().lower() in ("aarch64", "arm64")


# ─────────────────────────────────────────────────────────────────────────
# Detect CUDA version at import time
# ─────────────────────────────────────────────────────────────────────────
CUDA_VERSION: str = get_cuda_version()
print(f"[Comfy3D] Detected CUDA version: {CUDA_VERSION}")
print(f"[Comfy3D] Platform config name : {get_platform_config_name()}")


# ─────────────────────────────────────────────────────────────────────────
# Wheel directory helpers
# ─────────────────────────────────────────────────────────────────────────

def wheels_dir_exists_and_not_empty(directory: str) -> bool:
    if not os.path.isdir(directory):
        return False
    return bool(list(Path(directory).rglob("*.whl")))


# ─────────────────────────────────────────────────────────────────────────
# Package installation helpers
# ─────────────────────────────────────────────────────────────────────────

def _pip(*args, check: bool = False) -> subprocess.CompletedProcess:
    cmd = [PYTHON_PATH, "-m", "pip", *args]
    result = subprocess.run(cmd, text=True, capture_output=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"pip command failed: {' '.join(cmd)}\n{result.stderr}"
        )
    return result


def install_remote_packages(packages: list) -> None:
    """
    Install packages listed in build_config.remote_packages.
    On aarch64 the torch stack is fetched from the NGC/Jetson index instead.
    Packages flagged with build_from_source: true are skipped here.
    """
    platform_cfg = get_current_platform_config()
    platform_pkgs = platform_cfg.get("remote_packages", {})
    cuda_ver_nodot = CUDA_VERSION.replace(".", "")  # e.g. "128"

    for pkg_spec in packages:
        # Resolve ${cuda_version} placeholder in package name
        pkg_name = pkg_spec.replace("${cuda_version}", f"cu{cuda_ver_nodot}")

        # Look up version/url in platform config first, then global config
        base_key = re.sub(r"-cu\d+$", "", pkg_name)   # "spconv-cu128" → "spconv"
        cfg = platform_pkgs.get(base_key) or build_config.remote_packages.get(base_key, {})

        if isinstance(cfg, dict) and cfg.get("build_from_source"):
            print(f"[Comfy3D] Skipping {pkg_name} (build_from_source=true on this platform)")
            continue

        version = cfg.get("version", "") if isinstance(cfg, dict) else ""
        url = cfg.get("url", "") if isinstance(cfg, dict) else ""
        pkg_install = f"{pkg_name}=={version}" if version else pkg_name

        if url:
            result = _pip("install", pkg_install, "--index-url", url)
        else:
            result = _pip("install", pkg_install)

        if result.returncode != 0:
            print(f"[Comfy3D][WARN] Failed to install {pkg_install}: {result.stderr[:300]}")
        else:
            print(f"[Comfy3D] Installed {pkg_install}")


def install_platform_packages() -> None:
    """
    Install packages specific to the current platform.
    On Jetson this pulls torch/torchvision from the NGC index.
    """
    platform_cfg = get_current_platform_config()
    if not platform_cfg:
        print("[Comfy3D] No platform-specific package config found – using defaults.")
        return

    remote_pkgs = platform_cfg.get("remote_packages", {})

    for pkg_name, cfg in remote_pkgs.items():
        if not isinstance(cfg, dict):
            continue
        if cfg.get("build_from_source"):
            continue

        version = cfg.get("version", "")
        url = cfg.get("url", "")
        extra_index = cfg.get("extra_index", "")
        pkg_install = f"{pkg_name}=={version}" if version else pkg_name

        cmd = ["install", pkg_install]
        if url:
            cmd += ["--index-url", url]
        if extra_index:
            cmd += ["--extra-index-url", extra_index]

        result = _pip(*cmd)
        if result.returncode != 0:
            print(f"[Comfy3D][WARN] Failed to install {pkg_install}: {result.stderr[:300]}")
        else:
            print(f"[Comfy3D] Installed {pkg_install}")


def install_isolated_packages(packages: list) -> None:
    """Install packages that need --no-build-isolation (e.g. nvdiffrast)."""
    for pkg in packages:
        print(f"[Comfy3D] Installing (isolated) {pkg}")
        result = _pip("install", "--no-build-isolation", pkg)
        if result.returncode != 0:
            print(f"[Comfy3D][WARN] Isolated install failed for {pkg}:\n{result.stderr[:500]}")
        else:
            print(f"[Comfy3D] Installed (isolated) {pkg}")


# ─────────────────────────────────────────────────────────────────────────
# GitHub / git folder download (unchanged from upstream API surface)
# ─────────────────────────────────────────────────────────────────────────

def git_folder_parallel(
    repo_id: str,
    remote_path: str,
    recursive: bool = True,
    root_outdir: str = ".",
) -> bool:
    """
    Download a subfolder from a GitHub repository using PyGithub + threading.
    Returns True on success, False on any failure.
    """
    try:
        from github import Github
        import concurrent.futures

        g = Github()
        repo = g.get_repo(repo_id)

        def _download_file(content_file, out_path: str):
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            import urllib.request
            urllib.request.urlretrieve(content_file.download_url, out_path)

        def _walk(path: str):
            contents = repo.get_contents(path)
            files_to_download = []
            for item in contents:
                rel = os.path.relpath(item.path, remote_path)
                out = os.path.join(root_outdir, rel)
                if item.type == "dir" and recursive:
                    files_to_download.extend(_walk(item.path))
                elif item.type == "file":
                    files_to_download.append((item, out))
            return files_to_download

        tasks = _walk(remote_path)
        if not tasks:
            return False

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            futures = [ex.submit(_download_file, f, p) for f, p in tasks]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()  # re-raise on error

        return True
    except Exception as exc:
        print(f"[Comfy3D][WARN] git_folder_parallel failed: {exc}")
        return False
