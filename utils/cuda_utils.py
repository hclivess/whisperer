"""
CUDA helpers for faster-whisper / CTranslate2.

CTranslate2 loads cuBLAS 12 and cuDNN 9 at runtime (dlopen / LoadLibrary). Python users get them with
    pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
but the libraries still have to be discoverable. This module finds them (pip packages, CUDA_PATH,
a `cuda` folder next to the app, or a user-configured folder) and makes them loadable before the
first model is created, so the app works on a GPU without touching PATH / LD_LIBRARY_PATH.
"""
import ctypes
import glob
import os
import subprocess
import sys
from typing import Dict, List, Optional

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
_done_dirs: List[str] = []
_preloaded: Dict[str, bool] = {}


def _app_dir() -> str:
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return os.path.dirname(os.path.abspath(sys.argv[0]))
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _site_packages_dirs() -> List[str]:
    dirs = []
    for p in sys.path:
        if p and os.path.isdir(p) and p.rstrip("/\\").endswith(("site-packages", "dist-packages")):
            dirs.append(p)
    try:
        import site
        dirs += [d for d in site.getsitepackages() if os.path.isdir(d)]
        usp = site.getusersitepackages()
        if os.path.isdir(usp):
            dirs.append(usp)
    except Exception:
        pass
    return list(dict.fromkeys(dirs))


def candidate_lib_dirs(extra_dir: str = "") -> List[str]:
    """All folders that may contain cuBLAS / cuDNN, most specific first"""
    dirs: List[str] = []
    if extra_dir:
        dirs.append(extra_dir)
        for sub in ("bin", "lib", "lib64", os.path.join("lib", "x64")):
            dirs.append(os.path.join(extra_dir, sub))
    app = _app_dir()
    dirs += [os.path.join(app, "cuda"), app]
    # nvidia pip wheels: site-packages/nvidia/<pkg>/{lib,bin}
    for sp in _site_packages_dirs():
        for pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc", "cufft", "curand"):
            for sub in ("lib", "bin"):
                dirs.append(os.path.join(sp, "nvidia", pkg, sub))
        dirs.append(os.path.join(sp, "torch", "lib"))           # torch ships the same libs
        dirs.append(os.path.join(sp, "ctranslate2.libs"))
    for env in ("CUDA_PATH", "CUDA_HOME", "CUDNN_PATH"):
        root = os.environ.get(env)
        if root:
            dirs += [os.path.join(root, "bin"), os.path.join(root, "lib64"), os.path.join(root, "lib"),
                     os.path.join(root, "lib", "x64")]
    if sys.platform.startswith("linux"):
        dirs += ["/usr/local/cuda/lib64", "/usr/local/cuda/targets/x86_64-linux/lib",
                 "/usr/lib/x86_64-linux-gnu", "/usr/lib64"]
    return [d for d in dict.fromkeys(os.path.normpath(d) for d in dirs) if os.path.isdir(d)]


def _lib_patterns() -> Dict[str, List[str]]:
    """library name -> glob patterns (the order matters: cublasLt before cublas before cudnn)"""
    if sys.platform == "win32":
        return {"cublasLt": ["cublasLt64_12.dll"], "cublas": ["cublas64_12.dll"],
                "cudnn": ["cudnn64_9.dll"]}
    if sys.platform == "darwin":
        return {}
    return {"cublasLt": ["libcublasLt.so.12", "libcublasLt.so.12.*"],
            "cublas": ["libcublas.so.12", "libcublas.so.12.*"],
            "cudnn": ["libcudnn.so.9", "libcudnn.so.9.*"]}


def find_libraries(extra_dir: str = "") -> Dict[str, Optional[str]]:
    """Locate each required library; value is the full path or None"""
    found: Dict[str, Optional[str]] = {}
    dirs = candidate_lib_dirs(extra_dir)
    for name, patterns in _lib_patterns().items():
        found[name] = None
        for d in dirs:
            for pat in patterns:
                hits = sorted(glob.glob(os.path.join(d, pat)))
                if hits:
                    found[name] = hits[0]
                    break
            if found[name]:
                break
    return found


def setup_cuda(extra_dir: str = "") -> Dict[str, Optional[str]]:
    """
    Make the CUDA runtime libraries loadable for CTranslate2. Safe to call repeatedly.
    Returns the dict from find_libraries().
    """
    libs = find_libraries(extra_dir)
    dirs = {os.path.dirname(p) for p in libs.values() if p}
    for d in sorted(dirs):
        if d in _done_dirs:
            continue
        _done_dirs.append(d)
        if sys.platform == "win32":
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
        else:
            os.environ["LD_LIBRARY_PATH"] = d + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    if sys.platform.startswith("linux"):
        # dlopen by soname finds already-loaded libraries first, so preloading with RTLD_GLOBAL
        # lets CTranslate2 resolve them regardless of LD_LIBRARY_PATH at process start.
        for name in ("cublasLt", "cublas", "cudnn"):
            path = libs.get(name)
            if path and not _preloaded.get(path):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    _preloaded[path] = True
                except OSError:
                    _preloaded[path] = False
        # cuDNN 9 sub-libraries and nvrtc live next to libcudnn; preload them too so the
        # loader does not depend on LD_LIBRARY_PATH having been set before start-up
        if libs.get("cudnn"):
            d = os.path.dirname(libs["cudnn"])
            for pat in ("libnvrtc.so.12", "libnvrtc-builtins.so.12*", "libcudnn_graph.so.9", "libcudnn_*.so.9"):
                for extra in sorted(glob.glob(os.path.join(d, pat))):
                    if extra not in _preloaded:
                        try:
                            ctypes.CDLL(extra, mode=ctypes.RTLD_GLOBAL)
                            _preloaded[extra] = True
                        except OSError:
                            _preloaded[extra] = False
    return libs


def nvidia_smi_info() -> Optional[Dict[str, str]]:
    """GPU name / driver / memory from nvidia-smi, or None if no NVIDIA driver"""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10, creationflags=_CREATE_NO_WINDOW)
        line = (out.stdout or "").strip().splitlines()
        if out.returncode != 0 or not line:
            return None
        name, driver, mem = [x.strip() for x in line[0].split(",")[:3]]
        return {"name": name, "driver": driver, "memory_mb": mem, "count": str(len(line))}
    except Exception:
        return None


def cuda_status(extra_dir: str = "") -> Dict[str, object]:
    """Everything the UI needs to explain the GPU situation"""
    gpu = nvidia_smi_info()
    libs = setup_cuda(extra_dir) if gpu else find_libraries(extra_dir)
    missing = [k for k, v in libs.items() if not v]
    devices = 0
    if gpu and not missing:
        try:
            import ctranslate2
            devices = ctranslate2.get_cuda_device_count()
        except Exception:
            devices = 0
    if sys.platform == "darwin":
        text = "CUDA is not available on macOS — faster-whisper runs on the CPU (use whisper.cpp for Metal)."
    elif not gpu:
        text = "No NVIDIA GPU / driver detected (nvidia-smi not found) — CPU will be used."
    elif missing:
        text = (f"GPU: {gpu['name']} (driver {gpu['driver']}, {int(gpu['memory_mb']) // 1024} GB) detected, but CUDA "
                f"libraries are missing: {', '.join(missing)}.\n"
                "Click 'Download CUDA libraries' below to fetch them (~1 GB, one time), or install them with "
                "pip install nvidia-cublas-cu12 nvidia-cudnn-cu12, or select a folder containing them below.")
    elif devices == 0:
        text = (f"GPU: {gpu['name']} detected and CUDA libraries found, but CTranslate2 reports no CUDA device "
                "(driver too old for CUDA 12? try updating the NVIDIA driver).")
    else:
        text = (f"GPU ready: {gpu['name']} (driver {gpu['driver']}, {int(gpu['memory_mb']) // 1024} GB) — "
                f"cuBLAS: {os.path.dirname(libs['cublas'])}, cuDNN: {os.path.dirname(libs['cudnn'])}")
    return {"gpu": gpu, "libs": libs, "missing": missing, "devices": devices, "ready": devices > 0, "text": text}


# ----------------------------------------------------------------------------
# Downloading the runtime libraries from PyPI (official NVIDIA wheels are plain zip files)
# ----------------------------------------------------------------------------
_WHEEL_PACKAGES = ("nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12")


def _wheel_tag() -> Optional[str]:
    if sys.platform == "win32":
        return "win_amd64"
    if sys.platform.startswith("linux"):
        import platform
        return "manylinux" if platform.machine().lower() in ("x86_64", "amd64") else None
    return None


def _pick_wheel_url(package: str, major: int) -> str:
    """Newest release of `package` with the given major version that has a wheel for this platform"""
    import json
    import urllib.request
    tag = _wheel_tag()
    if not tag:
        raise RuntimeError("No NVIDIA wheels exist for this platform")
    with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as resp:
        data = json.load(resp)
    from packaging.version import Version
    releases = sorted(((Version(v), files) for v, files in data["releases"].items()
                       if not Version(v).is_prerelease), reverse=True)
    for ver, files in releases:
        if ver.major != major:
            continue
        for f in files:
            name = f["filename"]
            if name.endswith(".whl") and tag in name and ("x86_64" in name or "amd64" in name):
                return f["url"]
    raise RuntimeError(f"No {package} {major}.x wheel for {tag} on PyPI")


def download_cuda_libraries(dest_dir: str, progress=None, should_stop=None) -> List[str]:
    """
    Download cuBLAS 12 + cuDNN 9 from PyPI into dest_dir (only the shared libraries, ~1 GB download).
    progress(message, done_bytes, total_bytes) is called while downloading. Returns the extracted files.
    """
    import tempfile
    import urllib.request
    import zipfile

    os.makedirs(dest_dir, exist_ok=True)
    extracted: List[str] = []
    majors = {"nvidia-cublas-cu12": 12, "nvidia-cudnn-cu12": 9, "nvidia-cuda-nvrtc-cu12": 12}
    for package in _WHEEL_PACKAGES:
        if progress:
            progress(f"Resolving {package}…", 0, 0)
        url = _pick_wheel_url(package, majors[package])
        tmp = tempfile.NamedTemporaryFile(prefix="whisperer-", suffix=".whl", delete=False)
        tmp_path = tmp.name
        tmp.close()
        try:
            done, total, attempts = 0, 0, 0
            with open(tmp_path, "wb") as out:
                while True:
                    headers = {"User-Agent": "whisperer"}
                    if done:
                        headers["Range"] = f"bytes={done}-"
                    req = urllib.request.Request(url, headers=headers)
                    try:
                        with urllib.request.urlopen(req, timeout=120) as resp:
                            if done and resp.status != 206:
                                raise RuntimeError("server does not support resume")
                            if not total:
                                total = int(resp.headers.get("Content-Length") or 0)
                            while True:
                                if should_stop and should_stop():
                                    raise InterruptedError("Download cancelled")
                                chunk = resp.read(1 << 20)
                                if not chunk:
                                    break
                                out.write(chunk)
                                done += len(chunk)
                                if progress:
                                    progress(f"Downloading {os.path.basename(url)}", done, total)
                    except (OSError, RuntimeError) as exc:
                        attempts += 1
                        if attempts > 20:
                            raise RuntimeError(f"Download of {os.path.basename(url)} failed: {exc}") from exc
                        if progress:
                            progress(f"Connection dropped, resuming ({attempts})…", done, total)
                        continue
                    if total and done < total:
                        attempts += 1
                        if attempts > 20:
                            raise RuntimeError(f"Download of {os.path.basename(url)} incomplete ({done} of {total} bytes)")
                        continue
                    break
            if not zipfile.is_zipfile(tmp_path):
                raise RuntimeError(f"{os.path.basename(url)} is not a valid wheel archive")
            if progress:
                progress(f"Extracting {package}…", 0, 0)
            with zipfile.ZipFile(tmp_path) as zf:
                for info in zf.infolist():
                    base = os.path.basename(info.filename)
                    if not base or info.is_dir():
                        continue
                    if sys.platform == "win32":
                        ok = base.lower().endswith(".dll")
                    else:
                        ok = ".so" in base
                    if ok:
                        target = os.path.join(dest_dir, base)
                        with zf.open(info) as src, open(target, "wb") as dst:
                            while True:
                                block = src.read(1 << 20)
                                if not block:
                                    break
                                dst.write(block)
                        extracted.append(target)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    _done_dirs.clear()  # force re-registration of the new folder on next setup_cuda()
    return extracted


def default_cuda_dir() -> str:
    return os.path.join(_app_dir(), "cuda")
