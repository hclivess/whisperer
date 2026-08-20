#!/usr/bin/env python3
"""
Build a self-contained whisperer distribution (no Python required on the target) using Nuitka,
then package it as
    dist/whisperer-<version>-<os>-<arch>.zip      (Windows)
    dist/whisperer-<version>-<os>-<arch>.tar.gz   (Linux / macOS, keeps exec bits)

Usage:  python build.py            (run from the repository root)
Needs:  pip install -r requirements.txt nuitka
"""
import glob
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from config import APP_NAME, APP_VERSION  # noqa: E402

BUILD_DIR = os.path.join(ROOT, "build")
DIST_DIR = os.path.join(ROOT, "dist")
EXTRA_FILES = ("README.md", "LICENSE", "icon.ico")


def os_tag() -> str:
    return {"windows": "windows", "linux": "linux", "darwin": "macos"}.get(
        platform.system().lower(), platform.system().lower())


def arch_tag() -> str:
    machine = platform.machine().lower()
    return {"amd64": "x64", "x86_64": "x64", "arm64": "arm64", "aarch64": "arm64"}.get(machine, machine)


def metadata_flags(*names: str) -> list:
    """--include-distribution-metadata only for distributions that are actually installed"""
    from importlib.metadata import distribution, PackageNotFoundError
    flags = []
    for name in names:
        try:
            distribution(name)
        except PackageNotFoundError:
            continue
        flags.append(f"--include-distribution-metadata={name}")
    return flags


def nuitka_command() -> list:
    cmd = [
        sys.executable, "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--enable-plugin=pyside6",
        "--noinclude-qt-translations",
        # native inference stack used by faster-whisper
        "--include-package=faster_whisper",
        "--include-package-data=faster_whisper",
        "--include-package=ctranslate2",
        "--include-package=tokenizers",
        "--include-package=huggingface_hub",
        "--include-package=onnxruntime",
        "--include-package-data=onnxruntime",
        "--include-package=av",
        *metadata_flags("faster-whisper", "ctranslate2", "tokenizers", "huggingface_hub",
                        "onnxruntime", "av", "tqdm", "numpy", "requests", "httpx", "filelock", "packaging",
                        "pyyaml", "typing_extensions", "fsspec"),
        f"--output-dir={BUILD_DIR}",
        f"--output-filename={APP_NAME}",
        "--include-data-files=icon.ico=icon.ico",
        f"--product-name={APP_NAME}",
        f"--product-version={APP_VERSION}",
        f"--file-version={APP_VERSION}",
        "--file-description=Whisper subtitle generator GUI",
        "--copyright=MIT License",
    ]
    system = platform.system()
    if system == "Windows":
        cmd += ["--windows-console-mode=disable", "--windows-icon-from-ico=icon.ico"]
    elif system == "Darwin":
        cmd += ["--macos-create-app-bundle", "--macos-app-icon=none",
                f"--macos-app-name={APP_NAME}", f"--macos-app-version={APP_VERSION}"]
    cmd.append("main.py")
    return cmd


def find_output_dir() -> str:
    if platform.system() == "Darwin":
        patterns = ("main.app", f"{APP_NAME}.app", "*.app")
    else:
        patterns = ("main.dist", "*.dist")
    for pattern in patterns:
        matches = glob.glob(os.path.join(BUILD_DIR, pattern))
        if matches:
            return matches[0]
    raise SystemExit("Nuitka output directory not found in build/")


def package(out_dir: str) -> str:
    os.makedirs(DIST_DIR, exist_ok=True)
    stem = f"{APP_NAME}-{APP_VERSION}-{os_tag()}-{arch_tag()}"
    for name in EXTRA_FILES:
        src = os.path.join(ROOT, name)
        if os.path.exists(src):
            dest_root = os.path.join(out_dir, "Contents", "MacOS") if out_dir.endswith(".app") else out_dir
            shutil.copy2(src, os.path.join(dest_root, name))
    os.makedirs(os.path.join(out_dir if not out_dir.endswith(".app") else os.path.join(out_dir, "Contents", "MacOS"),
                             "presets"), exist_ok=True)
    top_name = f"{APP_NAME}.app" if out_dir.endswith(".app") else stem
    if platform.system() == "Windows":
        archive = os.path.join(DIST_DIR, stem + ".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
            for folder, _dirs, files in os.walk(out_dir):
                for fn in files:
                    full = os.path.join(folder, fn)
                    zf.write(full, os.path.join(top_name, os.path.relpath(full, out_dir)))
    else:
        archive = os.path.join(DIST_DIR, stem + ".tar.gz")
        with tarfile.open(archive, "w:gz") as tf:
            tf.add(out_dir, arcname=top_name)
    return archive


def main():
    os.chdir(ROOT)
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    cmd = nuitka_command()
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    archive = package(find_output_dir())
    print(f"\nBuilt {archive} ({os.path.getsize(archive) / (1024 * 1024):.1f} MB)")


if __name__ == "__main__":
    main()
