#!/usr/bin/env python3
"""
Build a self-contained whisperer distribution (no Python required on the target) using PyInstaller,
then package it as
    dist/whisperer-<version>-<os>-<arch>.zip      (Windows)
    dist/whisperer-<version>-<os>-<arch>.tar.gz   (Linux / macOS, keeps exec bits)

Usage:  python build.py            (run from the repository root)
Needs:  pip install -r requirements.txt pyinstaller
"""
import glob
import hashlib
import os
import platform
import re
import shlex
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
    """--copy-metadata only for distributions that are actually installed"""
    from importlib.metadata import distribution, PackageNotFoundError
    flags = []
    for name in names:
        try:
            distribution(name)
        except PackageNotFoundError:
            continue
        flags.append(f"--copy-metadata={name}")
    return flags


VERSION_INFO_TEMPLATE = """\
# Windows version resource. An executable with no publisher/product metadata looks like malware to
# SmartScreen and to antivirus heuristics; this does not replace a signature, but it is the cheap half.
VSVersionInfo(
  ffi=FixedFileInfo(filevers=%(vers)s, prodvers=%(vers)s, mask=0x3f, flags=0x0, OS=0x40004,
                    fileType=0x1, subtype=0x0, date=(0, 0)),
  kids=[
    StringFileInfo([StringTable("040904B0", [
      StringStruct("CompanyName", "%(author)s"),
      StringStruct("FileDescription", "%(description)s"),
      StringStruct("FileVersion", "%(version)s"),
      StringStruct("InternalName", "%(name)s"),
      StringStruct("LegalCopyright", "%(copyright)s"),
      StringStruct("OriginalFilename", "%(name)s.exe"),
      StringStruct("ProductName", "%(name)s"),
      StringStruct("ProductVersion", "%(version)s"),
    ])]),
    VarFileInfo([VarStruct("Translation", [1033, 1200])]),
  ]
)
"""


def write_version_info() -> str:
    """Windows version resource for the .exe, generated from config.py and LICENSE (returns its path)."""
    parts = [int(p) for p in re.findall(r"\d+", APP_VERSION)][:4]
    parts += [0] * (4 - len(parts))
    copyright_line = "MIT licensed"
    try:
        with open(os.path.join(ROOT, "LICENSE"), encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("Copyright"):
                    copyright_line = line.strip()
                    break
    except OSError:
        pass
    # "Copyright (c) 2026 Jan Kucera" -> "Jan Kucera"
    author = re.sub(r"^.*?\)\s*|^\s*copyright\s*", "", copyright_line, flags=re.I)
    author = re.sub(r"^\s*\d{4}(\s*[-,]\s*\d{4})?\s*", "", author).strip() or APP_NAME
    os.makedirs(BUILD_DIR, exist_ok=True)
    path = os.path.join(BUILD_DIR, "version_info.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(VERSION_INFO_TEMPLATE % {
            "vers": tuple(parts), "version": APP_VERSION, "name": APP_NAME, "author": author,
            "description": "Whisper subtitle generator and subtitle resyncer", "copyright": copyright_line})
    return path


def pyinstaller_command() -> list:
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir", "--windowed", "--noupx",
        f"--name={APP_NAME}",
        f"--distpath={os.path.join(BUILD_DIR, 'out')}",
        f"--workpath={os.path.join(BUILD_DIR, 'work')}",
        f"--specpath={BUILD_DIR}",
        # native inference stack used by faster-whisper
        "--collect-all=faster_whisper",
        "--collect-all=ctranslate2",
        "--collect-all=onnxruntime",
        "--collect-all=tokenizers",
        "--collect-all=huggingface_hub",
        "--collect-submodules=av",
        "--collect-binaries=av",
        "--hidden-import=psutil",
        *metadata_flags("faster-whisper", "ctranslate2", "tokenizers", "huggingface_hub", "onnxruntime", "av",
                        "tqdm", "numpy", "requests", "httpx", "filelock", "packaging", "PyYAML",
                        "typing_extensions", "fsspec", "hf-xet", "psutil"),
        # trim Qt modules we never use
        "--exclude-module=PySide6.QtWebEngineCore", "--exclude-module=PySide6.QtWebEngineWidgets",
        "--exclude-module=PySide6.Qt3DCore", "--exclude-module=PySide6.QtQuick", "--exclude-module=PySide6.QtQml",
        "--exclude-module=PySide6.QtMultimedia", "--exclude-module=PySide6.QtCharts", "--exclude-module=PySide6.QtPdf",
        "--exclude-module=torch", "--exclude-module=tkinter",
        f"--add-data={os.path.join(ROOT, 'icon.ico')}{os.pathsep}.",
    ]
    if platform.system() != "Darwin":
        cmd.append(f"--icon={os.path.join(ROOT, 'icon.ico')}")
    if platform.system() == "Windows":
        cmd.append(f"--version-file={write_version_info()}")
    cmd.append(os.path.join(ROOT, "main.py"))
    return cmd


def sign(out_dir: str) -> None:
    """
    Authenticode-sign the executable when SIGN_COMMAND is set (Windows releases).

    Unsigned executables are what makes SmartScreen say "Windows protected your PC" and antivirus heuristics
    flag a fresh PyInstaller build. SIGN_COMMAND is a full command line with {file} where the path goes, e.g.
        signtool sign /fd SHA256 /tr http://timestamp.digicert.com /td SHA256 /f cert.pfx /p PASS {file}
    A failing signature fails the build - a release must never go out silently unsigned.
    """
    command = os.environ.get("SIGN_COMMAND", "").strip()
    if not command:
        return
    exe = os.path.join(out_dir, f"{APP_NAME}.exe" if platform.system() == "Windows" else APP_NAME)
    if not os.path.exists(exe):
        raise SystemExit(f"SIGN_COMMAND is set but {exe} does not exist")
    print(f"Signing {exe}", flush=True)      # never the command line itself: it carries the key password
    if platform.system() == "Windows":
        # let Windows parse the command line - signtool lives under "C:\Program Files (x86)\..."
        subprocess.run(command.replace("{file}", f'"{exe}"'), check=True)
    else:
        subprocess.run([part.replace("{file}", exe) for part in shlex.split(command)], check=True)


def checksum(archive: str) -> str:
    """Write <archive>.sha256 next to the archive so downloads can be verified"""
    digest = hashlib.sha256()
    with open(archive, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    path = archive + ".sha256"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{digest.hexdigest()}  {os.path.basename(archive)}\n")
    return path


def find_output_dir() -> str:
    out = os.path.join(BUILD_DIR, "out")
    if platform.system() == "Darwin":
        app = os.path.join(out, f"{APP_NAME}.app")
        if os.path.isdir(app):
            return app
    folder = os.path.join(out, APP_NAME)
    if os.path.isdir(folder):
        return folder
    raise SystemExit("PyInstaller output directory not found in build/out")


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
    if "--checksum" in sys.argv:
        # CI re-runs this on an archive that came back signed from a signing service: the bytes changed,
        # so the .sha256 written at packaging time no longer matches what people download.
        for archive in sys.argv[sys.argv.index("--checksum") + 1:]:
            print(f"Checksum: {checksum(archive)}")
        return
    shutil.rmtree(BUILD_DIR, ignore_errors=True)
    cmd = pyinstaller_command()
    print("Running:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    out_dir = find_output_dir()
    sign(out_dir)
    archive = package(out_dir)
    print(f"\nBuilt {archive} ({os.path.getsize(archive) / (1024 * 1024):.1f} MB)")
    print(f"Checksum: {checksum(archive)}")


if __name__ == "__main__":
    main()
