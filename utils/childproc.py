"""
Child-process hygiene: quitting the app — normally, via Stop, or by being killed — never leaves ffmpeg, whisper-cli,
exiftool or any other helper running.

    from utils.childproc import popen, kill_all
    proc = popen(cmd, ...)          # drop-in for subprocess.Popen; the process is tracked
    kill_all()                      # called automatically at interpreter exit and on QApplication.aboutToQuit

Three layers:
  * atexit + QApplication.aboutToQuit -> kill_all() on every normal quit
  * Windows: every child is assigned to a Job object with KILL_ON_JOB_CLOSE -> the OS kills the tree the moment our
    process dies, crash or Task Manager included
  * Linux: every child gets PR_SET_PDEATHSIG=SIGKILL -> same guarantee from the kernel
  macOS has no parent-death signal; there the first layer applies (keep helpers stoppable).
"""
import atexit
import os
import subprocess
import sys
import threading
from typing import Optional, Set

_procs: Set[subprocess.Popen] = set()
_lock = threading.Lock()
_job = None

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _windows_job():
    global _job
    if _job is not None or sys.platform != "win32":
        return _job
    try:
        import ctypes
        from ctypes import wintypes

        class IO_COUNTERS(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("ReadOperationCount", "WriteOperationCount",
                                                            "OtherOperationCount", "ReadTransferCount",
                                                            "WriteTransferCount", "OtherTransferCount")]

        class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION), ("IoInfo", IO_COUNTERS),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32 = ctypes.windll.kernel32
        job = k32.CreateJobObjectW(None, None)
        info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
        info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # JobObjectExtendedLimitInformation
        _job = job
    except Exception:
        _job = False
    return _job


def register(proc: subprocess.Popen) -> subprocess.Popen:
    with _lock:
        _procs.add(proc)
    job = _windows_job()
    if job:
        try:
            import ctypes
            ctypes.windll.kernel32.AssignProcessToJobObject(job, int(proc._handle))  # noqa: SLF001
        except Exception:
            pass
    return proc


def _die_with_parent():
    """Linux: the kernel kills this child with SIGKILL the moment the parent dies — crash, SIGKILL, anything."""
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").prctl(1, 9, 0, 0, 0)          # PR_SET_PDEATHSIG = 1, SIGKILL = 9
    except Exception:
        pass


def popen(*args, **kwargs) -> subprocess.Popen:
    """subprocess.Popen that is tracked, hidden on Windows, and dies with us on Windows (Job) and Linux (PDEATHSIG)."""
    if sys.platform == "win32":
        kwargs.setdefault("creationflags", CREATE_NO_WINDOW)
    elif sys.platform.startswith("linux") and "preexec_fn" not in kwargs:
        kwargs["preexec_fn"] = _die_with_parent
    return register(subprocess.Popen(*args, **kwargs))


def run(*args, timeout: Optional[float] = None, **kwargs) -> subprocess.CompletedProcess:
    """subprocess.run equivalent for short helpers (ffprobe -show_format, exiftool -ver) that is still tracked."""
    kwargs.setdefault("stdout", subprocess.PIPE)
    kwargs.setdefault("stderr", subprocess.PIPE)
    stdin = kwargs.pop("input", None)
    proc = popen(*args, stdin=subprocess.PIPE if stdin is not None else kwargs.pop("stdin", None), **kwargs)
    try:
        out, err = proc.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        kill(proc)
        raise
    finally:
        forget(proc)
    return subprocess.CompletedProcess(args[0] if args else kwargs.get("args"), proc.returncode, out, err)


def forget(proc: subprocess.Popen) -> None:
    with _lock:
        _procs.discard(proc)


def kill(proc: subprocess.Popen) -> None:
    """Kill one process and everything it spawned."""
    if proc.poll() is not None:
        forget(proc)
        return
    try:
        import psutil
        try:
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except psutil.Error:
                    pass
        except psutil.Error:
            pass
    except ImportError:
        pass
    try:
        proc.kill()
        proc.wait(timeout=5)
    except Exception:
        pass
    forget(proc)


def kill_all() -> int:
    with _lock:
        procs = list(_procs)
    for p in procs:
        kill(p)
    return len(procs)


def install_qt_hook(app) -> None:
    """Call once after QApplication is created: kills every tracked child when the app quits for any reason."""
    try:
        app.aboutToQuit.connect(kill_all)
    except Exception:
        pass


atexit.register(kill_all)
# No Python-level SIGTERM handler on purpose: it cannot run while the main thread is inside a blocking Qt call and
# would turn a plain `kill` into a hang. Signals are covered by the kernel-level layers above.
