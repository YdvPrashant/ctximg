"""Live machine telemetry, so you can see whether the work is actually moving.

Deliberately does not import torch: the app should be able to show what your
hardware is doing before the model has ever been loaded, and importing torch
just to read a GPU counter would cost seconds on a cold page load.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
from dataclasses import dataclass

_GB = 1024 ** 3
_lock = threading.Lock()
_nvml_ready: bool | None = None


@dataclass
class Gpu:
    name: str = ""
    util: int | None = None          # percent
    mem_used: float = 0.0            # GB
    mem_total: float = 0.0           # GB
    temperature: int | None = None   # Celsius
    power: float | None = None       # Watts

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "util": self.util,
            "mem_used": round(self.mem_used, 2),
            "mem_total": round(self.mem_total, 2),
            "temperature": self.temperature,
            "power": round(self.power, 1) if self.power is not None else None,
        }


@dataclass
class Machine:
    cpu_percent: float = 0.0
    cpu_count: int = 0
    ram_used: float = 0.0
    ram_total: float = 0.0
    gpu: Gpu | None = None
    gpu_present: bool = False

    def as_dict(self) -> dict:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "cpu_count": self.cpu_count,
            "ram_used": round(self.ram_used, 2),
            "ram_total": round(self.ram_total, 2),
            "gpu_present": self.gpu_present,
            "gpu": self.gpu.as_dict() if self.gpu else None,
        }


def _start_nvml() -> bool:
    """Initialise NVML once. False when there is no NVIDIA GPU to read."""
    global _nvml_ready
    with _lock:
        if _nvml_ready is not None:
            return _nvml_ready
        try:
            import pynvml

            pynvml.nvmlInit()
            _nvml_ready = True
        except Exception:
            _nvml_ready = False
        return _nvml_ready


def gpu_stats() -> Gpu | None:
    if not _start_nvml():
        return None
    try:
        import pynvml

        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", "replace")
        memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu = Gpu(
            name=name,
            mem_used=memory.used / _GB,
            mem_total=memory.total / _GB,
        )
        # Each of these is optional on some cards and drivers, so a missing
        # reading must not cost us the ones that did work.
        for attr, call, scale in (
            ("util", lambda: pynvml.nvmlDeviceGetUtilizationRates(handle).gpu, 1),
            ("temperature", lambda: pynvml.nvmlDeviceGetTemperature(handle, 0), 1),
            ("power", lambda: pynvml.nvmlDeviceGetPowerUsage(handle), 1000),
        ):
            try:
                setattr(gpu, attr, call() / scale if scale != 1 else call())
            except Exception:
                pass
        return gpu
    except Exception:
        return None


def machine() -> Machine:
    """One snapshot of what the hardware is doing right now."""
    info = Machine(cpu_count=os.cpu_count() or 0)
    try:
        import psutil

        # interval=None reads since the previous call, so it never blocks.
        info.cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        info.ram_used = (memory.total - memory.available) / _GB
        info.ram_total = memory.total / _GB
    except Exception:
        pass

    info.gpu = gpu_stats()
    info.gpu_present = info.gpu is not None or bool(shutil.which("nvidia-smi"))
    return info


def prime() -> None:
    """Take a throwaway CPU reading so the first real one is not zero."""
    try:
        import psutil

        psutil.cpu_percent(interval=None)
    except Exception:
        pass


def gpu_name_without_torch() -> str | None:
    """The GPU's name even when NVML is unavailable, for the doctor report."""
    gpu = gpu_stats()
    if gpu is not None:
        return gpu.name
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 else None
    except Exception:
        return None


def hf_cache_dir() -> "os.PathLike | None":
    """Where huggingface_hub keeps downloaded models, without importing it."""
    from pathlib import Path

    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    home = os.environ.get("HF_HOME")
    base = Path(home) if home else Path.home() / ".cache" / "huggingface"
    hub = base / "hub"
    return hub if hub.is_dir() else (base if base.is_dir() else None)


def download_bytes() -> int:
    """Bytes of model download currently in flight.

    huggingface_hub streams into "*.incomplete" files, so their combined size
    is how far a download has got. Without this a multi-gigabyte first run is
    indistinguishable from a hang.
    """
    cache = hf_cache_dir()
    if cache is None:
        return 0
    total = 0
    try:
        for path in cache.rglob("*.incomplete"):
            try:
                total += path.stat().st_size
            except OSError:
                continue
    except OSError:
        return 0
    return total
