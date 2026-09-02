"""Device selection, batch sizing, and out-of-memory recovery.

None of these load a model - they exercise the decisions made around it.
"""

from __future__ import annotations

import numpy as np
import pytest

from ctximg import cli
from ctximg.embedder import (
    Embedder,
    ModelError,
    default_batch_size,
    is_out_of_memory,
    resolve_model,
)


class Props:
    def __init__(self, gb):
        self.total_memory = int(gb * 1024 ** 3)


# --- model tiers ----------------------------------------------------------


def test_gpu_gets_the_accurate_model_and_cpu_the_fast_one():
    assert resolve_model(None, "cuda").arch == "ViT-L-14"
    assert resolve_model(None, "cpu").arch == "ViT-B-32"


def test_explicit_model_overrides_the_tier():
    assert resolve_model("ViT-B-32", "cuda").arch == "ViT-B-32"
    spec = resolve_model("ViT-B-16-SigLIP/webli", "cpu")
    assert (spec.arch, spec.pretrained) == ("ViT-B-16-SigLIP", "webli")


def test_unknown_model_fails_loudly_with_alternatives():
    with pytest.raises(ModelError, match="Unknown model"):
        resolve_model("ViT-NOPE-99", "cpu")
    with pytest.raises(ModelError, match="Unknown checkpoint"):
        resolve_model("ViT-B-32/not_a_checkpoint", "cpu")


def test_model_id_is_stable_and_includes_the_checkpoint():
    assert resolve_model(None, "cpu").model_id == "ViT-B-32/laion2b_s34b_b79k"


# --- batch sizing ---------------------------------------------------------


def test_cpu_batch_is_fixed():
    assert default_batch_size("cpu") == 32


@pytest.mark.parametrize("vram_gb,expected", [(6, 32), (8, 64), (12, 64), (24, 128)])
def test_gpu_batch_scales_with_vram(monkeypatch, vram_gb, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: Props(vram_gb))
    assert default_batch_size("cuda") == expected


def test_batch_falls_back_when_the_card_cannot_be_queried(monkeypatch):
    import torch

    def boom(i):
        raise RuntimeError("no CUDA device")

    monkeypatch.setattr(torch.cuda, "get_device_properties", boom)
    assert default_batch_size("cuda") == 32


# --- out-of-memory recovery -----------------------------------------------


def test_out_of_memory_is_recognised_by_message():
    assert is_out_of_memory(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"))
    assert not is_out_of_memory(RuntimeError("some other failure"))
    assert not is_out_of_memory(ValueError("out of memory"))


class FakeCuda:
    def __init__(self):
        self.cleared = 0

    def empty_cache(self):
        self.cleared += 1


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def make_embedder(fail_above: int):
    """An Embedder whose encode step blows up above a given batch size."""
    embedder = Embedder.__new__(Embedder)
    embedder._torch = FakeTorch()
    sizes = []

    def encode(tensors):
        sizes.append(len(tensors))
        if len(tensors) > fail_above:
            raise RuntimeError("CUDA out of memory. Tried to allocate 512 MiB")
        return np.tile(np.arange(len(tensors), dtype=np.float32)[:, None], (1, 3))

    embedder._encode_images = encode
    return embedder, sizes


def test_oversized_batch_is_split_and_retried():
    embedder, sizes = make_embedder(fail_above=2)
    out = embedder.encode_images([object()] * 8)

    assert out.shape == (8, 3), "every image must still come back"
    assert max(sizes) == 8 and min(sizes) <= 2, "the batch must have been split down"
    assert embedder._torch.cuda.cleared > 0, "cache must be freed before retrying"


def test_a_single_image_that_will_not_fit_raises():
    embedder, _ = make_embedder(fail_above=0)
    with pytest.raises(RuntimeError, match="out of memory"):
        embedder.encode_images([object()])


def test_non_memory_errors_are_not_retried():
    embedder = Embedder.__new__(Embedder)
    embedder._torch = FakeTorch()
    calls = []

    def encode(tensors):
        calls.append(len(tensors))
        raise RuntimeError("model is broken")

    embedder._encode_images = encode
    with pytest.raises(RuntimeError, match="model is broken"):
        embedder.encode_images([object()] * 8)
    assert calls == [8], "a real error must surface immediately, not split"


# --- the silent-CPU-fallback warning --------------------------------------


def test_warning_fires_when_a_gpu_exists_but_torch_cannot_use_it(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:\\nvidia-smi.exe")
    warning = cli._gpu_missed_warning("cpu")
    assert warning is not None
    assert "CPU-only" in warning
    assert "download.pytorch.org/whl/cu126" in warning


def test_no_warning_when_running_on_the_gpu(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: "C:\\nvidia-smi.exe")
    assert cli._gpu_missed_warning("cuda") is None


def test_no_warning_on_a_machine_without_an_nvidia_gpu(monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert cli._gpu_missed_warning("cpu") is None
