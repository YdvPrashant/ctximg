"""CLIP-family model wrapper: images and text into one shared vector space.

torch and open_clip are imported lazily so that `ctximg config` and `ctximg
stats` stay instant - importing torch costs seconds and neither needs it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Three tiers, because accuracy and indexing time trade directly against each
# other and only you know which you want today.
#
# The GPU rates were measured on an RTX 3060 Laptop (6 GB) at fp16, batch 8.
# They are a reference point for the first estimate only: once a folder has
# actually been indexed, the observed rate replaces them.
#
# Anything larger than these was tried and rejected. ViT-H-14-378 (3.8 GB) and
# ViT-gopt-16-SigLIP2-384 (3.7 GB) spill past what a 6 GB card can hold once
# the display has taken its share, and collapse to about 1 img/s.
TIERS = {
    "fast": {
        "label": "Fast",
        "note": "Quickest to index. Good for a first pass over a big gallery.",
        "cuda": ("ViT-L-14", "laion2b_s32b_b82k", 100.0),
        "cpu": ("ViT-B-32", "laion2b_s34b_b79k", 20.0),
    },
    "balanced": {
        "label": "Balanced",
        "note": "SigLIP 2 at 384px. Markedly better than Fast at roughly half "
                "the speed.",
        "cuda": ("ViT-L-16-SigLIP2-384", "webli", 49.0),
        "cpu": ("ViT-B-16-SigLIP2", "webli", 4.0),
    },
    "best": {
        "label": "Most accurate",
        "note": "The largest model that fits comfortably. Best at detail and "
                "at long, specific descriptions.",
        "cuda": ("ViT-SO400M-16-SigLIP2-384", "webli", 32.0),
        "cpu": ("ViT-B-16-SigLIP2-384", "webli", 1.5),
    },
}

DEFAULT_TIER = "fast"


def tier_for(name: str) -> dict:
    return TIERS.get(name) or TIERS[DEFAULT_TIER]


def tier_spec(name: str, device: str) -> tuple[str, str, float]:
    """(arch, checkpoint, reference images/second) for a tier on a device."""
    return tier_for(name)["cuda" if device == "cuda" else "cpu"]


def describe_tiers(device: str, observed: dict | None = None) -> list[dict]:
    """The tiers as the app shows them, with measured rates where we have them."""
    observed = observed or {}
    out = []
    for key, tier in TIERS.items():
        arch, pretrained, rate = tier_spec(key, device)
        model_id = f"{arch}/{pretrained}"
        measured = observed.get(f"{model_id}|{device}")
        out.append({
            "key": key,
            "label": tier["label"],
            "note": tier["note"],
            "model": model_id,
            "arch": arch,
            "rate": round(measured or rate, 1),
            "measured": measured is not None,
        })
    return out

# Averaged over these, then renormalised. CLIP's text tower was trained on
# captions, so a bare noun phrase sits slightly off-distribution; the ensemble
# recentres it. Still one query vector against one space - no keyword matching.
PROMPT_TEMPLATES = (
    "{}",
    "a photo of {}",
    "a picture of {}",
    "a photo containing {}",
)


class ModelError(RuntimeError):
    """The configured model could not be resolved or loaded."""


@dataclass(frozen=True)
class ModelSpec:
    arch: str
    pretrained: str

    @property
    def model_id(self) -> str:
        return f"{self.arch}/{self.pretrained}"


def resolve_device(preference: str = "auto") -> str:
    """Pick the compute device, honouring an explicit config override."""
    import torch

    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise ModelError(
                "device is set to 'cuda' but this torch build cannot see a GPU. "
                "Run 'ctximg doctor' for details."
            )
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_precision(preference: str, device: str) -> str:
    """fp16 on a GPU unless told otherwise.

    Vectors are stored as float16 regardless, so computing in fp32 and then
    throwing the precision away costs memory and speed for nothing. On this
    class of card it is roughly three times faster, and it is what makes the
    larger models fit at all.
    """
    if preference in ("fp16", "fp32"):
        return preference
    return "fp16" if device == "cuda" else "fp32"


def resolve_model(name: str | None, device: str, tier: str = DEFAULT_TIER) -> ModelSpec:
    """Turn a config value into a concrete (arch, pretrained) pair.

    Accepts 'ViT-L-14', 'ViT-L-14/laion2b_s32b_b82k', or None to take the
    chosen tier's model for this device. An unknown name fails loudly with the
    valid alternatives rather than silently falling back.
    """
    import open_clip

    if not name:
        arch, pretrained, _ = tier_spec(tier, device)
        return ModelSpec(arch, pretrained)

    available = open_clip.list_pretrained()
    if "/" in name:
        arch, pretrained = name.split("/", 1)
        if (arch, pretrained) not in available:
            raise ModelError(
                f"Unknown checkpoint {name!r}. "
                f"Checkpoints for {arch!r}: {_options_for(available, arch)}"
            )
        return ModelSpec(arch, pretrained)

    candidates = [p for a, p in available if a == name]
    if not candidates:
        architectures = sorted({a for a, _ in available})
        raise ModelError(
            f"Unknown model {name!r}. Known architectures include: "
            f"{', '.join(architectures[:12])} ..."
        )
    preferred = next(
        (p for p in candidates if "laion2b" in p or p == "webli"), candidates[0]
    )
    return ModelSpec(name, preferred)


def _options_for(available, arch: str) -> str:
    options = [p for a, p in available if a == arch]
    return ", ".join(options) if options else "(no such architecture)"


def default_batch_size(device: str) -> int:
    """Scale the batch to the card.

    A 6 GB laptop GPU is also driving the display, so the batch that suits a
    24 GB desktop card would spend the run recovering from out-of-memory.
    """
    if device != "cuda":
        return 32
    try:
        import torch

        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        return 32
    if vram_gb < 8:
        return 32
    if vram_gb < 16:
        return 64
    return 128


def is_out_of_memory(exc: BaseException) -> bool:
    """True for a CUDA out-of-memory failure, whatever torch calls it today."""
    cuda_oom = getattr(__import__("torch"), "OutOfMemoryError", None)
    if cuda_oom is not None and isinstance(exc, cuda_oom):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class Embedder:
    """Loads the model once and encodes batches of images or text."""

    def __init__(
        self,
        model: str | None = None,
        device: str = "auto",
        tier: str = DEFAULT_TIER,
        precision: str = "auto",
    ):
        import open_clip
        import torch

        self.device = resolve_device(device)
        self.tier = tier
        self.spec = resolve_model(model, self.device, tier)
        self.precision = resolve_precision(precision, self.device)
        self._torch = torch

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.spec.arch,
                pretrained=self.spec.pretrained,
                precision=self.precision,
                device=self.device,
            )
        except Exception as exc:  # network failure, corrupt cache, bad name
            raise ModelError(
                f"Could not load {self.spec.model_id}: {exc}"
            ) from exc
        self.model.eval()
        try:
            self.tokenizer = open_clip.get_tokenizer(self.spec.arch)
        except ModuleNotFoundError as exc:
            # SigLIP checkpoints tokenise through HuggingFace, unlike the LAION
            # CLIP models which carry open_clip's own BPE tokenizer.
            raise ModelError(
                f"{self.spec.arch} needs the {exc.name!r} package for its "
                f"tokenizer. Install it with:\n"
                f"  .venv\\Scripts\\python -m pip install {exc.name}"
            ) from exc

        # Probe the output width rather than guessing per architecture.
        with torch.inference_mode():
            probe = self.model.encode_text(self.tokenizer(["probe"]).to(self.device))
        self.dim = int(probe.shape[-1])

        size = getattr(self.model.visual, "image_size", None)
        self.image_size = int(size[0] if isinstance(size, (tuple, list)) else size or 0)

    def describe(self) -> dict:
        """What the app shows about the model actually in use."""
        return {
            "model": self.model_id,
            "arch": self.spec.arch,
            "tier": self.tier,
            "device": self.device,
            "precision": self.precision,
            "dim": self.dim,
            "image_size": self.image_size,
        }

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def encode_images(self, tensors) -> np.ndarray:
        """Encode preprocessed image tensors into unit row vectors.

        On a CUDA out-of-memory the batch is split and retried rather than
        losing the run: VRAM available on a laptop GPU moves with whatever
        else is on screen, so a batch size that worked a minute ago can fail.
        """
        try:
            return self._encode_images(tensors)
        except Exception as exc:
            if len(tensors) <= 1 or not is_out_of_memory(exc):
                raise
            self._torch.cuda.empty_cache()
            middle = len(tensors) // 2
            return np.concatenate(
                [
                    self.encode_images(tensors[:middle]),
                    self.encode_images(tensors[middle:]),
                ]
            )

    def _encode_images(self, tensors) -> np.ndarray:
        torch = self._torch
        batch = torch.stack(tensors).to(self.device)
        # Preprocessing always yields fp32; the weights may not be.
        if self.precision == "fp16":
            batch = batch.half()
        with torch.inference_mode():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_text(self, texts: list[str]) -> np.ndarray:
        """Encode text into unit row vectors, one per input string."""
        torch = self._torch
        tokens = self.tokenizer(texts).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.float().cpu().numpy()

    def encode_query(self, query: str, ensemble: bool = True) -> np.ndarray:
        """Encode a search query into a single unit vector.

        With ensemble on, the query is averaged across a few caption templates
        and renormalised, which steadies short queries.
        """
        text = query.strip()
        if not text:
            raise ValueError("Query is empty")
        if not ensemble:
            return self.encode_text([text])[0]

        variants = [t.format(text) for t in PROMPT_TEMPLATES]
        mean = self.encode_text(variants).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        return mean / norm if norm > 0 else mean
