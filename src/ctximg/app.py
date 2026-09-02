"""Shared wiring: settings, the model, and the library of indexed folders.

The model is loaded lazily, so launching the app and listing folders cost
nothing - torch is only imported once you actually search or index.
"""

from __future__ import annotations

from . import config as config_mod


class App:
    def __init__(self, config: config_mod.Config | None = None):
        self.config = config if config is not None else config_mod.load()
        self._embedder = None
        self._library = None

    @property
    def embedder(self):
        """Load the CLIP model on first use."""
        if self._embedder is None:
            from .embedder import Embedder

            self._embedder = Embedder(
                self.config.model, self.config.device,
                tier=self.config.tier, precision=self.config.precision,
            )
        return self._embedder

    def peek_embedder(self):
        """The model if it is already loaded, else None - never loads it.

        Lets the folder list render instantly on a cold start instead of
        blocking on a multi-second model load just to draw checkboxes.
        """
        return self._embedder

    @property
    def library(self):
        if self._library is None:
            from .library import Library

            self._library = Library(self)
        return self._library

    def settings(self) -> dict:
        """What is configured, and what is actually running.

        Reported separately on purpose: "device: auto" is a preference, and
        "running on cuda" is a fact, and confusing the two is how people end
        up wondering why indexing is slow.
        """
        from . import sysinfo
        from .embedder import describe_tiers, resolve_precision

        embedder = self.peek_embedder()
        device = embedder.device if embedder else self._likely_device()
        return {
            "configured": {
                "tier": self.config.tier,
                "model": self.config.model,
                "device": self.config.device,
                "precision": self.config.precision,
                "thumb_size": self.config.thumb_size,
                "batch_size": self.config.batch_size,
            },
            "running": embedder.describe() if embedder else {
                "device": device,
                "precision": resolve_precision(self.config.precision, device),
            },
            "model_loaded": embedder is not None,
            "tiers": describe_tiers(device, self.config.rates),
            "machine": sysinfo.machine().as_dict(),
        }

    def apply_settings(self, changes: dict) -> dict:
        """Change settings and drop anything they invalidate.

        Switching tier or precision means a different model, so the loaded one
        has to go and the stacked vectors with it - otherwise the next search
        would run against a model that is no longer configured.
        """
        from . import config as config_mod

        allowed = {"tier", "model", "device", "precision", "thumb_size", "batch_size"}
        touched = []
        for key, value in changes.items():
            if key not in allowed:
                raise config_mod.ConfigError(f"{key!r} cannot be set here")
            self.config.set_value(key, "" if value is None else str(value))
            touched.append(key)
        config_mod.save(self.config)

        if {"tier", "model", "device", "precision"} & set(touched):
            self._embedder = None
            if self._library is not None:
                self._library.invalidate()
        return self.settings()

    def _likely_device(self) -> str:
        """The device we would use, without paying for a torch import."""
        from . import sysinfo

        if self.config.device in ("cpu", "cuda"):
            return self.config.device
        return "cuda" if sysinfo.gpu_stats() is not None else "cpu"

    def status(self) -> dict:
        library = self.library
        folders = library.folders()
        embedder = self.peek_embedder()
        return {
            "folders": [f.as_dict() for f in folders],
            "selected_photos": sum(f.photos for f in folders if f.state == "ready"
                                   and f.selected),
            "total_photos": sum(f.photos for f in folders),
            "model": embedder.model_id if embedder else None,
            "device": embedder.device if embedder else None,
            "precision": embedder.precision if embedder else None,
            "tier": self.config.tier,
            "model_loaded": embedder is not None,
            "job": library.job.as_dict(),
        }
