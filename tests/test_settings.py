"""Quality tiers, precision, machine telemetry and work estimates."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ctximg import sysinfo, workspace
from ctximg.app import App
from ctximg.config import Config, ConfigError
from ctximg.embedder import (
    DEFAULT_TIER,
    TIERS,
    describe_tiers,
    resolve_model,
    resolve_precision,
    tier_spec,
)
from ctximg.library import Library
from ctximg.web.server import create_app


class StubEmbedder:
    model_id = "stub/v1"
    dim = 3
    device = "cpu"
    precision = "fp32"
    tier = "fast"
    image_size = 224

    def describe(self):
        return {"model": self.model_id, "device": self.device, "arch": "stub",
                "precision": self.precision, "dim": self.dim,
                "tier": self.tier, "image_size": self.image_size}

    def encode_query(self, query, ensemble=True):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


# --- tiers ----------------------------------------------------------------


def test_every_tier_names_a_model_for_both_devices():
    for key in TIERS:
        for device in ("cpu", "cuda"):
            arch, checkpoint, rate = tier_spec(key, device)
            assert arch and checkpoint
            assert rate > 0, "a tier without a speed cannot produce an estimate"


def test_tiers_get_slower_as_they_get_better():
    rates = [tier_spec(k, "cuda")[2] for k in ("fast", "balanced", "best")]
    assert rates == sorted(rates, reverse=True), "the trade-off must be monotonic"


def test_an_unknown_tier_falls_back_rather_than_crashing():
    assert tier_spec("nonsense", "cuda") == tier_spec(DEFAULT_TIER, "cuda")


def test_the_tier_picks_the_model_when_none_is_named():
    assert resolve_model(None, "cuda", "fast").arch == "ViT-L-14"
    assert resolve_model(None, "cuda", "best").arch == "ViT-SO400M-16-SigLIP2-384"


def test_an_explicit_model_still_beats_the_tier():
    assert resolve_model("ViT-B-32", "cuda", "best").arch == "ViT-B-32"


def test_described_tiers_prefer_a_rate_measured_on_this_machine():
    plain = {t["key"]: t for t in describe_tiers("cuda")}
    assert plain["fast"]["measured"] is False

    model = plain["fast"]["model"]
    measured = {t["key"]: t for t in describe_tiers("cuda", {f"{model}|cuda": 12.5})}
    assert measured["fast"]["rate"] == 12.5
    assert measured["fast"]["measured"] is True
    assert measured["best"]["measured"] is False, "only the one we measured"


# --- precision ------------------------------------------------------------


def test_gpu_defaults_to_fp16_and_cpu_to_fp32():
    """Vectors are stored as float16 anyway, so fp32 on a GPU buys nothing."""
    assert resolve_precision("auto", "cuda") == "fp16"
    assert resolve_precision("auto", "cpu") == "fp32"


def test_an_explicit_precision_is_honoured_on_either_device():
    assert resolve_precision("fp32", "cuda") == "fp32"
    assert resolve_precision("fp16", "cpu") == "fp16"


def test_config_rejects_a_precision_that_does_not_exist():
    with pytest.raises(ConfigError, match="precision must be"):
        Config().set_value("precision", "int4")


def test_config_rejects_an_unknown_tier():
    with pytest.raises(ConfigError, match="tier must be"):
        Config().set_value("tier", "turbo")


# --- remembered rates -----------------------------------------------------


def test_rates_are_remembered_per_model_and_device():
    cfg = Config()
    cfg.remember_rate("a/b", "cuda", 31.66)
    cfg.remember_rate("a/b", "cpu", 2.1)
    assert cfg.rates == {"a/b|cuda": 31.66, "a/b|cpu": 2.1}


def test_a_nonsense_rate_is_not_remembered():
    cfg = Config()
    cfg.remember_rate("a/b", "cuda", 0.0)
    assert cfg.rates == {}


# --- machine telemetry ----------------------------------------------------


def test_machine_snapshot_has_the_shape_the_page_expects():
    info = sysinfo.machine().as_dict()
    assert set(info) >= {"cpu_percent", "cpu_count", "ram_used", "ram_total",
                         "gpu_present", "gpu"}
    assert info["cpu_count"] >= 1
    assert info["ram_total"] > 0
    if info["gpu"] is not None:
        assert set(info["gpu"]) >= {"name", "util", "mem_used", "mem_total"}
        assert info["gpu"]["mem_total"] > 0


def test_telemetry_never_raises_without_an_nvidia_gpu(monkeypatch):
    """A machine with no GPU must still render the page, not 500."""
    monkeypatch.setattr(sysinfo, "_nvml_ready", False)
    assert sysinfo.gpu_stats() is None
    assert sysinfo.machine().cpu_count >= 1


# --- estimates ------------------------------------------------------------


@pytest.fixture
def app_with_photos(tmp_path):
    gallery = tmp_path / "shoot"
    gallery.mkdir()
    for i in range(5):
        Image.new("RGB", (32, 32), (i * 30, 40, 40)).save(gallery / f"{i}.jpg", "JPEG")
    (gallery / "notes.txt").write_text("not a photo")

    app = App(Config())
    app._embedder = StubEmbedder()
    app._library = Library(app)
    return app, gallery


def test_estimate_counts_photos_and_predicts_the_time(app_with_photos):
    app, gallery = app_with_photos
    estimate = app.library.estimate(gallery)

    assert estimate["ok"] is True
    assert estimate["photos"] == 5, "non-images must not be counted"
    assert estimate["bytes"] > 0
    assert estimate["rate"] > 0
    assert estimate["eta"] == round(5 / estimate["rate"])
    assert estimate["already_indexed"] == 0


def test_estimate_reports_a_folder_that_is_not_there(app_with_photos, tmp_path):
    app, _ = app_with_photos
    estimate = app.library.estimate(tmp_path / "nowhere")
    assert estimate["ok"] is False
    assert "No such folder" in estimate["message"]


def test_estimate_creates_no_index(app_with_photos):
    app, gallery = app_with_photos
    app.library.estimate(gallery)
    assert not workspace.for_folder(gallery).exists(), "looking must not index"


def test_estimate_uses_a_measured_rate_once_we_have_one(app_with_photos):
    app, gallery = app_with_photos
    before = app.library.estimate(gallery)["rate"]

    arch, checkpoint, _ = tier_spec(app.config.tier, "cpu")
    app.config.remember_rate(f"{arch}/{checkpoint}", "cpu", 2.0)
    after = app.library.estimate(gallery)["rate"]

    assert after == 2.0 and after != before


# --- the settings API -----------------------------------------------------


@pytest.fixture
def client(app_with_photos):
    app, gallery = app_with_photos
    with TestClient(create_app(app)) as test_client:
        yield test_client, app, gallery


def test_settings_separates_what_is_configured_from_what_is_running(client):
    test_client, *_ = client
    body = test_client.get("/api/settings").json()

    assert body["configured"]["tier"] == "fast"
    assert body["configured"]["device"] == "auto", "a preference"
    assert body["running"]["device"] in ("cpu", "cuda"), "a fact"
    assert len(body["tiers"]) == 3
    assert body["machine"]["cpu_count"] >= 1


def test_changing_the_tier_persists_and_drops_the_loaded_model(client):
    test_client, app, _ = client
    assert app.peek_embedder() is not None

    body = test_client.post("/api/settings", json={"tier": "best"}).json()
    assert body["configured"]["tier"] == "best"

    from ctximg import config as config_mod

    assert config_mod.load().tier == "best"
    assert app.peek_embedder() is None, "the old model must not survive the switch"


def test_an_invalid_tier_is_refused(client):
    test_client, *_ = client
    response = test_client.post("/api/settings", json={"tier": "turbo"})
    assert response.status_code == 400
    assert "tier must be" in response.json()["detail"]


def test_machine_endpoint_is_small_and_does_not_load_the_model(client):
    test_client, app, _ = client
    app._embedder = None
    body = test_client.get("/api/machine").json()

    assert set(body) == {"machine", "job", "model_loaded"}
    assert body["model_loaded"] is False
    assert app.peek_embedder() is None, "polling telemetry must stay cheap"


def test_estimate_endpoint(client):
    test_client, _, gallery = client
    body = test_client.get("/api/estimate", params={"folder": str(gallery)}).json()
    assert body["ok"] is True and body["photos"] == 5


def test_job_status_carries_speed_and_time_remaining(client):
    test_client, *_ = client
    job = test_client.get("/api/status").json()["job"]
    assert set(job) >= {"running", "done", "total", "rate", "eta", "elapsed"}


# --- a job must never get stuck claiming to run ---------------------------


def test_a_job_that_fails_stops_claiming_to_run(app_with_photos, monkeypatch):
    """A stuck 'running' flag is what makes the app look hung with no way out."""
    import time

    app, gallery = app_with_photos

    def explode(self):
        raise RuntimeError("model is broken")

    monkeypatch.setattr(App, "embedder", property(explode))
    app._embedder = None

    started, _ = app.library.start_index(gallery)
    assert started

    for _ in range(100):
        if not app.library.job.progress.running:
            break
        time.sleep(0.05)

    assert app.library.job.progress.running is False, "the flag must always clear"
    assert app.library.job.error is not None
    assert "model is broken" in app.library.job.error


def test_stopping_when_nothing_runs_is_reported_not_silent(app_with_photos):
    app, _ = app_with_photos
    stopped, message = app.library.stop()
    assert stopped is False
    assert "Nothing is running" in message


def test_cancel_marks_the_progress_so_the_run_can_wind_down():
    from ctximg.indexer import Progress

    progress = Progress(running=True, phase="embedding")
    progress.cancel()
    assert progress.cancelled is True
    assert progress.phase == "stopping"


def test_cancelling_between_batches_keeps_what_was_embedded(tmp_path):
    """Stopping costs at most the batch in flight, never the whole run."""
    import numpy as np
    from PIL import Image

    from ctximg import workspace
    from ctximg.config import Config
    from ctximg.indexer import Progress, run_index
    from ctximg.store import Store

    gallery = tmp_path / "many"
    gallery.mkdir()
    for i in range(24):
        Image.new("RGB", (32, 32), (i * 9, 20, 20)).save(gallery / f"{i}.jpg", "JPEG")

    progress = Progress()

    class StopsEarly:
        model_id = "f/1"
        dim = 3
        device = "cpu"

        def preprocess(self, image):
            return np.zeros(3, np.float32)

        def encode_images(self, tensors):
            progress.cancel()          # stop requested mid-run
            return np.ones((len(tensors), 3), np.float32)

    space = workspace.for_folder(gallery)
    cfg = Config()
    cfg.batch_size = 4
    with Store(space.db_path) as store:
        result = run_index(store, StopsEarly(), cfg, space.thumbs_dir, [gallery],
                           progress=progress)
        assert result.interrupted is True
        assert 0 < result.embedded < 24, "some work kept, the rest left pending"
        assert store.stats().embedded == result.embedded
        assert progress.running is False
