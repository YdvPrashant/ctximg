"""The web layer, including the guard on serving original files."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ctximg import images as imglib
from ctximg import workspace
from ctximg.app import App
from ctximg.config import Config
from ctximg.library import Library
from ctximg.scan import FileEntry
from ctximg.store import Store
from ctximg.web.server import create_app


class StubEmbedder:
    model_id = "stub/v1"
    dim = 3
    device = "cpu"
    precision = "fp32"
    tier = "fast"
    image_size = 224

    def describe(self):
        return {"model": self.model_id, "device": self.device,
                "precision": self.precision, "dim": self.dim,
                "tier": self.tier, "image_size": self.image_size, "arch": "stub"}

    def encode_query(self, query, ensemble=True):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


def make_photo(path, colour=(180, 40, 40), size=(80, 60)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, colour).save(path, "JPEG")
    return path


@pytest.fixture
def client(tmp_path):
    gallery = tmp_path / "gallery"
    inside = make_photo(gallery / "inside.jpg")
    # A row pointing outside its own folder: what the path guard must catch.
    outside = make_photo(tmp_path / "private" / "outside.jpg")

    space = workspace.for_folder(gallery)
    ids = {}
    with Store(space.db_path) as store:
        store.set_meta(workspace.ROOT_KEY, str(gallery.resolve()))
        store.ensure_model("stub/v1", 3)
        store.sync_files([FileEntry(str(inside), 10, 1.0),
                          FileEntry(str(outside), 10, 1.0)])
        for image_id, path in store.pending():
            store.save_vectors([(image_id, np.array([1.0, 0.0, 0.0], np.float32),
                                 {"width": 80, "height": 60})])
            ids[path] = image_id

    app = App(Config())
    app._embedder = StubEmbedder()
    app._library = Library(app)

    with TestClient(create_app(app)) as test_client:
        yield test_client, app, space.key, ids, str(inside), str(outside)


def test_page_loads(client):
    test_client, *_ = client
    response = test_client.get("/")
    assert response.status_code == 200
    assert "Describe a photo" in response.text


def test_ping_identifies_the_app(client):
    """A second launch uses this to tell our app from a stranger on the port."""
    test_client, *_ = client
    assert test_client.get("/api/ping").text.strip() == "ctximg"


def test_search_returns_ranked_results_with_folder_labels(client):
    test_client, _, key, *_ = client
    body = test_client.get("/api/search", params={"q": "anything", "k": 5}).json()

    assert body["count"] == 2
    first = body["results"][0]
    assert set(first) >= {"id", "folder", "folder_name", "score", "match", "rank", "path"}
    assert first["folder"] == key
    assert first["rank"] == 1
    scores = [r["score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)


def test_search_rejects_an_empty_query(client):
    test_client, *_ = client
    assert test_client.get("/api/search", params={"q": ""}).status_code == 422


def test_search_k_is_bounded(client):
    test_client, *_ = client
    assert test_client.get("/api/search", params={"q": "x", "k": 0}).status_code == 422
    assert test_client.get("/api/search", params={"q": "x", "k": 9999}).status_code == 422


def test_similar_excludes_the_source_photo(client):
    test_client, _, key, ids, inside, _ = client
    body = test_client.get(
        "/api/similar", params={"folder": key, "id": ids[inside]}
    ).json()
    assert ids[inside] not in [r["id"] for r in body["results"]]


# --- media, and the path guard -------------------------------------------


def test_image_serves_a_file_inside_its_folder(client):
    test_client, _, key, ids, inside, _ = client
    response = test_client.get(f"/image/{key}/{ids[inside]}")
    assert response.status_code == 200
    assert response.content[:2] == b"\xff\xd8"  # JPEG magic


def test_image_refuses_a_file_outside_its_indexed_folder(client):
    test_client, _, key, ids, _, outside = client
    response = test_client.get(f"/image/{key}/{ids[outside]}")
    assert response.status_code == 403
    assert "outside" in response.json()["detail"]


def test_image_404s_for_an_unknown_folder_or_image(client):
    test_client, _, key, *_ = client
    assert test_client.get(f"/image/nosuchfolder/1").status_code == 404
    assert test_client.get(f"/image/{key}/999999").status_code == 404


def test_image_404s_when_the_file_is_gone(client, tmp_path):
    test_client, _, key, ids, inside, _ = client
    (tmp_path / "gallery" / "inside.jpg").unlink()
    assert test_client.get(f"/image/{key}/{ids[inside]}").status_code == 404


def test_thumb_is_generated_on_a_cache_miss(client):
    test_client, _, key, ids, inside, _ = client
    info = workspace.find(key)
    cached = imglib.thumb_path(info.workspace().thumbs_dir, ids[inside])
    assert not cached.exists()

    assert test_client.get(f"/thumb/{key}/{ids[inside]}").status_code == 200
    assert cached.exists(), "the miss must populate the cache"


def test_thumb_of_an_out_of_folder_file_is_refused(client):
    test_client, _, key, ids, _, outside = client
    assert test_client.get(f"/thumb/{key}/{ids[outside]}").status_code == 403


# --- folders --------------------------------------------------------------


def test_status_reports_folders_and_model(client):
    test_client, *_ = client
    body = test_client.get("/api/status").json()
    assert len(body["folders"]) == 1
    assert body["folders"][0]["photos"] == 2
    assert body["model"] == "stub/v1"
    assert body["job"]["running"] is False


def test_folders_can_be_selected_and_deselected(client):
    test_client, _, key, *_ = client

    body = test_client.post("/api/folders/select", json={"keys": []}).json()
    assert body["folders"][0]["selected"] is True, "empty selection means all"

    body = test_client.post("/api/folders/select", json={"keys": [key]}).json()
    assert body["folders"][0]["selected"] is True


def test_adding_a_folder_that_does_not_exist_is_reported(client, tmp_path):
    test_client, *_ = client
    response = test_client.post(
        "/api/folders/add", json={"folder": str(tmp_path / "nope")}
    )
    assert response.status_code == 409
    assert "No such folder" in response.json()["message"]


def test_reindexing_an_unknown_folder_404s(client):
    test_client, *_ = client
    assert test_client.post("/api/folders/nosuchkey/reindex").status_code == 404


def test_forget_removes_the_folder_but_not_the_photos(client, tmp_path):
    test_client, _, key, *_ = client
    response = test_client.delete(f"/api/folders/{key}")

    assert response.status_code == 200
    assert response.json()["folders"] == []
    assert (tmp_path / "gallery" / "inside.jpg").exists()


def test_forget_an_unknown_folder_404s(client):
    test_client, *_ = client
    assert test_client.delete("/api/folders/nosuchkey").status_code == 404
