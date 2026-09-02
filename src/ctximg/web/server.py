"""Local web app. Binds to 127.0.0.1 only - this serves your private photos."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel

from .. import images as imglib
from .. import workspace

STATIC = Path(__file__).parent / "static"


class FolderRequest(BaseModel):
    folder: str


class SelectionRequest(BaseModel):
    keys: list[str]


class SettingsRequest(BaseModel):
    tier: str | None = None
    precision: str | None = None
    device: str | None = None


def create_app(app) -> FastAPI:
    api = FastAPI(title="ctximg", docs_url=None, redoc_url=None)

    def locate(folder_key: str, image_id: int) -> Path:
        """Map (folder, image) to a file, refusing anything outside that folder.

        Both halves come from the URL, so the resolved path is checked against
        the folder's own root before any bytes are served.
        """
        info = workspace.find(folder_key)
        if info is None:
            raise HTTPException(404, "unknown folder")
        from ..store import Store

        with Store(info.workspace().db_path) as store:
            row = store.get_image(image_id)
        if row is None:
            raise HTTPException(404, "unknown image")
        try:
            target = Path(row["path"]).resolve()
            root = info.root.resolve()
        except OSError:
            raise HTTPException(404, "unreadable path") from None
        if target != root and root not in target.parents:
            raise HTTPException(403, "outside its indexed folder")
        if not target.is_file():
            raise HTTPException(404, "file no longer on disk")
        return target

    # --- page -------------------------------------------------------------

    @api.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))

    @api.get("/api/ping", response_class=PlainTextResponse)
    def ping() -> str:
        """Lets a second launch recognise this app rather than a stranger."""
        return "ctximg"

    # --- search -----------------------------------------------------------

    @api.get("/api/search")
    def search(
        q: str = Query(..., min_length=1),
        k: int = Query(80, ge=1, le=500),
        raw: bool = False,
    ):
        query = q.strip()
        if not query:
            return {"query": q, "count": 0, "results": []}
        hits = app.library.search(query, k=k, ensemble=not raw)
        return {"query": query, "count": len(hits),
                "results": [h.as_dict() for h in hits]}

    @api.get("/api/similar")
    def similar(folder: str, id: int, k: int = Query(80, ge=1, le=500)):
        hits = app.library.similar(folder, id, k=k)
        return {"count": len(hits), "results": [h.as_dict() for h in hits]}

    # --- media ------------------------------------------------------------

    @api.get("/thumb/{folder_key}/{image_id}")
    def thumb(folder_key: str, image_id: int):
        source = locate(folder_key, image_id)
        info = workspace.find(folder_key)
        cached = imglib.thumb_path(info.workspace().thumbs_dir, image_id)
        if not cached.exists():
            try:
                image, _ = imglib.open_upright(source)
                imglib.write_thumb(image, cached, app.config.thumb_size)
            except Exception:
                raise HTTPException(404, "no thumbnail") from None
        return FileResponse(cached, media_type="image/jpeg")

    @api.get("/image/{folder_key}/{image_id}")
    def image(folder_key: str, image_id: int):
        return FileResponse(locate(folder_key, image_id))

    # --- folders ----------------------------------------------------------

    @api.get("/api/status")
    def status():
        return app.status()

    @api.get("/api/settings")
    def settings():
        return app.settings()

    @api.post("/api/settings")
    def change_settings(request: SettingsRequest):
        changes = {k: v for k, v in request.model_dump().items() if v is not None}
        if not changes:
            return app.settings()
        from .. import config as config_mod

        try:
            return app.apply_settings(changes)
        except config_mod.ConfigError as exc:
            raise HTTPException(400, str(exc)) from None

    @api.get("/api/machine")
    def machine():
        """Small and cheap: this is polled every second while the page is open."""
        from .. import sysinfo

        job = app.library.job
        return {
            "machine": sysinfo.machine().as_dict(),
            "job": job.as_dict(),
            "model_loaded": app.peek_embedder() is not None,
        }

    @api.get("/api/estimate")
    def estimate(folder: str):
        return app.library.estimate(folder)

    @api.get("/api/folders")
    def folders():
        return {"folders": [f.as_dict() for f in app.library.folders()]}

    @api.post("/api/folders/select")
    def select(request: SelectionRequest):
        return {"folders": [f.as_dict() for f in app.library.set_selected(request.keys)]}

    @api.post("/api/folders/add")
    def add_folder(request: FolderRequest):
        started, message = app.library.start_index(request.folder)
        if not started:
            return JSONResponse({"started": False, "message": message}, 409)
        return {"started": True, "message": message}

    @api.post("/api/folders/{folder_key}/reindex")
    def reindex(folder_key: str, rebuild: bool = False):
        info = workspace.find(folder_key)
        if info is None:
            raise HTTPException(404, "unknown folder")
        if not info.root.is_dir():
            raise HTTPException(409, "that folder is no longer on disk")
        started, message = app.library.start_index(info.root, rebuild=rebuild)
        if not started:
            return JSONResponse({"started": False, "message": message}, 409)
        return {"started": True, "message": message}

    @api.post("/api/folders/stop")
    def stop_indexing():
        stopped, message = app.library.stop()
        return {"stopped": stopped, "message": message}

    @api.post("/api/quit")
    def quit_app():
        """Shut the whole app down. It is local and single-user, so this is fine."""
        import os
        import threading

        def bye():
            threading.Event().wait(0.4)
            os._exit(0)

        threading.Thread(target=bye, daemon=True).start()
        return {"quitting": True}

    @api.delete("/api/folders/{folder_key}")
    def forget(folder_key: str):
        if not app.library.forget(folder_key):
            raise HTTPException(404, "unknown folder")
        return {"folders": [f.as_dict() for f in app.library.folders()]}

    @api.get("/api/browse")
    def browse():
        """Open a native folder picker. Legitimate because this server is local."""
        chosen = _pick_folder()
        return {"path": chosen or ""}

    return api


def _pick_folder() -> str | None:
    """Show the Windows folder dialog on its own thread, and never crash the app."""
    import queue
    import threading

    answer: queue.Queue = queue.Queue(maxsize=1)

    def ask():
        try:
            import tkinter
            from tkinter import filedialog

            root = tkinter.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                answer.put(filedialog.askdirectory(title="Choose a photo folder") or None)
            finally:
                root.destroy()
        except Exception:
            answer.put(None)

    thread = threading.Thread(target=ask, daemon=True)
    thread.start()
    try:
        return answer.get(timeout=180)
    except queue.Empty:
        return None
