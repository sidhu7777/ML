import types
import os

from flask import Flask

os.environ.setdefault("SMTP_HOST", "localhost")
os.environ.setdefault("SMTP_PORT", "587")
os.environ.setdefault("SMTP_USER", "test@example.com")
os.environ.setdefault("SMTP_PASS", "test")
os.environ.setdefault("BASE_URL", "http://127.0.0.1:8080")

import tools.report.routes as report_routes


class ImmediateThread:
    def __init__(self, target=None, args=(), daemon=None):
        self._target = target
        self._args = args
        self.daemon = daemon

    def start(self):
        self._target(*self._args)


def _make_app():
    app = Flask(__name__)
    app.register_blueprint(report_routes.report_bp, url_prefix="/api/report")
    return app


def test_report_status_ready_with_s3_link(monkeypatch):
    report_routes.REPORT_JOBS.clear()

    monkeypatch.setattr(report_routes, "generate_report", lambda **kwargs: None)
    monkeypatch.setattr(
        report_routes,
        "get_project_by_id",
        lambda project_id: {"Download_path": "https://example.com/report.pdf"},
    )
    monkeypatch.setattr(report_routes.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(report_routes, "db", types.SimpleNamespace(engine=object()))

    app = _make_app()
    client = app.test_client()

    resp = client.post("/api/report/generate", json={"project_id": 149, "user_id": 158})
    assert resp.status_code == 202
    report_id = resp.get_json()["report_id"]

    status_resp = client.get(f"/api/report/status/{report_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.get_json()
    assert status_data["status"] == "ready"
    assert status_data["download_url"] == "https://example.com/report.pdf"


def test_report_download_redirects_to_s3_when_local_missing():
    report_id = "report-redirect"
    report_routes.REPORT_JOBS.clear()
    report_routes.REPORT_JOBS[report_id] = {
        "status": "ready",
        "download_url": "https://example.com/report.pdf",
    }

    app = _make_app()
    client = app.test_client()

    resp = client.get(f"/api/report/download/{report_id}", follow_redirects=False)
    assert resp.status_code == 302
    assert resp.headers["Location"] == "https://example.com/report.pdf"


def test_report_events_returns_ready_event_for_completed_job():
    report_id = "report-events"
    report_routes.REPORT_JOBS.clear()
    report_routes.REPORT_JOBS[report_id] = {
        "status": "ready",
        "project_id": 149,
        "user_id": 158,
        "download_url": "https://example.com/report.pdf",
    }

    app = _make_app()
    client = app.test_client()

    resp = client.get(f"/api/report/events/{report_id}")
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    body = resp.get_data(as_text=True)
    assert "event: report_status" in body
    assert '"status": "ready"' in body
    assert '"download_url": "https://example.com/report.pdf"' in body
