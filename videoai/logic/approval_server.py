"""The short-lived server that carries a Save click from the page to the disk.

It exists for one reason: a browser page cannot write a file, and every
alternative put the burden on the creator — download this, then run that, then
tell me where it went. That burden is what lost an afternoon's work once
already.

Deliberately small and deliberately temporary. It binds to the loopback
interface only, mints a token for the session so a stray tab cannot post into
the project, serves exactly two paths, and is expected to be stopped as soon as
the approval is done.
"""
from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from videoai.core.store import ArtifactStore
from videoai.logic.approval import ApplyResult, apply_decisions, save_decisions_copy

# Nothing here should ever be reachable from another machine.
HOST = "127.0.0.1"


@dataclass
class ApprovalSession:
    """One approval, and what the creator did during it."""

    project_dir: Path
    page_html: str
    token: str = field(default_factory=lambda: secrets.token_urlsafe(16))
    saves: list[ApplyResult] = field(default_factory=list)
    on_save: object = None

    @property
    def work_dir(self) -> Path:
        return self.project_dir / "work"

    def record(self, document: dict) -> ApplyResult:
        store = ArtifactStore(self.work_dir)
        save_decisions_copy(self.work_dir, document)
        result = apply_decisions(store, document)
        self.saves.append(result)
        if callable(self.on_save):
            self.on_save(result)
        return result


def _handler_for(session: ApprovalSession):
    class Handler(BaseHTTPRequestHandler):
        # The default logger writes a line per request to stderr, which buries
        # the one message the creator actually needs.
        def log_message(self, *args) -> None:  # noqa: D102
            return

        def _authorised(self) -> bool:
            return self.headers.get("X-Approval-Token") == session.token

        def do_GET(self) -> None:  # noqa: N802 - required name
            if self.path.split("?")[0] not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = session.page_html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - required name
            if self.path.split("?")[0] != "/save":
                self.send_error(404)
                return
            if not self._authorised():
                # A page from some other session, or none at all.
                self.send_error(403, "bad approval token")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                document = json.loads(self.rfile.read(length).decode("utf-8"))
                result = session.record(document)
            except Exception as error:  # noqa: BLE001 - reported to the page
                payload = json.dumps({"ok": False, "error": str(error)[:200]})
                body = payload.encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            body = json.dumps({"ok": True, "summary": result.summary()}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def serve(session: ApprovalSession) -> tuple[HTTPServer, str]:
    """Start the server and return it with the URL to open.

    Port 0 lets the OS pick a free one, so two approvals can be open at once and
    neither has to know about the other.
    """
    server = HTTPServer((HOST, 0), _handler_for(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    return server, f"http://{HOST}:{port}/?token={session.token}"
