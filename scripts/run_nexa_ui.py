"""Standalone preview runner for the NEXA Robotics & Teenage Engineering Spatial UI.

Provides:
  - Static file serving for nexa_spatial_ui.html
  - Local REST API for intent extraction, multi-candidate sizing, and test revision feedback.

Run:
    python scripts/run_nexa_ui.py
"""
from __future__ import annotations

import http.server
import json
import os
import socketserver
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tesla_lab import nexa_engine  # noqa: E402

HTML_FILE = ROOT / "tesla_lab" / "nexa_spatial_ui.html"
PORT = int(os.environ.get("PORT", 8080))


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class NexaHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, data: dict, status: int = 200) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            with open(HTML_FILE, "rb") as f:
                self.wfile.write(f.read())
            return
        if self.path == "/kinematic":
            kinematic_file = ROOT / "tesla_lab" / "kinematic_ui.html"
            if kinematic_file.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                with open(kinematic_file, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self._send_json({"error": "kinematic_ui.html not found"}, status=404)
            return
        super().do_GET()

    def do_POST(self) -> None:
        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        try:
            body = json.loads(post_data.decode("utf-8")) if post_data else {}
        except Exception:
            body = {}

        if self.path != "/api/analyze-part-photo":
            print(f">>> POST {self.path}: prompt={str(body.get('prompt'))[:80]!r}, mat={body.get('material')!r}", flush=True)
        else:
            print(f">>> POST {self.path} (image length: {len(body.get('image', ''))})", flush=True)

        if self.path == "/api/extract-intent":
            prompt = body.get("prompt", "")
            res = nexa_engine.extract_intent(prompt)
            self._send_json(res)
            return

        elif self.path == "/api/generate-blueprint":
            res = nexa_engine.size_candidates(body)
            self._send_json(res)
            return

        elif self.path == "/api/test-feedback":
            current_cand = body.get("current_candidate", {})
            feedback = body.get("feedback", "")
            rev = body.get("current_rev", "REV 1.0")
            res = nexa_engine.iterate_revision(current_cand, feedback, rev)
            self._send_json(res)
            return

        elif self.path == "/api/export-stl":
            cand = body.get("candidate", {})
            stl_text = nexa_engine.generate_stl_for_candidate(cand)
            payload = stl_text.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "model/stl")
            self.send_header("Content-Disposition", f"attachment; filename={cand.get('id', 'model')}.stl")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)
            return

        elif self.path == "/api/analyze-part-photo":
            image_data = body.get("image", "")
            res = nexa_engine.analyze_part_photo(image_data)
            self._send_json(res)
            return

        elif self.path == "/api/kinematic-state":
            from tesla_lab.kinematic_engine import EngineSpec, compute_cylinder_kinematics
            theta = float(body.get("theta_deg", 0.0))
            rpm = float(body.get("rpm", 2400.0))
            spec = EngineSpec(rpm_nominal=rpm)
            cyls = compute_cylinder_kinematics(spec, theta)
            self._send_json({"cylinders": cyls, "theta_deg": theta})
            return

        self._send_json({"error": "Endpoint not found"}, status=404)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if not HTML_FILE.exists():
        sys.stderr.write(f"Error: Could not find HTML file at {HTML_FILE}\n")
        sys.exit(1)

    ThreadedHTTPServer.allow_reuse_address = True
    port = PORT
    is_cloud = bool(os.environ.get("RAILWAY_ENVIRONMENT") or os.environ.get("PORT") or os.environ.get("RENDER"))
    ports_to_try = [port] if "PORT" in os.environ else range(PORT, PORT + 20)
    for p in ports_to_try:
        try:
            with ThreadedHTTPServer(("", p), NexaHandler) as httpd:
                url = f"http://localhost:{p}"
                print("=" * 68, flush=True)
                print("EVIE // HARDWARE CO-PILOT - SPATIAL UI & REST API", flush=True)
                print("=" * 68, flush=True)
                print(f"Server active: {url} (Listening on port {p})", flush=True)
                print(f"Target file:   {HTML_FILE}", flush=True)
                print("API endpoints: /api/extract-intent, /api/generate-blueprint, /api/test-feedback, /api/export-stl, /api/analyze-part-photo", flush=True)
                print("=" * 68, flush=True)
                if not is_cloud:
                    try:
                        webbrowser.open(url)
                    except Exception:
                        pass
                httpd.serve_forever()
        except OSError:
            continue


if __name__ == "__main__":
    main()
