"""Serve Pramaan's browser frontend and API on one local origin."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import signal
import sys
import uuid
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from website.backend import JobStore
from website.local_llm import LocalRuntime

STATIC = Path(__file__).parent / "static"
FILES = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
         "/app.js": ("app.js", "text/javascript"), "/ui.js": ("ui.js", "text/javascript"),
         "/styles.css": ("styles.css", "text/css"), "/favicon.svg": ("favicon.svg", "image/svg+xml")}


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, code, data, content_type="application/json; charset=utf-8", filename=None):
            raw = json.dumps(data, ensure_ascii=False, allow_nan=False).encode() if isinstance(data, (dict, list)) else data
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", 'attachment; filename="' + filename + '"')
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def local_request(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in allowed:
                self.send(403, {"error": "Open the website using its localhost address"})
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {"http://" + host for host in allowed}:
                self.send(403, {"error": "Submit searches from the local website"})
                return False
            return True

        def do_GET(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            try:
                if path in FILES:
                    name, kind = FILES[path]
                    self.send(200, (STATIC / name).read_bytes(), kind + "; charset=utf-8")
                elif path == "/api/health":
                    self.send(200, store.ready)
                elif path == "/api/jobs":
                    self.send(200, {"jobs": store.history()})
                elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}", path):
                    self.send(200, store.snapshot(path.split("/")[3], details=True))
                elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}/(json|text|log)", path):
                    raw, kind, name = store.download(path.split("/")[3], path.split("/")[4])
                    self.send(200, raw, kind, name)
                else:
                    self.send(404, {"error": "Not found"})
            except KeyError:
                self.send(404, {"error": "That search or result was not found"})

        def do_POST(self):
            if not self.local_request():
                return
            path = urlsplit(self.path).path
            try:
                if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                    raise ValueError("Send application/json")
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 8_000_000:
                    self.send(413, {"error": "Request must be smaller than 8 MB"})
                    return
                self.connection.settimeout(15)
                data = json.loads(self.rfile.read(size))
                if path=='/api/gemini/configure':
                    if not isinstance(data,dict):raise ValueError('Send a JSON object')
                    self.send(200,store.configure_gemini(data))
                elif path in {"/api/search", "/api/example", "/api/task4"}:
                    mode = {"/api/search": "live", "/api/example": "example", "/api/task4": "task4"}[path]
                    self.send(202, store.submit(data, mode))
                elif re.fullmatch(r"/api/jobs/[a-f0-9]{32}/cancel", path):
                    self.send(200, store.cancel(path.split("/")[3]))
                else:
                    self.send(404, {"error": "Not found"})
            except KeyError:
                self.send(404, {"error": "Search not found"})
            except (ValueError, TypeError, TimeoutError) as exc:
                self.send(400, {"error": str(exc)[:400]})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pramaan research website — browser frontend and Python backend")
    parser.add_argument("--port", type=int, default=8790)
    parser.add_argument("--python", help="Existing pipeline Python interpreter")
    parser.add_argument("--runs-dir", type=Path)
    parser.add_argument('--instance-id',help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("Choose a port between 1 and 65535")
    runtime=LocalRuntime()
    store=server=None
    try:
        runtime.start()
        store = JobStore(directory=args.runs_dir, python=args.python)
        store.ready.update(application='pramaan',instance_id=args.instance_id or uuid.uuid4().hex)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(store))
        server.daemon_threads = True
        print(f"Pramaan website: http://127.0.0.1:{server.server_port}", flush=True)
        def stop_server(signum,frame):raise KeyboardInterrupt
        signal.signal(signal.SIGTERM,stop_server)
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:server.server_close()
        if store is not None:store.close()
        runtime.close()


if __name__ == "__main__":
    main()
