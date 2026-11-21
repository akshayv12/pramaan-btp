"""Exercise the actual HTTP handlers on an ephemeral loopback port."""
import http.client
from http.server import ThreadingHTTPServer
import json
import sys
import tempfile
import threading
import unittest

from website.backend import JobStore
from website.server import make_handler
from test_backend import READY, wait_for


class HttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.store = JobStore(directory=cls.temp.name, python=sys.executable, ready=READY)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cls.store))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)
        cls.store.close()
        cls.temp.cleanup()

    def request(self, method, path, data=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=10)
        body = json.dumps(data).encode() if data is not None else None
        connection.request(method, path, body, {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        raw = response.read()
        result = (response.status, dict(response.getheaders()), raw)
        connection.close()
        return result

    def test_frontend_and_health_share_one_origin(self):
        for path, kind in (("/", "text/html"), ("/app.js", "text/javascript"), ("/styles.css", "text/css"), ("/api/health", "application/json")):
            with self.subTest(path=path):
                status, headers, body = self.request("GET", path)
                self.assertEqual(status, 200)
                self.assertIn(kind, headers["Content-Type"])
                self.assertTrue(body)
                self.assertIn("default-src 'self'", headers["Content-Security-Policy"])

    def test_foreign_host_and_origin_are_rejected(self):
        for headers in ({"Host": "attacker.example"}, {"Origin": "https://attacker.example"}):
            status, _, _ = self.request("POST", "/api/search", {"query": "India"}, headers)
            self.assertEqual(status, 403)

    def test_invalid_inputs_return_errors_without_starting_jobs(self):
        before = len(self.store.jobs)
        for path, data in (("/api/search", {}), ("/api/search", {"query": "India", "provider": "gemini"}), ("/api/task4", {"payload": {}})):
            with self.subTest(path=path):
                status, _, body = self.request("POST", path, data)
                self.assertEqual(status, 400)
                self.assertTrue(json.loads(body)["error"])
        self.assertEqual(len(self.store.jobs), before)

    def test_request_size_limit(self):
        status, _, _ = self.request("POST", "/api/search", None, {"Content-Length": "8000001"})
        self.assertEqual(status, 413)

    def test_unknown_files_are_not_exposed(self):
        for path in ("/../backend.py", "/.env", "/api/jobs/" + "0" * 32, "/api/jobs/not-an-id/log"):
            self.assertEqual(self.request("GET", path)[0], 404)

    def test_search_poll_export_and_history(self):
        status, _, body = self.request("POST", "/api/example", {})
        self.assertEqual(status, 202)
        identity = json.loads(body)["id"]
        completed = wait_for(self.store, identity)
        self.assertEqual(completed["status"], "completed", completed.get("error"))
        status, _, body = self.request("GET", "/api/jobs/" + identity)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["answers"][0]["sources"])
        status, headers, body = self.request("GET", f"/api/jobs/{identity}/json")
        self.assertEqual(status, 200)
        self.assertIn("attachment", headers["Content-Disposition"])
        self.assertIn("task5_output", json.loads(body))
        _, _, body = self.request("GET", "/api/jobs")
        self.assertIn(identity, [job["id"] for job in json.loads(body)["jobs"]])


if __name__ == "__main__":
    unittest.main()
