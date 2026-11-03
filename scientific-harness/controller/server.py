"""Serve loopback operations while a worker evaluates candidates."""

import json
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

from controller.campaign import Campaign
from controller.progress import logger


class ScientificServer(ThreadingHTTPServer):
    """Bind loopback only; trust local callers and never serve files or executable code."""

    def __init__(self, campaign: Campaign, port: int = 8910) -> None:
        self.campaign = campaign
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="scientific-worker")
        self.futures: dict[str, Future] = {}
        self.queue_lock = Lock()
        super().__init__(("127.0.0.1", port), ScientificHandler)
        for evaluation in campaign.snapshot()["evaluations"]:
            self.schedule(evaluation)

    def schedule(self, evaluation: dict[str, object]) -> None:
        if evaluation["status"] != "queued":
            return
        identifier = evaluation["run_id"]
        with self.queue_lock:
            if identifier not in self.futures:
                self.futures[identifier] = self.worker.submit(self.campaign.execute, identifier)

    def wait(self, identifier: str) -> dict[str, object]:
        self.campaign.evaluation(identifier)
        with self.queue_lock:
            future = self.futures.get(identifier)
        if future is not None:
            try:
                future.result(timeout=30)
            except TimeoutError:
                pass
        return self.campaign.evaluation(identifier)

    def server_close(self) -> None:
        super().server_close()
        self.worker.shutdown(wait=True)


class ScientificHandler(BaseHTTPRequestHandler):
    server: ScientificServer

    def do_GET(self) -> None:
        self.handle_operation("GET")

    def do_POST(self) -> None:
        self.handle_operation("POST")

    def handle_operation(self, method: str) -> None:
        try:
            if method == "GET" and self.path == "/v1/evidence":
                result = self.server.campaign.public_evidence()
            elif method == "GET" and self.path == "/v1/campaign":
                result = self.server.campaign.snapshot()
            elif (
                method == "GET"
                and self.path.startswith("/v1/runs/")
                and self.path.endswith("/wait")
            ):
                result = self.server.wait(self.path[len("/v1/runs/") : -len("/wait")])
            elif method == "POST" and self.path == "/v1/evaluations":
                size = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(size))
                keys = {"hypothesis", "reason"}
                if not isinstance(value, dict) or set(value) != keys:
                    raise ValueError(f"request must contain exactly {sorted(keys)}")
                result = self.server.campaign.submit(value["hypothesis"], value["reason"])
                self.server.schedule(result)
            elif method == "POST" and self.path == "/v1/finish":
                size = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(size))
                keys = {"outcome", "reason"}
                if not isinstance(value, dict) or set(value) != keys:
                    raise ValueError(f"request must contain exactly {sorted(keys)}")
                result = self.server.campaign.finish(value["outcome"], value["reason"])
            elif method == "POST" and self.path == "/v1/model_calls/reserve":
                size = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(size))
                result = self.server.campaign.reserve_model_call(value)
            elif method == "POST" and self.path == "/v1/model_calls/settle":
                size = int(self.headers.get("Content-Length", "0"))
                value = json.loads(self.rfile.read(size))
                result = self.server.campaign.settle_model_call(value)
            else:
                self.reply(404, {"error": "unknown controller operation"})
                return
            self.reply(200, result)
        except (ValueError, KeyError) as error:
            if method == "POST" and self.path in (
                "/v1/evaluations",
                "/v1/finish",
                "/v1/model_calls/reserve",
                "/v1/model_calls/settle",
            ):
                logger.warning("Request rejected (%s): %s", self.path, error)
            self.reply(400, {"error": str(error)})

    def reply(self, status: int, value: dict[str, object]) -> None:
        body = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # A lost client wait must not cancel accepted numerical work.
            pass

    def log_message(self, _format: str, *_arguments: object) -> None:
        # Only reviewed progress events are printed, never raw HTTP requests or headers.
        pass
