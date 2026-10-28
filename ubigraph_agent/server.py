"""Same-origin local dashboard and event API, using Python's standard HTTP server."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import urllib.request
from urllib.parse import urlsplit, parse_qs
import webbrowser

from .english import english
from .config import AgentConfig
from .engine import V3Engine
from .runtime import AgentRuntime
from ubigraph_drone import DroneService

STATIC = Path(__file__).resolve().parent/"web"


def make_handler(runtime):
    class Handler(BaseHTTPRequestHandler):
        server_version = "UbiGraph/1.0"
        protocol_version = "HTTP/1.1"

        def handle(self):
            try:
                super().handle()
            except (BrokenPipeError, ConnectionResetError):
                # A normal browser reload may close an idle keep-alive socket.
                pass

        def log_message(self, format, *args):
            # Paths and status only; no headers, prompts, payloads or API credentials.
            if "/events" not in self.path:
                print(f"[HTTP] {self.command} {urlsplit(self.path).path} {args[1] if len(args)>1 else ''}", flush=True)

        def send_json(self, data, status=200):
            body = json.dumps(english(data), ensure_ascii=False, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 40000:
                raise ValueError("Invalid request size")
            origin = self.headers.get("Origin")
            allowed = {f"http://{self.headers.get('Host')}", f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
            if origin and origin not in allowed:
                raise ValueError("Submit requests from the local console")
            value = json.loads(self.rfile.read(size))
            if not isinstance(value, dict):
                raise ValueError("Request must be aJSONobject")
            return value

        def do_GET(self):
            path = urlsplit(self.path).path
            try:
                if path == "/api/agents":
                    return self.send_json(runtime.hub.directory())
                if path == "/api/operations":
                    params = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                    if "seed" in params: params["seed"] = int(params["seed"])
                    context = runtime.engine.context(params)
                    overview = runtime.engine.inspect(runtime.engine.names, context)
                    fleet = runtime.drones.snapshot(True)
                    return self.send_json({"context": context, "overview": overview, "fleet": fleet,
                        "agents": runtime.hub.directory()["agents"], "activity": runtime.memory.summary(),
                        "provider": runtime.cfg.public(), "provider_status": runtime.provider_status})
                if path.startswith("/api/drone/plans/") and path.endswith("/commands"):
                    return self.send_json(runtime.drones.command_manifest(path.split("/")[-2]))
                if path == "/api/health":
                    return self.send_json({"status": "ok", "version": "1.0.0", "algorithm": "V3", "drone_simulation_enabled": True,
                                           "provider": runtime.cfg.public(), "provider_status": runtime.provider_status})
                if path == "/api/drone/state":
                    return self.send_json(runtime.drones.snapshot())
                if path == "/api/drone/events":
                    return self.drone_events()
                if path.startswith("/api/drone/missions/"):
                    return self.send_json(runtime.drones.mission(path.split("/")[-1]))
                if path == "/api/provider":
                    return self.send_json({"models": runtime.models, "status": runtime.provider_status})
                if path == "/api/snapshot":
                    params = {k: v[0] for k, v in parse_qs(urlsplit(self.path).query).items()}
                    if "seed" in params:
                        params["seed"] = int(params["seed"])
                    context = runtime.engine.context(params)
                    return self.send_json(runtime.engine.inspect(runtime.engine.names, context))
                if path == "/api/bootstrap":
                    context = runtime.engine.context()
                    return self.send_json({"capabilities": runtime.engine.capabilities(), "context": context,
                         "provider": runtime.cfg.public(), "provider_status": runtime.provider_status, "models": runtime.models,
                         "overview": runtime.engine.inspect(runtime.engine.names, context),
                         "spatial": runtime.engine.spatial("BRUUNS", context), "benchmarks": runtime.engine.benchmarks()})
                if path == "/api/sessions":
                    return self.send_json({"sessions": runtime.memory.sessions()})
                if path.startswith("/api/sessions/"):
                    return self.send_json(runtime.session_view(path.split("/")[-1]))
                if path.startswith("/api/tasks/"):
                    parts = path.strip("/").split("/")
                    if len(parts) == 4 and parts[3] == "events":
                        return self.events(parts[2])
                    if len(parts) == 3:
                        return self.send_json({"task": runtime.memory.task(parts[2]), "events": runtime.memory.events(parts[2])})
                assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                          "/styles.css": ("styles.css", "text/css; charset=utf-8"), "/favicon.svg": ("favicon.svg", "image/svg+xml"),
                          "/product.css": ("product.css", "text/css; charset=utf-8"),
                          "/product.js": ("product.js", "text/javascript; charset=utf-8"),
                          "/drone.js": ("drone.js", "text/javascript; charset=utf-8"), "/drone.css": ("drone.css", "text/css; charset=utf-8")}
                if path not in assets:
                    return self.send_json({"error": "Resource not found"}, 404)
                name, mime = assets[path]
                data = (STATIC/name).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'")
                self.end_headers()
                self.wfile.write(data)
            except (ValueError, FileNotFoundError) as exc:
                self.send_json({"error": runtime.cfg.redact(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except Exception as exc:
                self.send_json({"error": runtime.cfg.redact(exc)[:500]}, 500)

        def do_POST(self):
            try:
                path, data = urlsplit(self.path).path, self.body()
                if path.startswith("/api/drone/"):
                    action = path.removeprefix("/api/drone/")
                    actions = {"plan": runtime.drones.plan, "start": runtime.drones.start,
                               "control": runtime.drones.control, "settings": runtime.drones.settings,
                               "configure": runtime.drones.configure_drone, "reset": runtime.drones.reset}
                    if action == "cancel":
                        return self.send_json(runtime.drones.cancel(data["mission_id"]))
                    if action not in actions:
                        return self.send_json({"error": "Drone API route not found"}, 404)
                    return self.send_json(actions[action](**data))
                if path == "/api/sessions":
                    return self.send_json(runtime.memory.create_session(runtime.engine.context(data.get("context"))))
                if path == "/api/chat":
                    identity = runtime.submit(data["session_id"], data["message"], data.get("context"), data.get("model"), data.get("agent", "orchestrator"), data.get("execution_policy", "review"))
                    return self.send_json({"task_id": identity})
                if path.startswith("/api/tasks/") and path.endswith("/cancel"):
                    runtime.cancel(path.strip("/").split("/")[2])
                    return self.send_json({"status": "cancelling"})
                return self.send_json({"error": "API route not found"}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": runtime.cfg.redact(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError):
                pass

            except Exception as exc:
                self.send_json({"error": runtime.cfg.redact(exc)[:500]}, 500)

        def drone_events(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            try:
                while not runtime.drones.stopping.is_set():
                    value = runtime.drones.snapshot()
                    self.wfile.write(("data: " + json.dumps(english(value), ensure_ascii=False, allow_nan=False) + "\n\n").encode())
                    self.wfile.flush()
                    time.sleep(.25)
            except (BrokenPipeError, ConnectionResetError):
                return

        def events(self, identity):
            runtime.memory.task(identity)
            query = parse_qs(urlsplit(self.path).query)
            last = int(self.headers.get("Last-Event-ID") or query.get("after", [0])[0])
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            self.close_connection = True
            ping = time.monotonic()
            while True:
                events = runtime.memory.events(identity, last)
                for event in events:
                    packet = f"id: {event['id']}\ndata: {json.dumps(english(event),ensure_ascii=False)}\n\n".encode()
                    self.wfile.write(packet)
                    self.wfile.flush()
                    last = event["id"]
                status = runtime.memory.task(identity)["status"]
                if status not in ("queued", "running"):
                    # A terminal event may have committed after this iteration's first read.
                    for event in runtime.memory.events(identity, last):
                        self.wfile.write(f"id: {event['id']}\ndata: {json.dumps(english(event),ensure_ascii=False)}\n\n".encode())
                        self.wfile.flush()
                    return
                if time.monotonic()-ping > 12:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
                    ping = time.monotonic()
                with runtime.lock:
                    job = runtime.jobs.get(identity)
                if job:
                    with job.condition:
                        job.condition.wait(timeout=.6)
                else:
                    time.sleep(.6)

    return Handler


def main():
    parser = argparse.ArgumentParser(description="UbiGraph Urban intelligence workspace")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--open-browser", action="store_true", help="Open browser when ready; reuse an existing service")
    args = parser.parse_args()
    cfg = AgentConfig.load()
    cfg.host, cfg.port = args.host or cfg.host, args.port or cfg.port
    url = f"http://{cfg.host}:{cfg.port}"
    if args.open_browser:
        try:
            with urllib.request.urlopen(url+"/api/health", timeout=1) as response:
                health = json.load(response)
            if health.get("algorithm") == "V3" and health.get("status") == "ok":
                print("Workspace already running:"+url, flush=True)
                webbrowser.open(url)
                return
        except (OSError, ValueError):
            pass
    print("Loading UbiGraph Data and V3 Model...", flush=True)
    runtime = AgentRuntime(cfg, V3Engine(cfg.run_dir), drone_service=DroneService(cfg.runtime_dir/"drone"))
    threading.Thread(target=runtime.discover, daemon=True).start()
    server = ThreadingHTTPServer((cfg.host, cfg.port), make_handler(runtime))
    print(f"UbiGraph Agent: http://{cfg.host}:{server.server_port}", flush=True)
    print(f"ResponsesModel: {cfg.model} · Reasoning effort: {cfg.reasoning_effort} · Browser drone motion simulation enabled (no hardware)", flush=True)
    if args.open_browser:
        threading.Timer(.5, lambda: webbrowser.open(f"http://{cfg.host}:{server.server_port}")).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping service...", flush=True)
    finally:
        runtime.drones.close()
        server.server_close()


if __name__ == "__main__":
    main()
