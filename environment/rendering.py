"""
Rendering bridge for LiquidityDeskEnv.

The environment itself stays pure NumPy so that training runs fast. This module
is the only place that knows about visualisation, and it is imported only when
a demo is actually being rendered.

Architecture
------------
    LiquidityDeskEnv.render_state()   ->  DeskRenderer.push(state)
                                              |
                                      [ websocket, port 8766 ]
                                              |
                                      assets/scene.html  (Three.js)
                                              ^
                                      [ http, port 8765 ]

Two small servers run on daemon threads:

* an ``http.server`` serving ``assets/`` so the browser can load the vendored
  Three.js build with no network access, and
* a websocket server that broadcasts one JSON state snapshot per simulation
  step.

Frames are dropped rather than queued if the browser cannot keep up, so a slow
viewer can never stall the simulation loop.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import http.server
import json
import os
import socketserver
import threading
import webbrowser
from pathlib import Path
from typing import Any

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

DEFAULT_HTTP_PORT = 8765
DEFAULT_WS_PORT = 8766


# --------------------------------------------------------------------------- #
# Static file server
# --------------------------------------------------------------------------- #

class _QuietHTTPHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler without the per-request console spam."""

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        pass

    def end_headers(self) -> None:
        # the scene is edited during development; never let the browser cache it
        self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()


class _ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True

    # SO_REUSEADDR means opposite things on the two platforms. On POSIX it lets
    # a restart reclaim a port still in TIME_WAIT, which is what we want. On
    # Windows it permits binding an address that is actively in use, so a second
    # renderer would silently bind the same port as the first instead of
    # failing and letting start() retry the next one.
    allow_reuse_address = os.name != "nt"


PORT_ATTEMPTS = 25

# Ports are claimed by binding with retry, never by probing first. A probe that
# binds and closes, then binds again later, is a race: two renderers built
# before either starts would both pick the same port. On Windows it is worse
# still, because SO_REUSEADDR lets a probe succeed against a port that is
# already in use, so an occupied port reports as free.


# --------------------------------------------------------------------------- #
# Renderer
# --------------------------------------------------------------------------- #

class DeskRenderer:
    """Serves the Three.js scene and streams environment state to it."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        http_port: int = DEFAULT_HTTP_PORT,
        ws_port: int = DEFAULT_WS_PORT,
        assets_dir: Path | str = ASSETS_DIR,
        open_browser: bool = True,
    ) -> None:
        self.host = host
        self.assets_dir = Path(assets_dir)
        self.open_browser = open_browser

        if not (self.assets_dir / "scene.html").exists():
            raise FileNotFoundError(
                f"scene.html not found in {self.assets_dir}. "
                "The repository ships it under assets/."
            )

        # preferred ports; the actual ones are settled when start() binds
        self._http_pref = http_port
        self._ws_pref = ws_port
        self.http_port = http_port
        self.ws_port = ws_port
        self._bind_error: BaseException | None = None

        self._httpd: _ThreadedHTTPServer | None = None
        self._http_thread: threading.Thread | None = None

        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None
        self._clients: dict[Any, asyncio.Queue] = {}
        self._stop_evt: asyncio.Event | None = None

        self._viewer_evt = threading.Event()
        self._ready_evt = threading.Event()
        self._started = False

        # runtime errors reported back by the browser scene
        self.scene_errors: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.http_port}/scene.html?ws={self.ws_port}"

    def start(self) -> str:
        """Start both servers and optionally open the browser. Returns the URL."""
        if self._started:
            return self.url
        self._start_http()
        self._start_ws()
        if not self._ready_evt.wait(timeout=10.0):
            self.stop()
            raise RuntimeError(
                "websocket server did not come up within 10s"
            ) from self._bind_error
        if self._bind_error is not None:
            err = self._bind_error
            self.stop()
            raise RuntimeError("could not start the websocket server") from err
        self._started = True
        if self.open_browser:
            with contextlib.suppress(Exception):
                webbrowser.open(self.url)
        return self.url

    def wait_for_viewer(self, timeout: float = 30.0) -> bool:
        """Block until the browser scene connects. True if a viewer attached."""
        return self._viewer_evt.wait(timeout=timeout)

    @property
    def viewer_count(self) -> int:
        return len(self._clients)

    def push(self, state: dict[str, Any]) -> None:
        """Broadcast one state snapshot. Never blocks; drops frames if needed."""
        if self._loop is None or not self._clients:
            return
        try:
            message = json.dumps(state, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            return
        with contextlib.suppress(RuntimeError):
            self._loop.call_soon_threadsafe(self._enqueue, message)

    def stop(self) -> None:
        if self._loop is not None and self._stop_evt is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._stop_evt.set)
        if self._ws_thread is not None:
            self._ws_thread.join(timeout=3.0)
        if self._httpd is not None:
            with contextlib.suppress(Exception):
                self._httpd.shutdown()
                self._httpd.server_close()
        if self._http_thread is not None:
            self._http_thread.join(timeout=3.0)
        self._started = False

    def __enter__(self) -> DeskRenderer:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    # ------------------------------------------------------------------ #
    # HTTP
    # ------------------------------------------------------------------ #

    def _start_http(self) -> None:
        handler = functools.partial(
            _QuietHTTPHandler, directory=str(self.assets_dir)
        )
        last: OSError | None = None
        for offset in range(PORT_ATTEMPTS):
            port = self._http_pref + offset
            try:
                self._httpd = _ThreadedHTTPServer((self.host, port), handler)
            except OSError as exc:
                last = exc
                continue
            self.http_port = port
            break
        else:
            raise RuntimeError(
                f"no free http port in "
                f"{self._http_pref}-{self._http_pref + PORT_ATTEMPTS - 1}"
            ) from last

        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="desk-http",
            daemon=True,
        )
        self._http_thread.start()

    # ------------------------------------------------------------------ #
    # WebSocket
    # ------------------------------------------------------------------ #

    def _start_ws(self) -> None:
        self._ws_thread = threading.Thread(
            target=self._run_ws_loop, name="desk-ws", daemon=True
        )
        self._ws_thread.start()

    def _run_ws_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve_ws())
        finally:
            with contextlib.suppress(Exception):
                self._loop.close()

    async def _serve_ws(self) -> None:
        import websockets

        self._stop_evt = asyncio.Event()
        last: OSError | None = None

        for offset in range(PORT_ATTEMPTS):
            port = self._ws_pref + offset
            try:
                async with websockets.serve(
                    self._handle_client,
                    self.host,
                    port,
                    ping_interval=20,
                    max_queue=4,
                ):
                    self.ws_port = port
                    self._ready_evt.set()
                    await self._stop_evt.wait()
                return
            except OSError as exc:
                last = exc
                continue

        # every candidate port was taken; unblock start() so it can report
        self._bind_error = last
        self._ready_evt.set()

    async def _handle_client(self, ws: Any) -> None:
        queue: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._clients[ws] = queue
        self._viewer_evt.set()
        pump = asyncio.create_task(self._pump(ws, queue))
        reader = asyncio.create_task(self._read(ws))
        try:
            await ws.wait_closed()
        finally:
            pump.cancel()
            reader.cancel()
            self._clients.pop(ws, None)

    async def _read(self, ws: Any) -> None:
        """
        Relay diagnostics sent back by the scene.

        A WebGL failure in the browser is otherwise invisible from here: the
        page renders a black canvas with its HUD still drawn on top, and the
        simulation happily streams frames into the void. The scene posts its
        runtime errors back over the same socket so they land in the terminal.
        """
        try:
            async for raw in ws:
                try:
                    payload = json.loads(raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(payload, dict):
                    continue
                if payload.get("type") == "error":
                    label = payload.get("label", "error")
                    message = payload.get("message", "")
                    stack = payload.get("stack", "")
                    self.scene_errors.append(payload)
                    print(
                        f"\n  [scene error] {label}: {message}\n"
                        + ("\n".join(f"      {ln}" for ln in stack.splitlines()[:12]))
                        + "\n",
                        flush=True,
                    )
        except Exception:
            return

    @staticmethod
    async def _pump(ws: Any, queue: asyncio.Queue) -> None:
        while True:
            message = await queue.get()
            try:
                await ws.send(message)
            except Exception:
                return

    def _enqueue(self, message: str) -> None:
        """Runs on the event loop thread. Drops the stalest frame when full."""
        for queue in self._clients.values():
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(message)
