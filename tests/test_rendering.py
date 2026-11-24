"""
End-to-end tests for the rendering bridge.

These cover everything between the environment and the browser: the static file
server, the vendored Three.js build, the websocket transport, and the shape of
the state frames the scene consumes. WebGL rendering itself is the only part
that cannot be asserted here.

This also doubles as the serialisation contract: if a field the scene reads is
renamed or dropped from `render_state()`, these fail rather than the scene
silently going blank.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

import pytest

from environment import DeskConfig, LiquidityDeskEnv
from environment.rendering import DeskRenderer


@pytest.fixture
def renderer():
    r = DeskRenderer(open_browser=False)
    r.start()
    try:
        yield r
    finally:
        r.stop()


def fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=10) as resp:
        assert resp.status == 200, f"{url} returned {resp.status}"
        return resp.read()


# --------------------------------------------------------------------------- #
# Static assets
# --------------------------------------------------------------------------- #

def test_serves_the_scene_page(renderer):
    body = fetch(f"http://{renderer.host}:{renderer.http_port}/scene.html")
    text = body.decode("utf-8")
    assert "importmap" in text, "the scene needs an importmap to resolve 'three'"
    assert "./js/scene.js" in text
    assert "./vendor/three/build/three.module.min.js" in text


def test_serves_the_scene_module(renderer):
    body = fetch(f"http://{renderer.host}:{renderer.http_port}/js/scene.js")
    text = body.decode("utf-8")
    assert "from 'three'" in text
    assert "three/addons/" in text


@pytest.mark.parametrize(
    "path",
    [
        "/vendor/three/build/three.module.min.js",
        "/vendor/three/examples/jsm/controls/OrbitControls.js",
        "/vendor/three/examples/jsm/postprocessing/EffectComposer.js",
        "/vendor/three/examples/jsm/postprocessing/UnrealBloomPass.js",
        "/vendor/three/examples/jsm/postprocessing/OutputPass.js",
        "/vendor/three/examples/jsm/environments/RoomEnvironment.js",
    ],
)
def test_vendored_three_is_served_offline(renderer, path):
    """Every Three.js file the scene imports must be present in the repo."""
    body = fetch(f"http://{renderer.host}:{renderer.http_port}{path}")
    assert len(body) > 400, f"{path} looks truncated"


def test_scene_has_no_remote_dependencies():
    """A marker with no network must still get the GUI."""
    from environment.rendering import ASSETS_DIR

    for name in ("scene.html", "js/scene.js"):
        text = (ASSETS_DIR / name).read_text(encoding="utf-8")
        for bad in ("https://unpkg.com", "https://cdn.", "https://esm.sh",
                    "http://unpkg.com", "cdn.jsdelivr.net"):
            assert bad not in text, f"{name} still references {bad}"


def test_unknown_path_is_not_served(renderer):
    with pytest.raises(urllib.error.HTTPError) as exc:
        fetch(f"http://{renderer.host}:{renderer.http_port}/nope.html")
    assert exc.value.code == 404


# --------------------------------------------------------------------------- #
# Websocket transport
# --------------------------------------------------------------------------- #

REQUIRED_KEYS = {
    "t", "episode_len", "currencies", "balances_native", "pools_usd",
    "pool_fill", "rates", "rate_change", "vol_regime", "devaluation", "flow",
    "paused", "pressure", "hazard", "spread", "equity_usd", "initial_equity_usd",
    "high_water_usd",
    "drawdown", "action", "action_name", "reward", "rebalance",
    "pending_order", "stats",
}


def collect_frames(renderer: DeskRenderer, count: int) -> list[dict]:
    """Drive the environment and read `count` frames back over the websocket."""
    import websockets

    env = LiquidityDeskEnv(DeskConfig(), render_mode="state")
    env.reset(seed=11)
    frames: list[dict] = []

    async def client() -> None:
        uri = f"ws://{renderer.host}:{renderer.ws_port}"
        async with websockets.connect(uri) as ws:
            # the server registers the connection on its own loop; wait for it
            # or the first pushes are dropped as having no audience
            for _ in range(100):
                if renderer.viewer_count > 0:
                    break
                await asyncio.sleep(0.02)
            assert renderer.viewer_count > 0, "server never registered the client"

            for _ in range(count):
                env.step(env.action_space.sample())
                renderer.push(env.render_state())
                raw = await asyncio.wait_for(ws.recv(), timeout=10)
                frames.append(json.loads(raw))

    asyncio.run(client())
    env.close()
    return frames


def test_websocket_streams_complete_state_frames(renderer):
    frames = collect_frames(renderer, 6)
    assert len(frames) == 6

    for frame in frames:
        missing = REQUIRED_KEYS - set(frame)
        assert not missing, f"state frame is missing {sorted(missing)}"

        assert frame["currencies"] == ["NGN", "RWF", "KES", "USDT"]
        assert len(frame["pools_usd"]) == 4
        assert len(frame["pool_fill"]) == 4
        assert len(frame["rates"]) == 3
        assert len(frame["paused"]) == 3
        assert len(frame["devaluation"]) == 3
        # the scene maps pool_fill straight onto liquid column height
        assert all(0.0 <= v <= 1.0 for v in frame["pool_fill"])
        assert frame["equity_usd"] > 0.0
        assert isinstance(frame["action_name"], str)


def test_frames_advance_in_step_order(renderer):
    frames = collect_frames(renderer, 5)
    steps = [f["t"] for f in frames]
    assert steps == sorted(steps), f"frames arrived out of order: {steps}"
    assert steps[-1] > steps[0]


def test_push_without_a_viewer_is_harmless(renderer):
    """The simulation must never block or raise because nobody is watching."""
    env = LiquidityDeskEnv(DeskConfig(), render_mode="state")
    env.reset(seed=12)
    for _ in range(50):
        env.step(0)
        renderer.push(env.render_state())
    env.close()


def test_state_is_json_serialisable_every_step():
    """json.dumps with allow_nan=False, so a NaN leak fails loudly."""
    env = LiquidityDeskEnv(DeskConfig(), render_mode="state")
    env.reset(seed=13)
    for _ in range(300):
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        json.dumps(env.render_state(), allow_nan=False)
        if term or trunc:
            break
    env.close()


def test_two_renderers_do_not_fight_over_ports():
    a = DeskRenderer(open_browser=False)
    b = DeskRenderer(open_browser=False)
    try:
        a.start()
        b.start()
        assert a.http_port != b.http_port
        assert a.ws_port != b.ws_port
        assert fetch(f"http://{a.host}:{a.http_port}/scene.html")
        assert fetch(f"http://{b.host}:{b.http_port}/scene.html")
    finally:
        a.stop()
        b.stop()
