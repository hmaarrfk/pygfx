"""
Tests for the srgb <-> physical colour conversion in the shaders.

These conversions no longer use pow(); see the comment at the top of
pygfx/renderers/wgpu/wgsl/colorspace.wgsl for why. The property that matters
for correctness, and the one the approximations were fitted under, is that
every 8-bit srgb value survives a round trip through the renderer unchanged.
"""

import numpy as np
import pygfx as gfx
from rendercanvas.offscreen import RenderCanvas


ALL_256 = np.arange(256, dtype=np.uint8)


def _render_ramp(colors):
    """Render one opaque quad per given 8-bit value and read the result back."""
    n = len(colors)
    canvas = RenderCanvas(size=(n, 1))
    renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1)
    scene = gfx.Scene()
    for i, c in enumerate(colors):
        v = float(c) / 255
        mesh = gfx.Mesh(
            gfx.plane_geometry(1, 1),
            gfx.MeshBasicMaterial(color=(v, v, v, 1.0)),
        )
        mesh.local.x = i + 0.5
        scene.add(mesh)
    camera = gfx.OrthographicCamera()
    camera.show_rect(0, n, -0.5, 0.5)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    return np.asarray(renderer.target.draw())[0]


def test_every_8bit_value_round_trips_exactly():
    """Every one of the 256 8-bit srgb values must come out of the renderer
    as itself, through srgb2physical() on the way in and physical2srgb() on
    the way out."""
    result = _render_ramp(ALL_256)
    for channel in range(3):
        got = result[:, channel]
        bad = np.nonzero(got != ALL_256)[0]
        assert len(bad) == 0, (
            f"channel {channel}: {len(bad)} of 256 values did not round trip, "
            f"first few: {[(int(v), int(got[v])) for v in bad[:8]]}"
        )


def test_round_trip_is_monotonic():
    """A brighter input must never come out darker. Guards against a refit of
    the approximations introducing a wobble."""
    result = _render_ramp(ALL_256)[:, 0].astype(int)
    assert np.all(np.diff(result) >= 0), "round trip is not monotonic"


def test_python_side_conversions_agree_with_the_definition():
    """The Python-side helpers are exact; they define what the shader
    approximations are fitted to."""
    from pygfx.utils.color import _physical2srgb, _srgb2physical

    for i in range(256):
        c = i / 255
        assert abs(_physical2srgb(_srgb2physical(c)) - c) < 1e-6
        assert round(_physical2srgb(_srgb2physical(c)) * 255) == i
