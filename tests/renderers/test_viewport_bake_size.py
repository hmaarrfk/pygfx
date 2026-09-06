"""
Test that bake functions are given the size of the viewport, not of the canvas.

``renderer.render(..., rect=...)`` draws into part of the canvas, and already
computes the logical size of that part in order to set the camera's view size.
Bake functions were handed the size of the whole canvas instead. The only bake
function in pygfx is the line shader's, which uses the size to convert ndc to
logical pixels, so a dashed line drawn into a sub-viewport came out with its
dashes scaled by the ratio between the canvas and the viewport -- about 2.6x too
fine in a typical side-by-side layout.

The test measures a property that does not depend on the adapter or on where the
line happens to land: the on-screen dash period of a line drawn into a viewport
must be the same as when the same line is drawn into the whole canvas.
"""

import numpy as np
import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def render_dashed_line(view_width):
    """A dashed line on a 900x400 canvas, drawn into `view_width` px of it.

    The camera is sized to the area being drawn into, so one model unit is one
    logical pixel however wide that area is, and the dash period can be read
    straight off the image. Thickness 8 with a [2, 2] pattern asks for a period
    of 8 * 4 = 32 px. Passing the full width is the reference: the renderer then
    computes the same logical size for the viewport as for the canvas, so it
    cannot tell the two apart and the bug cannot show.
    """
    target = gfx.Texture(
        dim=2, size=(900, 400, 1), format=wgpu.TextureFormat.rgba8unorm
    )
    renderer = gfx.WgpuRenderer(target)
    renderer.ppaa = "none"
    renderer.pixel_ratio = 1

    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    scene.add(
        gfx.Line(
            gfx.Geometry(positions=np.array([[-140, 0, 0], [140, 0, 0]], np.float32)),
            gfx.LineMaterial(
                thickness=8,
                color="#fff",
                aa=False,
                dash_pattern=[2, 2],
                thickness_space="screen",
            ),
        )
    )
    camera = gfx.OrthographicCamera(view_width, 400)
    renderer.render(scene, camera, flush=False, rect=(0, 0, view_width, 400))
    renderer.flush()
    return renderer.snapshot()[:, :view_width, 0].astype(int)


def dash_period(image):
    """The mean on-screen dash period along the line's own row.

    None if the dashes have merged into one solid run, which is itself a
    symptom rather than a measurement failure: squeeze the pattern far enough
    and the round caps close the gaps.
    """
    ink = image > 128
    assert ink.any(), "nothing was drawn"
    row = ink[int(ink.sum(axis=1).argmax())]
    starts = int((np.diff(row.astype(int)) == 1).sum())
    if starts < 3:
        return None
    lit = np.flatnonzero(row)
    return (lit.max() - lit.min()) / starts


def test_dash_period_is_the_same_in_a_viewport_as_full_canvas():
    """The dashes must not care which part of the canvas they are drawn into.

    300 px against the full 900 is a threefold difference, so a wrong size is
    unmistakable rather than a rounding difference.
    """
    full = dash_period(render_dashed_line(900))
    viewport = dash_period(render_dashed_line(300))

    assert full == pytest.approx(32, rel=0.15), (
        f"the full-canvas reference is itself wrong: {full}"
    )
    assert viewport is not None, (
        "the line drawn into a 300 px viewport came out solid: its pattern was "
        "squeezed until the round caps closed the gaps, which is what "
        "converting ndc with the 900 px canvas size does to a third-width "
        "viewport"
    )
    assert viewport == pytest.approx(full, rel=0.1), (
        f"dashes differ between a 300 px viewport ({viewport:.1f} px) and the "
        f"900 px canvas ({full:.1f} px), whose widths differ by 3.00x"
    )
