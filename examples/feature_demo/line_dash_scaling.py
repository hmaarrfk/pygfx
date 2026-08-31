"""
Dash scaling
============

How the dash pattern responds to zoom. Scroll to zoom, drag to pan.

All four rows are the same geometry with the same ``dash_pattern``. They do not
look the same, and that is the point: each has settled on a different dash size
for this particular view, within the range its settings allow.

The top row is the default, ``dash_scaling="continuous"``: the dashes keep
exactly the on-screen size asked for, but to do that their number along the
line has to change continuously, so they slide. A dash drifts in proportion to
its distance from the start of its line piece, which is why the far end of the
long line races while the near end barely moves.

The other three are ``dash_scaling="quantized"``, which anchors the pattern to
the object and lets only its period follow the view, snapped to a power of two.
Nothing slides: each time the period changes, every dash splits in two (or
pairs of dashes merge), and no dash ever moves. The dashes therefore have to
change size between splits, over a range of exactly a factor of two -- that
width is what makes them split instead of move, and is not adjustable. What
``dash_max_scale`` chooses is where that factor of two sits relative to the
size you asked for:

* ``dash_max_scale=2`` -- never finer than asked. Dashes grow to twice the
  requested size, then split back to it.
* ``dash_max_scale=sqrt(2)`` (the default) -- straddles the requested size, so
  the dashes are never off by more than sqrt(2) in either direction.
* ``dash_max_scale=1`` -- never coarser than asked. Dashes shrink to half the
  requested size, then merge back to it.

Watch the far end of the long line for the sliding, and watch any one dash to
see it split rather than move.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(1000, 700))
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))


def circle(n, x=0.0, y=0.0, r=55.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [x + r * np.sin(t), y + r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


nanpoint = np.full((1, 3), np.nan, np.float32)


def geometry(y):
    """A long straight line plus two loops, so drift shows up at both ends."""
    straight = np.array([[-430, y + 70, 0], [430, y + 70, 0]], np.float32)
    return gfx.Geometry(
        positions=np.vstack(
            [
                straight,
                nanpoint,
                circle(96, -150, y),
                nanpoint,
                circle(6, 150, y),
            ]
        ).astype(np.float32)
    )


# dash_scaling, dash_max_scale, colour, label
VARIANTS = [
    ("continuous", None, "#f55", "continuous: exact size, but slides"),
    ("quantized", 2.0, "#5f5", "quantized, dash_max_scale=2: never finer than asked"),
    (
        "quantized",
        2**0.5,
        "#59f",
        "quantized, dash_max_scale=sqrt(2): closest to asked",
    ),
    ("quantized", 1.0, "#fd5", "quantized, dash_max_scale=1: never coarser than asked"),
]

for i, (dash_scaling, dash_max_scale, color, label) in enumerate(VARIANTS):
    y = 240 - i * 170
    material = gfx.LineMaterial(
        thickness=8,
        color=color,
        loop=True,
        aa=True,
        dash_pattern=[2, 2],
        thickness_space="screen",
        dash_scaling=dash_scaling,
    )
    if dash_max_scale is not None:
        material.dash_max_scale = dash_max_scale
    scene.add(gfx.Line(geometry(y), material))

    text = gfx.Text(
        text=label,
        font_size=15,
        screen_space=True,
        anchor="middle-left",
        material=gfx.TextMaterial(color=color),
    )
    text.local.position = (-430, y + 95, 0)
    scene.add(text)

camera = gfx.OrthographicCamera(1000, 700)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
