"""
Dash scaling
============

The four ways a dash pattern can respond to zoom. Scroll to zoom, drag to pan.

All four rows are the same geometry with the same ``dash_pattern``. What differs
is ``thickness_space``, which says what the pattern is measured in, and
``dash_scaling``, which says how that measurement is allowed to follow the view.

* **screen + continuous** (red) is the default. The dashes keep exactly the
  on-screen size asked for, but to do that their number along the line has to
  change continuously, so they slide. A dash drifts in proportion to its
  distance from the start of its line piece, which is why the far end of the
  long line races while the near end barely moves.

* **screen + quantized** (green) anchors the pattern to the object and lets only
  its period follow the view, snapped to a power of two. Nothing ever slides:
  each time you zoom in by a factor of two, every dash splits in half and a new
  dash appears in each gap. Between splits the dashes breathe on screen by up to
  about 1.5x, and the level is snapped with hysteresis so that a view parked on
  a boundary does not flicker between two levels.

* **model** (blue) measures the pattern in the object's own units, so the
  pattern is painted onto the geometry. Nothing slides here either, but nothing
  splits: zoom in far enough and one dash fills the screen.

* **world** (yellow) is the same, in scene units, so it ignores the object's own
  transform but still scales with the camera.

Note while comparing: ``thickness_space`` governs the line *thickness* as well
as the pattern, so the bottom two rows also get thicker as you zoom in. Only the
top two keep a constant on-screen thickness.
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


# thickness_space, dash_scaling, colour, and what it does while you zoom
VARIANTS = [
    ("screen", "continuous", "#f55", "screen + continuous: constant size, slides"),
    ("screen", "quantized", "#5f5", "screen + quantized: splits, never slides"),
    ("model", "continuous", "#59f", "model: painted on the object, never splits"),
    (
        "world",
        "continuous",
        "#fd5",
        "world: as model, but ignores the object transform",
    ),
]

for i, (thickness_space, dash_scaling, color, label) in enumerate(VARIANTS):
    y = 240 - i * 170
    scene.add(
        gfx.Line(
            geometry(y),
            gfx.LineMaterial(
                thickness=8,
                color=color,
                loop=True,
                aa=True,
                dash_pattern=[2, 2],
                thickness_space=thickness_space,
                dash_scaling=dash_scaling,
            ),
        )
    )
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
