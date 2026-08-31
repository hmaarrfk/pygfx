"""
Dash scaling
============

Comparing ``LineMaterial.dash_scaling`` while zooming. Scroll to zoom.

The two rows are the same geometry with the same dash pattern, and differ only
in how the pattern responds to the view scale.

* **continuous** (top, red) keeps the on-screen dash size exactly as asked, but
  to do so the number of dashes along the line has to change continuously, so
  the dashes slide along the line as you zoom. A dash drifts by an amount
  proportional to its distance from the start of the line, which is why the far
  end of the long line races while the near end barely moves.

* **quantized** (bottom, green) anchors the pattern to the object and lets only
  its period follow the view scale, snapped to a power of two. The dashes never
  move: every time you zoom in by a factor of two, each dash splits in half and
  a new dash appears in each gap. The price is that between those splits the
  dashes grow and shrink on screen, by at most a factor of sqrt(2).

Watch the far end of the horizontal line, and the seam at the top of the
circles, to see the difference.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


canvas = RenderCanvas(size=(1000, 600))
renderer = gfx.WgpuRenderer(canvas)
scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))


def circle(n, x=0.0, y=0.0, r=60.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [x + r * np.sin(t), y + r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


nanpoint = np.full((1, 3), np.nan, np.float32)


def geometry(y):
    """A long straight line plus two loops, so drift is visible at both ends."""
    straight = np.array([[-420, y + 90, 0], [420, y + 90, 0]], np.float32)
    return gfx.Geometry(
        positions=np.vstack(
            [
                straight,
                nanpoint,
                circle(96, -150, y - 40),
                nanpoint,
                circle(6, 150, y - 40),
            ]
        ).astype(np.float32)
    )


for y, color, scaling in [(140, "#f55", "continuous"), (-140, "#5f5", "quantized")]:
    scene.add(
        gfx.Line(
            geometry(y),
            gfx.LineMaterial(
                thickness=8,
                color=color,
                loop=True,
                aa=True,
                dash_pattern=[2, 2],
                thickness_space="screen",
                dash_scaling=scaling,
            ),
        )
    )

camera = gfx.OrthographicCamera(1000, 600)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
