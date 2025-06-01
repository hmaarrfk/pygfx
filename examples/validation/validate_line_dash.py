"""
Dashing
=======

* Demoing densely sampled lines with dashing.
* Thicker lines show the same pattern (i.e. dash phase increases with thickness).
* Dash offset is expressed in same space as the pattern.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx


canvas = WgpuCanvas(size=(1000, 1000))
renderer = gfx.WgpuRenderer(canvas)


N = 48_000
# N_cameras = (6, 4)
N_cameras = (12, 8)
lines = np.zeros(N_cameras, dtype=object)
scene = gfx.Scene()
material = gfx.LineMaterial(
    thickness=12,
    # dash_pattern=[0, 2, 2, 2],
    color=(0.0, 1.0, 1.0, 1.),
    # dash_offset=0,
)

for i in np.ndindex(N_cameras):
    x = np.sin((np.random.rand(N) * 2 * np.pi).cumsum() / 1000)
    y = np.cos((np.random.rand(N) * 2 * np.pi).cumsum() / 1000)

    positions = np.ascontiguousarray(np.array([
        x * 100,
        y * 100,
        np.zeros_like(x)
    ], np.float32).T)

    line = gfx.Line(
        gfx.Geometry(positions=positions),
        material=material,
    )
    line.material.thickness_space = "screen"
    line.local.position = 220 * i[0], 220 * i[1], 0
    scene.add(line)
    lines[i] = line

camera = gfx.OrthographicCamera()
camera.show_object(scene)

# controller = gfx.OrbitController(camera, register_events=renderer)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    run()
