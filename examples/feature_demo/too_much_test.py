import os
import random
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import argparse

parser = argparse.ArgumentParser(description="Render Mark with Too Much Text")
parser.add_argument(
    "--render-mask", type=int, default=1, help="Render mask parameter"
)
args = parser.parse_args()
render_mask = args.render_mask

scene = gfx.Scene()

scene.add(gfx.Background.from_color("#fff", "#000"))

material = gfx.TextMaterial(color="#B4F8C8", outline_color="#000", outline_thickness=0.15)
assert material.aa
for i in range(1000):
    text = gfx.Text(
        gfx.TextGeometry(
            text=f"{i}",
            font_size=40,
            screen_space=True,
        ),
        material=material,
        render_mask=render_mask,
    )
    text.local.position = (
        # Random number between -400 and 400
        random.randint(-400, 300),
        random.randint(-300, 200),
        0)

    scene.add(text)

canvas = WgpuCanvas(title=f"Render Mask {render_mask}")
renderer = gfx.WgpuRenderer(canvas)

camera = gfx.OrthographicCamera(800, 600)
controller = gfx.PanZoomController(camera, register_events=renderer)

if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()
