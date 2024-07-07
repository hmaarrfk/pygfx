import os
import numpy as np
import imageio.v3 as iio
from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx

if "PYTEST_CURRENT_TEST" not in os.environ:
    import argparse
    parser = argparse.ArgumentParser(description="Frustum Culling Demo")
    # make it a flag with the ability to use no as a prefix
    parser.add_argument(
        "--frustum-culling", action="store_true", help="Enable frustum culling"
    )

    args = parser.parse_args()
    frustum_culling = args.frustum_culling
else:
    frustum_culling = False

canvas = WgpuCanvas()
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

im = iio.imread("imageio:astronaut.png")[:, :, 1]
im_shape = im.shape[:2]

geometry = gfx.Geometry(grid=gfx.Texture(im, dim=2))

image_matrix_shape = (50, 50)

for i in np.ndindex(image_matrix_shape):
    image = gfx.Image(
        geometry=geometry,
        material=gfx.ImageBasicMaterial(
            # Do something silly to show a semblence of difference between the images
            clim=(0 + i[0] * 5, 255 - i[1] * 5),
            opacity=0.5,
        ),
        frustum_culling=frustum_culling,
    )
    image.local.position = (
        i[1] * im_shape[1] * 1.1 - 256,
        i[0] * im_shape[0] * 1.1 - 256,
        0,
    )
    scene.add(image)

camera = gfx.OrthographicCamera(512, 512)
camera.local.scale_y = -1

controller = gfx.PanZoomController(
    camera,
    register_events=renderer,
)


if __name__ == "__main__":
    canvas.request_draw(lambda: renderer.render(scene, camera))
    run()

