"""
Anisotropy Barn Lamp
====================

This example demonstrates the anisotropy effect in a glTF model.
The visually distinct feature is the elongated appearance of the specular reflection.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import imageio.v3 as iio
from rendercanvas.auto import RenderCanvas, loop
import wgpu
import pygfx as gfx

# The material of this model has many texture maps, and each map has a uniform
# buffer of its own (for its uv transform). Together with the buffers for the
# object, the material and the lights, that makes 13 uniform buffers in the
# fragment shader, while WebGPU guarantees 12. So we ask for more, if the GPU
# has it. This must be done before the renderer is created.
limit_name = "max-uniform-buffers-per-shader-stage"
uniform_buffers_needed = 13
adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
if adapter.limits[limit_name] >= uniform_buffers_needed:
    gfx.renderers.wgpu.select_adapter(adapter)
    gfx.renderers.wgpu.set_wgpu_limits(**{limit_name: uniform_buffers_needed})

# Init
canvas = RenderCanvas(size=(1280, 720), title="Anisotropy")
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

uniform_buffers_available = renderer.device.limits[limit_name]
can_draw_model = uniform_buffers_available >= uniform_buffers_needed

directional_light = gfx.DirectionalLight(intensity=2.5)
directional_light.local.position = (0.5, 0, 0.866)
scene.add(directional_light)

# Read cube image and turn it into a 3D image (a 4d array)
env_img = iio.imread("imageio:meadow_cube.jpg")
cube_size = env_img.shape[1]
env_img.shape = 6, cube_size, cube_size, env_img.shape[-1]

# Create environment map
env_tex = gfx.Texture(
    env_img, dim=2, size=(cube_size, cube_size, 6), generate_mipmaps=True
)

gltf_path = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/AnisotropyBarnLamp/glTF-Binary/AnisotropyBarnLamp.glb"

gltf = gfx.load_gltf(gltf_path, quiet=True)

# gfx.print_scene_graph(gltf.scene)  # Uncomment to see the tree structure
if can_draw_model:
    scene.add(gltf.scene)


def add_env_map(obj):
    if isinstance(obj, gfx.Mesh) and isinstance(obj.material, gfx.MeshStandardMaterial):
        obj.material.env_map = env_tex
        obj.material.env_map_intensity = 0.5


gltf.scene.traverse(add_env_map)

scene.add(gfx.AmbientLight(intensity=0.1))

# Create camera and controller
camera = gfx.PerspectiveCamera(45, 640 / 480)
camera.show_object(gltf.scene, view_dir=(1.8, -0.6, -2.7))
controller = gfx.OrbitController(camera, register_events=renderer)

# If the GPU cannot draw the model, say so in the scene, instead of failing
if not can_draw_model:
    scene.add(gfx.Background.from_color("#222"))
    text = gfx.Text(
        text=(
            "This model cannot be drawn on this GPU.\n\n"
            f"Its material needs {uniform_buffers_needed} uniform buffers "
            "in the fragment shader,\n"
            f"this GPU (or browser) allows {uniform_buffers_available}."
        ),
        font_size=24,
        screen_space=True,
        text_align="center",
        anchor="middle-center",
        material=gfx.TextMaterial(color="#fff"),
    )
    text.local.position = camera.local.position + camera.local.forward * 2
    scene.add(text)


def animate():
    renderer.render(scene, camera)
    canvas.request_draw()


if __name__ == "__main__":
    renderer.request_draw(animate)
    loop.run()
