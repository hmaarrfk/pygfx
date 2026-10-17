"""
Light Effects
=============

Lighting effect demonstration examples with adjustable parameters
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import math

from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la

from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui


canvas = RenderCanvas(title="Light_viewer", size=(1000, 600), max_fps=60)
renderer = gfx.renderers.WgpuRenderer(canvas)
scene = gfx.Scene()

mesh = gfx.Mesh(
    gfx.torus_knot_geometry(10, 3, 128, 32),
    material=gfx.MeshPhongMaterial(color="#00aaff"),
)
scene.add(mesh)

# Point Light1
point_light1 = gfx.PointLight("#ffffff")
point_light1.add(gfx.PointLightHelper())
scene.add(point_light1)

# Point Light2
point_light2 = gfx.PointLight("#80ff80")
point_light2.visible = False
point_light2.add(gfx.PointLightHelper())
scene.add(point_light2)

# Directional light
directional_light = gfx.DirectionalLight("#ffff00")
directional_light.visible = False
directional_light.world.x = -25
directional_light.world.y = 20
directional_light.add(gfx.DirectionalLightHelper(5))
scene.add(directional_light)

# Ambient light
ambient_light = gfx.AmbientLight()
scene.add(ambient_light)

camera = gfx.PerspectiveCamera(70, 16 / 9)
camera.world.z = 50
camera.show_pos((0, 0, 0))

gfx.OrbitController(camera, register_events=renderer)


# The state of the panel that is not a property of the scene
state = {
    "back_side": False,
    "auto_rotate": False,
    "light1_move": False,
    "light2_move": False,
}
t1 = 0
t2 = 0
scale = 30


def place_lights():
    point_light1.world.x = math.sin(t1 + math.pi / 3) * scale
    point_light1.world.y = math.sin(t1 + 1) * 5 + 15
    point_light1.world.z = math.cos(t1 + math.pi / 3) * scale

    point_light2.world.x = math.sin(t2 - math.pi / 3) * scale
    point_light2.world.y = math.sin(t2 + 2) * 5 + 15
    point_light2.world.z = math.cos(t2 - math.pi / 3) * scale


place_lights()


def checkbox(name, target, property):
    changed, value = imgui.checkbox(name, bool(getattr(target, property)))
    if changed:
        setattr(target, property, value)
    return value


def color_edit(name, target, property):
    changed, value = imgui.color_edit3(name, getattr(target, property).rgb)
    if changed:
        setattr(target, property, value)


def slider(name, target, property, min, max):
    changed, value = imgui.slider_float(name, getattr(target, property), min, max)
    if changed:
        setattr(target, property, value)


def light_controls(name, light, move_key=None):
    # imgui needs a unique id for widgets with the same label
    imgui.push_id(name)
    enabled = checkbox(name, light, "visible")
    imgui.begin_disabled(not enabled)
    color_edit("Color", light, "color")
    slider("Intensity", light, "intensity", 0, 5)
    if move_key:
        _, state[move_key] = imgui.checkbox("Auto Move", state[move_key])
    imgui.end_disabled()
    imgui.pop_id()
    imgui.separator()


def draw_imgui():
    imgui.set_next_window_size((300, 0), imgui.Cond_.always)
    imgui.set_next_window_pos(
        (canvas.get_logical_size()[0] - 300, 0), imgui.Cond_.always
    )
    imgui.begin("Lights", None, imgui.WindowFlags_.no_collapse)

    material = mesh.material
    checkbox("Flat Shading", material, "flat_shading")
    checkbox("Wireframe", material, "wireframe")
    changed, state["back_side"] = imgui.checkbox("Back Side", state["back_side"])
    if changed:
        material.side = "back" if state["back_side"] else "front"
    _, state["auto_rotate"] = imgui.checkbox("Auto Rotate", state["auto_rotate"])
    color_edit("Diffuse", material, "color")
    color_edit("Specular", material, "specular")
    color_edit("Emissive", material, "emissive")
    slider("Shininess", material, "shininess", 1, 100)
    imgui.separator()

    light_controls("Point Light 1", point_light1, "light1_move")
    light_controls("Point Light 2", point_light2, "light2_move")
    light_controls("Directional Light", directional_light)
    light_controls("Ambient Light", ambient_light)

    imgui.end()


gui_renderer = ImguiRenderer(renderer.device, canvas)
gui_renderer.set_gui(draw_imgui)


def animate():
    global t1, t2

    if state["auto_rotate"]:
        rot = la.quat_from_euler((0.01, 0.02, 0.0))
        mesh.local.rotation = la.quat_mul(rot, mesh.local.rotation)

    if state["light1_move"] and point_light1.visible:
        t1 += 0.01
    if state["light2_move"] and point_light2.visible:
        t2 += 0.02
    place_lights()

    renderer.render(scene, camera)
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    canvas.request_draw(animate)
    loop.run()
