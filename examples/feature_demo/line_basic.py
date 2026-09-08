"""
Line Drawing
============

Drawing a line with a shape that makes it interesting for demonstrating/testing
various aspects of line rendering: acute corners, nodes closer together than the
line is thick, nan-separated pieces, two spirals that close on themselves, and a
z-offset per node so the depth buffer is not boring.

Every knob is on the panel to the right, which sits in its own strip of the
canvas so it never covers the line. The four buttons at the top are the presets
that keys 1-4 used to select, and the keys still work.

Use the middle-mouse button to set the position of the last point -- that sweeps
a single join through every angle, which is the quickest way to see a corner go
from mitred to "broken".

**What to look at.** Press *translucent*, press `p` to turn ppaa off, and
zoom in on the
zigzag corners. A uniformly translucent line must composite to one value
everywhere it is inked, so any corner that is a different shade is a corner drawn
either twice (brighter) or not at all (darker). ``alpha_mode`` decides which of
those two you get, which is why it is on the panel next to the opacity.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx
import pylinalg as la
from wgpu.utils.imgui import ImguiRenderer
from imgui_bundle import imgui


TITLE = "Line Drawing"
CANVAS_SIZE = 1400, 800
PANEL_WIDTH = 340
LABEL_WIDTH = -150

ALPHA_MODES = ["auto", "blend", "solid", "dither", "weighted_blend", "add"]
DASH_PATTERNS = [
    ("none", ()),
    ("2, 2", (2, 2)),
    ("4, 2", (4, 2)),
    ("1, 3", (1, 3)),
    ("uneven", (4, 2, 3, 2, 2, 2, 1, 2, 0, 2)),
]
THICKNESS_SPACES = ["screen", "world", "model"]


canvas = RenderCanvas(size=CANVAS_SIZE, title=TITLE)
renderer = gfx.WgpuRenderer(canvas)

scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))

positions = [[200 + np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [[np.nan, np.nan, np.nan]]
positions += [[400 - np.sin(i) * i * 6, 200 + np.cos(i) * i * 6, 0] for i in range(20)]
positions += [[np.nan, np.nan, np.nan]]
positions += [
    [100, 450, 0],
    [102, 450, 0],
    [104, 450, 0],
    [106, 450, 0],
    [200, 450, 0],
    [200, 445, 0],
    [400, 440, 0],
    [300, 400, 0],
    [300, 390, 0],
    [400, 370, 0],
    [350, 350, 0],
]

# Spiral away in z (to make the depth buffer less boring)
for i in range(len(positions)):
    positions[i][2] = i

geometry = gfx.Geometry(positions=positions)

# Two materials rather than one, because "debug" is a different class. They are
# kept in step with each other, and only the assignment to line.material changes.
material = gfx.LineMaterial(thickness=22.0, color=(0.8, 0.7, 0.0))
debug_material = gfx.LineDebugMaterial(thickness=22.0, color=(0.8, 0.7, 0.0))
line = gfx.Line(geometry, material)
scene.add(line)

camera = gfx.OrthographicCamera(600, 500)
camera.local.position = (300, 250, 0)
home = camera.get_state()

# The render area is the canvas minus the panel strip. The controller measures
# pan and zoom against a viewport's rect, so it gets that rect rather than the
# whole canvas -- otherwise a drag moves the scene by the wrong distance. It is
# also what keeps the panel usable: `pointer_down` and `wheel` are gated on
# `Viewport.is_inside`, so events over the panel do not pan the camera as well.
view = gfx.Viewport(renderer, rect=(0, 0, CANVAS_SIZE[0] - PANEL_WIDTH, CANVAS_SIZE[1]))
controller = gfx.PanZoomController(camera)
renderer.add_event_handler(
    lambda event: controller.handle_event(event, view),
    "pointer_down",
    "pointer_move",
    "pointer_up",
    "key_down",
    "key_up",
    "wheel",
    "before_render",
)

gui_renderer = ImguiRenderer(renderer.device, canvas)

state = {
    "thickness": 22.0,
    "thickness_space": 0,
    "opacity": 1.0,
    "alpha_mode": 0,
    "aa": False,
    "debug": False,
    "loop": False,
    "dash_pattern": 0,
    "dash_offset": 0.0,
    "animate_dash": False,
    "min_node_distance": 0.0,
}

# The presets that keys 1-4 used to build.
PRESETS = {
    "solid": dict(opacity=1.0, aa=False, dash_pattern=0, debug=False, alpha_mode=0),
    "translucent": dict(opacity=0.5, aa=True, dash_pattern=0, debug=False),
    "dashed": dict(opacity=0.5, aa=True, dash_pattern=4, debug=False),
    "debug": dict(opacity=1.0, aa=False, dash_pattern=0, debug=True),
}
applied = {}


def apply_state():
    if state == applied:
        return
    applied.clear()
    applied.update(state)

    for mat in (material, debug_material):
        mat.thickness = state["thickness"]
        mat.thickness_space = THICKNESS_SPACES[state["thickness_space"]]
        mat.opacity = state["opacity"]
        mat.alpha_mode = ALPHA_MODES[state["alpha_mode"]]
        mat.aa = state["aa"]
        mat.loop = state["loop"]
        mat.dash_pattern = DASH_PATTERNS[state["dash_pattern"]][1]
        mat.dash_offset = state["dash_offset"]
        mat.min_node_distance = state["min_node_distance"]

    wanted = debug_material if state["debug"] else material
    if line.material is not wanted:
        line.material = wanted


@renderer.add_event_handler("key_down")
def change_material(event):
    """The original key bindings, kept, now driving the same state as the panel."""
    if event.key == "1":
        state.update(PRESETS["solid"])
    elif event.key == "2":
        state.update(PRESETS["translucent"])
    elif event.key == "3":
        state.update(PRESETS["dashed"])
    elif event.key == "4":
        state.update(PRESETS["debug"])
    elif event.key == "o":
        state["dash_offset"] += 4
    elif event.key == "a":
        state["aa"] = not state["aa"]
    elif event.key == "p":
        renderer.ppaa = "ddaa" if renderer.ppaa == "none" else "none"
    elif event.key == "r":
        renderer.pixel_ratio = 2 if renderer.pixel_ratio == 1 else 1
    renderer.request_draw()


@renderer.add_event_handler("pointer_move", "pointer_down")
def set_last_node(event):
    if event.modifiers:
        return
    if 3 in event.buttons or event.button == 3:
        # Against the viewport, not the canvas: the panel takes up the rest of
        # the width, so using the canvas size would land the point short.
        x0, y0, w, h = view.rect
        ndcx, ndcy = 2 * (event.x - x0) / w - 1, 1 - 2 * (event.y - y0) / h
        pos = la.vec_transform((ndcx, ndcy, 0), np.linalg.pinv(camera.camera_matrix))
        line.geometry.positions.data[-1, :2] = pos[0], pos[1]
        line.geometry.positions.update_range(len(positions) - 1, 1)
        renderer.request_draw()


def draw_imgui():
    display = gui_renderer.backend.io.display_size
    imgui.set_next_window_size((PANEL_WIDTH, display.y), imgui.Cond_.always)
    imgui.set_next_window_pos((display.x - PANEL_WIDTH, 0), imgui.Cond_.always)
    is_expand, _ = imgui.begin(
        TITLE,
        None,
        flags=imgui.WindowFlags_.no_move
        | imgui.WindowFlags_.no_resize
        | imgui.WindowFlags_.no_collapse,
    )
    if is_expand:
        imgui.push_item_width(LABEL_WIDTH)

        imgui.separator_text("Presets (keys 1-4)")
        for i, (name, preset) in enumerate(PRESETS.items()):
            if i:
                imgui.same_line()
            if imgui.button(name):
                state.update(preset)
        if imgui.button("reset view"):
            camera.set_state(home)

        imgui.separator_text("The line")
        _, state["thickness"] = imgui.slider_float(
            "thickness", state["thickness"], 0.5, 60.0
        )
        _, state["thickness_space"] = imgui.combo(
            "thickness_space",
            state["thickness_space"],
            THICKNESS_SPACES,
            len(THICKNESS_SPACES),
        )
        _, state["aa"] = imgui.checkbox("aa (key a)", state["aa"])
        _, state["loop"] = imgui.checkbox("loop", state["loop"])
        imgui.set_item_tooltip(
            "Close each nan-separated piece. The two spirals become loops; the\n"
            "third piece does too, which sends a segment back across the scene."
        )

        imgui.separator_text("Transparency")
        _, state["opacity"] = imgui.slider_float("opacity", state["opacity"], 0.05, 1.0)
        _, state["alpha_mode"] = imgui.combo(
            "alpha_mode", state["alpha_mode"], ALPHA_MODES, len(ALPHA_MODES)
        )
        imgui.set_item_tooltip(
            "What happens where the line covers a pixel more than once, which is\n"
            "what an acute corner does.\n\n"
            "Under 'auto' the depth test keeps one fragment and drops the other,\n"
            "so an overlap can come out darker than the rest of the stroke.\n"
            "Under 'blend' nothing arbitrates and both composite, so the same\n"
            "overlap comes out brighter. Neither is right: a uniformly\n"
            "translucent line should be one flat value everywhere."
        )

        imgui.separator_text("Dashing")
        _, state["dash_pattern"] = imgui.combo(
            "dash_pattern",
            state["dash_pattern"],
            [name for name, _ in DASH_PATTERNS],
            len(DASH_PATTERNS),
        )
        dashed = bool(DASH_PATTERNS[state["dash_pattern"]][1])
        imgui.begin_disabled(not dashed)
        _, state["dash_offset"] = imgui.slider_float(
            "dash_offset (o)", state["dash_offset"] % 8.0, 0.0, 8.0
        )
        _, state["animate_dash"] = imgui.checkbox("animate", state["animate_dash"])
        imgui.end_disabled()
        if dashed:
            imgui.text_wrapped(
                "Dashing lowers the mitre limit from 100 to 1.5 (about 90 "
                "degrees), so far more corners become 'broken' -- rendered as "
                "two capped segments rather than one mitred join."
            )

        imgui.separator_text("Geometry")
        _, state["min_node_distance"] = imgui.slider_float(
            "min_node_distance", state["min_node_distance"], 0.0, 30.0
        )
        imgui.set_item_tooltip(
            "Skip nodes closer together than this many logical pixels on screen,\n"
            "so the neighbours join past them. 0 is off.\n\n"
            "It removes the overlap where a segment is shorter than the line is\n"
            "thick -- but it does so by moving the line, and the deviation is as\n"
            "large as the threshold. Around 12 it also eats the innermost turn of\n"
            "each spiral. Treat it as a decimation knob, not a fix."
        )
        _, state["debug"] = imgui.checkbox("debug material (key 4)", state["debug"])
        imgui.set_item_tooltip(
            "Draw the triangles the line is actually made of. Watch the mitre\n"
            "spike grow past a corner as you drag the last point with the\n"
            "middle mouse button and the angle sharpens."
        )

        imgui.pop_item_width()
    imgui.end()


gui_renderer.set_gui(draw_imgui)


def animate():
    if state["animate_dash"]:
        state["dash_offset"] += 0.1
    apply_state()

    width, height = canvas.get_logical_size()
    view.rect = 0, 0, max(1, width - PANEL_WIDTH), height
    renderer.render(scene, camera, flush=False, rect=view.rect)
    renderer.flush()
    gui_renderer.render()
    canvas.request_draw()


if __name__ == "__main__":
    print(__doc__)
    renderer.request_draw(animate)
    loop.run()
