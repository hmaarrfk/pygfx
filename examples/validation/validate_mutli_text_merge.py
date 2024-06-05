"""
Text justification
==================

Example demonstrating the capabilities of text to be aligned and justified
according to the user's decision.

This demo enables one to interactively control the alignment and the
justification of text anchored to the center of the screen.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'


from wgpu.gui.auto import WgpuCanvas, run
import pygfx as gfx
import numpy as np

scene = gfx.Scene()


scene.add(gfx.Background.from_color("#fff", "#000"))
text = (
    " \n  \n\n"  # Some newlines and spaces before the text starts.
    "  Lorem ipsum\n"  # Some space at the very beginning of the line
    "Bonjour World Olá\n"  # some text that isn't equal in line
    "py gfx\n"  # a line with exactly 1 word (with a non breaking space inside)
    "last line  \n"  # a line with some space at the end
    "\n  \n\n"  # Some newlines and space at the end
)

font_size = 20

text_bottom_right = gfx.TextGeometry(
    text=text,
    font_size=font_size,
    screen_space=True,
    text_align="right",
    anchor="bottom-right",
)

text_top_right = gfx.TextGeometry(
    text=text,
    font_size=font_size,
    screen_space=True,
    text_align="right",
    anchor="top-right",
)

text_bottom_left = gfx.TextGeometry(
    text=text,
    font_size=font_size,
    screen_space=True,
    text_align="left",
    anchor="bottom-left",
)

text_top_left = gfx.TextGeometry(
    text=text,
    font_size=font_size,
    screen_space=True,
    text_align="left",
    anchor="top-left",
)

from pygfx.resources import Buffer
import ipdb; ipdb.set_trace()

multi_text_geometry = gfx.Geometry()
multi_text_geometry.screen_space = text_top_left.screen_space
multi_text_geometry.ref_glyph_size = text_top_left.ref_glyph_size

multi_text_geometry.positions=Buffer(np.concatenate([
    text_bottom_right.positions.data
    text_top_right.positions.data,
    text_bottom_left.positions.data,
    text_top_left.positions.data
]))
multi_text_geometry.indices = Buffer(np.concatenate([
    text_bottom_right.indices.data,
    text_top_right.indices.data,
    text_bottom_left.indices.data,
    text_top_left.indices.data,
]))
multi_text_geometry.sizes= Buffer(np.concatenate([
        text_bottom_right.sizes.data,
        text_top_right.sizes.data,
        text_bottom_left.sizes.data,
        text_top_left.sizes.data,
]))

text_merged = gfx.Text(
    multi_text_geometry,
    gfx.TextMaterial(color="#DA9DFF", outline_color="#000", outline_thickness=0.15),
)

points = gfx.Points(
    gfx.Geometry(positions=[(0, 0, 0)]),
    gfx.PointsMaterial(color="#f00", size=10),
)
scene.add(
    text_merged,
    points,
)

camera = gfx.OrthographicCamera(4, 3)

renderer = gfx.renderers.WgpuRenderer(WgpuCanvas(size=(800, 600)))

renderer.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    run()
