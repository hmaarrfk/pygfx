"""
Line join seam
==============

* A row of increasingly sharp corners, dashed on top and solid below.
* The dash phase is chosen so that a dash straddles each corner.
* The lines are translucent, which is what makes any seam plain.

Every pixel of a uniformly translucent line over a flat background must
composite to the same value, so a corner is right here only if it is one flat
colour. The dashed row used to fail that from 70 degrees down, in two different
ways depending on the alpha mode: a darker hairline drawn across the inside of
each corner, or -- with `alpha_mode="blend"` -- a brighter patch there instead.

A line used to overlap itself wherever a join is "broken" -- a corner too sharp
to be mitred, which the shader covers with the two segments' own caps instead.
Those faces are coplanar and nothing can arbitrate between them properly. With
a depth test one is dropped, and if the survivor carried a partial antialiasing
alpha its sibling's solid coverage went with it: that was the dark hairline.
With none, both composite and the overlap is painted twice: that was the bright
patch. The shader now divides the corner at the plane that bisects it, so each
face keeps only its own side and the overlap is not drawn at all. The pixel is
inked once, so neither failure has anywhere to happen and the result no longer
depends on the alpha mode.

Dashing is what makes these corners broken joins at all: the shader mitres up to
`max_vec_mag`, which is 100 when solid but drops to 1.5 (about 90 degrees) when
dashing. Hence the solid row below was clean at the same angles all along -- it
is the control, not a second example of the bug. A solid line breaks its joins
too, but only at corners sharp enough that the mitre would run past the segment.

The corners are drawn separately rather than as one polygon so that each gets
the same dash phase, which is what makes the row comparable.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'run'

import numpy as np
from rendercanvas.auto import RenderCanvas, loop
import pygfx as gfx


THICKNESS = 14.0
# The dash phase at the corner is LEG_LENGTH / THICKNESS in dash units, so this
# puts the corner one unit into a two-unit stroke: a dash sits astride it. The
# [2, 2] pattern has a period of 4 and 126 / 14 = 9 = 2 * 4 + 1, so any leg length
# of THICKNESS * (4k + 1) keeps that same phase -- handy if you want longer legs.
LEG_LENGTH = 126.0
ANGLES = [90, 70, 60, 45, 30, 15, 10, 5]

# Two legs meeting at `angle` end up 2 * LEG_LENGTH * sin(angle / 2) apart, which
# for the sharpest corners here is less than the line is wide: 22 units at 10
# degrees and 11 at 5, against a stroke of 14. So below about 10 degrees the two
# legs genuinely lie on top of each other near the apex, and the bright patch you
# see there is the stroke overlapping *itself* -- a hairpin -- not the join being
# drawn twice. That is a different defect, it is not what the bisector cut
# addresses, and no amount of leg length removes it: the legs of a 5 degree corner
# need 160 units before they clear each other, which is the whole neighbourhood of
# the join. The corners from 15 degrees up are the ones that test the join.
DASHED_Y, SOLID_Y, LABEL_Y = 30.0, -220.0, -265.0
GAP = 50.0

# A sharp corner is much narrower than a blunt one, so the columns are packed by
# their actual width; at a fixed pitch the sharp end of the row would be mostly
# empty. The camera is then framed on what the rows actually occupy, so that the
# joins -- the point of the example -- are as large as they can be.
widths = [2 * LEG_LENGTH * np.sin(np.radians(a / 2)) for a in ANGLES]
span = sum(widths) + GAP * (len(ANGLES) - 1)
edge = -span / 2
centers = []
for width in widths:
    centers.append(edge + width / 2)
    edge += width + GAP

top = DASHED_Y + LEG_LENGTH
bottom = LABEL_Y - 40
view_width = span + 100
view_height = (top - bottom) + 60

canvas = RenderCanvas(size=(1600, round(1600 * view_height / view_width)))
renderer = gfx.WgpuRenderer(canvas)
renderer.ppaa = "none"  # the seam is the shader's own, not the AA pass's

scene = gfx.Scene()
scene.add(gfx.Background.from_color("#000"))


def corner(angle_deg, cx, cy):
    """Two legs meeting at `angle_deg`, opening upwards from (cx, cy)."""
    half = np.radians(angle_deg / 2)
    dx, dy = LEG_LENGTH * np.sin(half), LEG_LENGTH * np.cos(half)
    return np.array(
        [[cx - dx, cy + dy, 0], [cx, cy, 0], [cx + dx, cy + dy, 0]], np.float32
    )


for angle, x in zip(ANGLES, centers, strict=True):
    for dashed, y, color in (
        (True, DASHED_Y, (1.0, 0.85, 0.33, 0.4)),
        (False, SOLID_Y, (0.47, 0.68, 1.0, 0.4)),
    ):
        scene.add(
            gfx.Line(
                gfx.Geometry(positions=corner(angle, x, y)),
                gfx.LineMaterial(
                    thickness=THICKNESS,
                    color=color,
                    aa=True,
                    dash_pattern=[2, 2] if dashed else (),
                ),
            )
        )
    label = gfx.Text(
        text=f"{angle}",
        font_size=18,
        screen_space=True,
        anchor="middle-center",
        material=gfx.TextMaterial(color="#888"),
    )
    label.local.position = (x, LABEL_Y, 0)
    scene.add(label)

camera = gfx.OrthographicCamera(view_width, view_height)
camera.local.position = (0, (top + bottom) / 2, 0)
controller = gfx.PanZoomController(camera, register_events=renderer)

canvas.request_draw(lambda: renderer.render(scene, camera))

if __name__ == "__main__":
    print(__doc__)
    loop.run()
