"""
A line must ink every pixel of a broken join exactly once.

A line used to overlap itself wherever a join is "broken" -- a corner too sharp
to be mitred, which the shader covers with the two segments' own caps instead.
The two faces are coplanar, so nothing can arbitrate between them properly.
With a depth test one is dropped, and where the survivor is an antialiased edge
lying inside its sibling's solid interior the pixel keeps a partial alpha: a
dark hairline across the corner. With no depth test both composite and the
overlap is painted twice: a bright patch, the more obvious of the two.

The shader now divides the corner at the plane that bisects it instead, so each
face keeps only its own side and the overlap is not drawn at all. Because the
pixel is inked once, neither failure has anywhere to happen and the result no
longer depends on the alpha mode or on ``depth_compare``. These tests hold it
to that, in both alpha modes.

The measurement uses a *translucent* line, which is much the better probe:

* it is exact. Every inked pixel of a uniformly translucent line over a flat
  background must be one value, so any departure is a defect and its direction
  says which one -- darker means a fragment was dropped, brighter means one was
  composited twice.
* it is stable. The opaque seam depends on the pixel ratio (invisible at 1,
  224/255 at 2, 241/255 at 4), so an opaque test can silently pass. The
  translucent one shows the defect at every pixel ratio.

Note that dashing is what makes a 60 degree corner a broken join at all: the
shader mitres up to ``max_vec_mag``, which is 100 when solid but drops to 1.5
(about 90 degrees) when dashing. A *solid* 60 degree corner is properly mitred
and is clean.
"""

import numpy as np
import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


SIZE = 200
THICKNESS = 14.0
ALPHA = 0.4
# The criterion is uniformity, not a particular value: a line of one colour and
# one alpha over a flat background must composite to one value everywhere, and
# what that value is does not matter. (It is not 255*ALPHA, incidentally --
# compositing happens in linear space, so 0.4 comes out at 170, not 102.)
# Antialiasing makes the *edge* pixels legitimately different, so the interior
# is eroded before measuring, and a few levels of slack are allowed.
TOLERANCE = 4
# The dash phase at the corner is leg_length / THICKNESS in dash units, so this
# puts the corner one unit into a two-unit stroke: a dash sits astride it.
LEG_LENGTH = 70.0


def erode(mask, iterations=2):
    """Shrink a boolean mask by `iterations` pixels, in all 8 directions."""
    for _ in range(iterations):
        m = mask
        mask = np.pad(
            m[1:-1, 1:-1]
            & m[:-2, 1:-1]
            & m[2:, 1:-1]
            & m[1:-1, :-2]
            & m[1:-1, 2:]
            & m[:-2, :-2]
            & m[:-2, 2:]
            & m[2:, :-2]
            & m[2:, 2:],
            1,
        )
    return mask


def render_corner(
    angle_deg, *, dashed=True, alpha=ALPHA, pixel_ratio=None, alpha_mode=None, **kwargs
):
    """A translucent white corner on black, rendered offscreen.

    The corner sits at the origin with both legs running upwards, so that the
    inside of it -- the interesting part -- is in view.
    """
    half = np.radians(angle_deg / 2)
    dx, dy = LEG_LENGTH * np.sin(half), LEG_LENGTH * np.cos(half)
    positions = np.array([[-dx, dy, 0], [0, 0, 0], [dx, dy, 0]], np.float32)

    target = gfx.Texture(
        dim=2, size=(SIZE, SIZE, 1), format=wgpu.TextureFormat.rgba8unorm
    )
    renderer = gfx.WgpuRenderer(target)
    # The seam is the line shader's own. Post-processing AA would only blur the
    # evidence and make the numbers adapter-dependent.
    renderer.ppaa = "none"
    if pixel_ratio is not None:
        renderer.pixel_ratio = pixel_ratio

    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    scene.add(
        gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=THICKNESS,
                color=(1, 1, 1, alpha),
                aa=True,
                dash_pattern=[2, 2] if dashed else (),
                **({} if alpha_mode is None else {"alpha_mode": alpha_mode}),
                **kwargs,
            ),
        )
    )

    camera = gfx.OrthographicCamera(SIZE, SIZE)
    camera.local.position = (0, 45, 0)
    renderer.render(scene, camera)
    return renderer.snapshot()[..., 0].astype(int)


def interior_range(image):
    """The (min, max) of the pixels strictly inside the ink."""
    interior = erode(image > 40)
    assert interior.any(), "nothing was drawn, so the measurement proves nothing"
    return int(image[interior].min()), int(image[interior].max())


# "auto" is what a LineMaterial picks by default; "blend" is what the line
# examples ask for explicitly. They used to fail in opposite directions -- auto
# dropped a fragment and went dark, blend composited twice and went bright --
# which is why both belong here.
ALPHA_MODES = ["auto", "blend"]


@pytest.mark.parametrize("alpha_mode", ALPHA_MODES)
@pytest.mark.parametrize("angle", [70, 60, 45, 30])
def test_no_seam_across_a_broken_join(angle, alpha_mode):
    """A uniformly translucent line must render to one uniform value.

    This is the requirement. It used to fail as 70..170 in "auto" (a fragment
    dropped, so a dark hairline) and as 170..210 in "blend" (both faces
    composited, so a bright patch).
    """
    low, high = interior_range(render_corner(angle, alpha_mode=alpha_mode))
    assert high - low <= TOLERANCE, (
        f"the inside of a {angle} degree corner is not uniform "
        f"in alpha_mode={alpha_mode!r}: {low}..{high}"
    )


@pytest.mark.parametrize("angle", [90, 70, 60, 45])
def test_a_mitred_join_is_clean(angle):
    """A solid line mitres these corners, and mitred joins do not seam.

    This is the control. It shows the defect belongs to the broken join and not
    to sharp corners as such, and it guards the measurement: if this ever fails,
    the harness is wrong rather than the shader.
    Thirty degrees and sharper is excluded, for a reason that has nothing to do
    with the join: at those angles the two legs of the test's V meet each other
    a long way from the node, and the concave corner where their inner edges
    cross is a pixel or two too dark (a spread of about 18 at 30 degrees). That
    is the ordinary antialiasing error at a concave corner -- coverage is taken
    from the distance to the nearer of the two edges, which underestimates it --
    and it happens the same way whether the join is mitred or broken.
    """
    low, high = interior_range(render_corner(angle, dashed=False))
    assert high - low <= TOLERANCE, f"{angle} degrees: {low}..{high}"


def test_the_corner_does_not_depend_on_depth_compare():
    """`depth_compare="<="` was the obvious candidate, and it was the wrong lever.

    It does remove the *dark* seam, because both coplanar fragments then survive
    instead of one being dropped -- but surviving means both composite, so the
    overlap is painted twice and the seam comes back as a bright patch. At a 60
    degree corner it read 170..210 where the default read 70..170. It moved the
    error rather than removing it, and it is not acceptable as a fix.

    What dividing the corner at the bisector buys is that the question stops
    being asked: the pixel is covered by one face, so no depth comparison can
    make it darker or brighter. This asserts that -- a stricter guard than the
    old one, because it fails the moment overlapping geometry comes back,
    whichever direction the error then takes.
    """
    plain = interior_range(render_corner(60))
    assert plain[1] - plain[0] <= TOLERANCE, f"the corner should be uniform: {plain}"
    for compare in ("<", "<="):
        low, high = interior_range(render_corner(60, depth_compare=compare))
        assert high - low <= TOLERANCE, (
            f"depth_compare={compare!r} changes the corner: {low}..{high}. "
            "Some pixel is being covered more than once again."
        )


@pytest.mark.parametrize("alpha_mode", ALPHA_MODES)
@pytest.mark.parametrize("pixel_ratio", [1, 2, 4])
def test_the_corner_is_uniform_at_every_pixel_ratio(pixel_ratio, alpha_mode):
    """Why this file measures a translucent line and not an opaque one.

    The opaque seam was a resampling coincidence: invisible at pixel ratio 1,
    224/255 at 2, 241/255 at 4. So a fix judged on an opaque line can pass by
    accident at one ratio and fail at another. The translucent measurement is
    the same at all of them, and this pins that down: a fix that only lines up
    the samples at one ratio does not pass here.
    """
    low, high = interior_range(
        render_corner(60, pixel_ratio=pixel_ratio, alpha_mode=alpha_mode)
    )
    assert high - low <= TOLERANCE, f"pixel ratio {pixel_ratio}: {low}..{high}"


# --- the same requirement on the shape from examples/feature_demo/line_basic.py ---
#
# A single V does not exercise everything: that example is a polyline of many
# acute corners with a different z at every node, so the depth test is live, and
# it is where the double drawing was reported. These render it offscreen and ask
# the same question of it.

THICK = 22.0


def render_polyline(points, *, alpha_mode=None, thickness=THICK, size=400):
    """A translucent polyline on black, each node at its own depth."""
    positions = np.array(
        [[x, y, float(i)] for i, (x, y) in enumerate(points)], np.float32
    )
    target = gfx.Texture(
        dim=2, size=(size, size, 1), format=wgpu.TextureFormat.rgba8unorm
    )
    renderer = gfx.WgpuRenderer(target)
    renderer.ppaa = "none"

    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    scene.add(
        gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=thickness,
                color=(1, 1, 1, ALPHA),
                aa=True,
                **({} if alpha_mode is None else {"alpha_mode": alpha_mode}),
            ),
        )
    )
    camera = gfx.OrthographicCamera(size, size)
    camera.local.position = (
        float(positions[:, 0].mean()),
        float(positions[:, 1].mean()),
        0,
    )
    renderer.render(scene, camera)
    return renderer.snapshot()[..., 0].astype(int)


def overdrawn(image):
    """(once-drawn value, number of interior pixels brighter than it).

    Brighter than the once-drawn value can only mean the pixel was composited
    more than once, which is what "double drawn" looks like.
    """
    interior = erode(image > 30)
    assert interior.any(), "nothing was drawn, so the measurement proves nothing"
    once = int(np.bincount(image[interior]).argmax())
    return once, int((image[interior] > once + TOLERANCE).sum())


# A zigzag of corners from about 90 degrees down to about 20, every segment
# comfortably longer than the line is thick.
ZIGZAG = [
    (-140, -60),
    (-60, 60),
    (10, -60),
    (60, 55),
    (95, -55),
    (120, 50),
    (137, -50),
]


@pytest.mark.parametrize("alpha_mode", ALPHA_MODES)
def test_acute_corners_are_not_drawn_twice(alpha_mode):
    """The reported defect: bright wedges at the corners of an acute zigzag.

    Every node sits at its own z, so this is the 3D case and the depth test is
    live. Before the corner was divided this painted hundreds of pixels twice.
    """
    image = render_polyline(ZIGZAG, alpha_mode=alpha_mode)
    once, n = overdrawn(image)
    assert n == 0, f"{n} pixels of the zigzag are drawn more than once ({once})"


@pytest.mark.parametrize("alpha_mode", ALPHA_MODES)
def test_acute_corners_are_not_drawn_twice_when_dashed(alpha_mode):
    """Dashing breaks joins from about 90 degrees, so it sees far more of them."""
    positions = np.array(
        [[x, y, float(i)] for i, (x, y) in enumerate(ZIGZAG)], np.float32
    )
    target = gfx.Texture(
        dim=2, size=(400, 400, 1), format=wgpu.TextureFormat.rgba8unorm
    )
    renderer = gfx.WgpuRenderer(target)
    renderer.ppaa = "none"
    scene = gfx.Scene()
    scene.add(gfx.Background.from_color("#000"))
    scene.add(
        gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=THICK,
                color=(1, 1, 1, ALPHA),
                aa=True,
                dash_pattern=(4, 2, 3, 2, 2, 2, 1, 2),
                **({} if alpha_mode is None else {"alpha_mode": alpha_mode}),
            ),
        )
    )
    camera = gfx.OrthographicCamera(400, 400)
    camera.local.position = (
        float(positions[:, 0].mean()),
        float(positions[:, 1].mean()),
        0,
    )
    renderer.render(scene, camera)
    _, n = overdrawn(renderer.snapshot()[..., 0].astype(int))
    assert n == 0, f"{n} pixels of the dashed zigzag are drawn more than once"


@pytest.mark.xfail(strict=True, reason="not reachable from one node and its neighbours")
def test_a_segment_shorter_than_the_line_is_thick():
    """The one local case that is still drawn twice, recorded so it is not lost.

    A corner is divided between the two segments that meet at it, and a face
    learns about the corner at each of its own two ends. That is enough as long
    as a segment is long enough to keep its neighbours apart. When it is not --
    here a five-unit segment on a line twenty-two thick -- the cap at one end of
    it reaches past the node at the other end and lands on the segment *after*
    that one, which is two steps away along the path and which no face involved
    can see. Reaching it needs nodes i-2 and i+2, i.e. a wider stencil than the
    shader has today.

    This is the geometry straight out of examples/feature_demo/line_basic.py.
    """
    image = render_polyline(
        [(-100, 5), (0, 5), (0, 0), (200, -5)], thickness=THICK, size=400
    )
    once, n = overdrawn(image)
    assert n == 0, f"{n} pixels drawn more than once ({once})"
