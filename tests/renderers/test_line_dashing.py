"""
Test the cumulative distance that the line shader bakes to parametrize dashes.

The buffer is checked directly (rather than via a screenshot) because the
values have exact expected answers, which makes the assertions sharp.
"""

import numpy as np
import pytest
import pygfx as gfx
from pygfx.renderers.wgpu.shaders.lineshader import (
    LineShader,
    DASH_LEVEL_HYSTERESIS,
)


NAN = np.full((1, 3), np.nan, np.float32)


def regular_polygon(n, x=0.0, r=1.0):
    """The nodes of a regular n-gon, without repeating the first one."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack(
        [x + r * np.sin(t), r * np.cos(t), np.zeros_like(t)], axis=1
    ).astype(np.float32)


def polygon_side_length(n, r):
    return 2 * r * np.sin(np.pi / n)


def bake(positions, **material_kwargs):
    """Bake the line distance buffer for the given positions, and return it."""
    material_kwargs.setdefault("thickness_space", "model")
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(thickness=1, dash_pattern=[2, 2], **material_kwargs),
    )
    camera = gfx.OrthographicCamera(100, 100)
    shader = LineShader(line)
    shader.bake_function(line, camera, (100, 100))
    return shader.line_distance_buffer.data


def test_cumdist_of_a_plain_line():
    positions = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], np.float32)
    cumdist = bake(positions)
    assert np.allclose(cumdist, [0, 3, 7])


def test_cumdist_restarts_at_each_line_piece():
    """Each nan-separated piece starts its dash pattern from scratch.

    Without this, the dash phase of a piece is offset by the total length of
    all the pieces before it, so the dashes of the later pieces race whenever
    something changes those lengths (such as the zoom level).
    """
    positions = np.vstack(
        [
            np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0]], np.float32),
            NAN,
            np.array([[0, 0, 0], [5, 0, 0]], np.float32),
            NAN,
            NAN,  # successive nans must not confuse the piece detection
            np.array([[0, 0, 0], [0, 2, 0]], np.float32),
        ]
    ).astype(np.float32)
    cumdist = bake(positions)
    assert np.allclose(cumdist[0:3], [0, 3, 7])
    assert np.allclose(cumdist[4:6], [0, 5])
    assert np.allclose(cumdist[8:10], [0, 2])


def test_cumdist_closes_a_loop():
    """The node that closes a loop holds the cumdist of the *whole* loop.

    If it held the cumdist of the loop's first node instead, the shader would
    measure the closing segment as if it spanned the entire loop, and its
    dashes would come out that many times too dense. See gh-1103.
    """
    n, r = 5, 10.0
    side = polygon_side_length(n, r)
    cumdist = bake(regular_polygon(n, r=r), loop=True)

    # One extra element, to hold the cumdist of the closed loop
    assert len(cumdist) == n + 1
    assert np.allclose(cumdist, side * np.arange(n + 1))


def test_cumdist_closes_several_loops():
    """Every loop closes on itself, and each one starts from zero."""
    shapes = [(4, 10.0), (3, 5.0), (6, 7.0)]
    positions = np.vstack(
        [
            x
            for n, r in shapes
            for x in (regular_polygon(n, r=r), NAN)
            # the last nan is harmless: a trailing nan is not a loop
        ]
    ).astype(np.float32)
    cumdist = bake(positions, loop=True)

    i = 0
    for n, r in shapes:
        side = polygon_side_length(n, r)
        assert np.allclose(cumdist[i : i + n + 1], side * np.arange(n + 1))
        i += n + 1


def test_cumdist_honors_the_draw_range():
    """Loops are detected relative to the draw range, at the right nodes."""
    n, r = 4, 10.0
    side = polygon_side_length(n, r)
    positions = np.vstack([NAN, NAN, regular_polygon(n, r=r), NAN]).astype(np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=1, dash_pattern=[2, 2], loop=True, thickness_space="model"
        ),
    )
    line.geometry.positions.draw_range = 2, n + 1
    camera = gfx.OrthographicCamera(100, 100)
    shader = LineShader(line)
    shader.bake_function(line, camera, (100, 100))

    cumdist = shader.line_distance_buffer.data
    assert np.allclose(cumdist[2 : 2 + n + 1], side * np.arange(n + 1))


def test_cumdist_with_nonfinites_in_other_thickness_spaces():
    """Nans must not trip up the transform to world/screen space."""
    positions = np.vstack([regular_polygon(4, r=10.0), NAN]).astype(np.float32)
    for thickness_space in ["model", "world", "screen"]:
        cumdist = bake(positions, loop=True, thickness_space=thickness_space)
        assert np.all(np.isfinite(cumdist))
        assert cumdist[0] == 0
        assert cumdist[4] > cumdist[3] > 0


# ----- quantized dash scaling


def bake_quantized(positions, view_width, thickness=10.0, logical_size=(1000, 1000)):
    """Bake with dash_scaling='quantized' at a given ortho camera width."""
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=thickness,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    camera = gfx.OrthographicCamera(view_width, view_width)
    shader = LineShader(line)
    shader.bake_function(line, camera, logical_size)
    return shader.line_distance_buffer.data.copy(), shader._dash_level


def test_dash_scaling_only_applies_to_screen_space():
    """In model/world space the pattern is anchored to the object already."""
    positions = np.array([[0, 0, 0], [10, 0, 0]], np.float32)
    for thickness_space in ["model", "world"]:
        line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=1,
                dash_pattern=[2, 2],
                thickness_space=thickness_space,
                dash_scaling="quantized",
            ),
        )
        shader = LineShader(line)
        assert shader["dash_scaling"] == "continuous"
        assert shader["cumdist_space"] == thickness_space


def test_quantized_period_is_a_power_of_two_of_model_units():
    """One dash unit spans 2**level model units, whatever the zoom."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    for view_width, expected_level in [(400, 2), (200, 1), (100, 0), (50, -1)]:
        cumdist, level = bake_quantized(positions, view_width)
        assert level == expected_level
        # cumdist is in dash units, so the far node sits at length / 2**level
        assert np.allclose(cumdist[1], 100.0 / 2.0**level)


def test_quantized_dashes_split_rather_than_slide():
    """Every dash edge of a coarser level survives at the next finer level.

    This is the property that makes the dashes appear to split in two rather
    than travel along the line when the view scale changes.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    dash_size = 4.0

    def edges(view_width):
        cumdist, _ = bake_quantized(positions, view_width)
        # phase is linear in x along this straight line
        model_per_period = dash_size * 100.0 / cumdist[1]
        return np.arange(0, 100.0 + 1e-9, model_per_period)

    previous = None
    for view_width in np.linspace(320, 160, 17):  # a smooth 2x zoom in
        current = edges(view_width)
        if previous is not None:
            # every old edge must still be an edge, to within float error
            distance = np.abs(previous[:, None] - current[None, :]).min(axis=1)
            assert distance.max() < 1e-3, f"a dash edge moved by {distance.max()}"
        previous = current


def test_quantized_on_screen_size_stays_near_the_requested_one():
    """Rounding the log2 keeps the dash unit within sqrt(2) of `thickness`."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    thickness, logical = 10.0, 1000
    for view_width in np.geomspace(40, 400, 25):
        _, level = bake_quantized(positions, view_width, thickness=thickness)
        model_units_per_pixel = view_width / logical
        on_screen = 2.0**level / model_units_per_pixel
        assert (
            thickness / np.sqrt(2) - 1e-6 <= on_screen <= thickness * np.sqrt(2) + 1e-6
        )


def test_quantized_level_does_not_flicker_at_a_boundary():
    """The level snap has hysteresis, so a view parked on a boundary is stable.

    Without it, jitter of a fraction of a percent flips the level on more than
    half of all frames, and the dashes stutter between splitting and merging.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=10,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    shader = LineShader(line)
    # thickness * view_width / logical_size == 2 ** (level + 0.5) at a boundary
    boundary = 100 * 2**0.5
    rng = np.random.default_rng(0)

    levels = []
    for _ in range(200):
        view_width = boundary * (1 + rng.uniform(-0.05, 0.05))
        shader.bake_function(
            line, gfx.OrthographicCamera(view_width, view_width), (1000, 1000)
        )
        levels.append(shader._dash_level)
    assert len(set(levels)) == 1, f"level flickered between {sorted(set(levels))}"


def test_quantized_level_still_follows_a_real_zoom():
    """Hysteresis must delay a level change, not prevent it."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=10,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    shader = LineShader(line)
    levels = []
    for view_width in np.geomspace(400, 50, 60):  # a 3-octave zoom in
        shader.bake_function(
            line, gfx.OrthographicCamera(view_width, view_width), (1000, 1000)
        )
        levels.append(shader._dash_level)
    # Monotonically decreasing, one step at a time, spanning three octaves
    steps = np.diff(levels)
    assert set(np.unique(steps)) <= {0, -1}
    assert levels[0] - levels[-1] == 3


def test_dash_max_scale_places_the_octave():
    """`dash_max_scale` chooses where the factor-of-two band sits.

    The band is always exactly one octave wide -- that is what makes a level
    change split each dash rather than move it -- so the only freedom is where
    it sits relative to the size that `dash_pattern` asks for.
    """
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    thickness, logical = 10.0, 1000
    overshoot = 2**DASH_LEVEL_HYSTERESIS  # hysteresis lets it stray this much

    for dash_max_scale in [1.0, 2**0.5, 2.0]:
        line = gfx.Line(
            gfx.Geometry(positions=positions),
            gfx.LineMaterial(
                thickness=thickness,
                dash_pattern=[2, 2],
                thickness_space="screen",
                dash_scaling="quantized",
                dash_max_scale=dash_max_scale,
            ),
        )
        shader = LineShader(line)
        ratios, levels = [], []
        for view_width in np.geomspace(400, 25, 400):  # four octaves of zoom
            shader.bake_function(
                line, gfx.OrthographicCamera(view_width, view_width), (logical, logical)
            )
            ratios.append(2.0**shader._dash_level / (view_width / logical) / thickness)
            levels.append(shader._dash_level)

        # the level only ever steps down, one at a time
        assert set(np.diff(levels)) <= {0, -1}
        # and the size stays inside the octave, give or take the hysteresis
        assert min(ratios) >= dash_max_scale / 2 / overshoot - 1e-6
        assert max(ratios) <= dash_max_scale * overshoot + 1e-6
        # the band really is a factor of two wide, not something narrower
        assert max(ratios) / min(ratios) > 1.5


def test_dash_max_scale_default_matches_round():
    """The default sqrt(2) is exactly the `round(log2(...))` behaviour."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    for view_width, expected_level in [(400, 2), (200, 1), (100, 0), (50, -1)]:
        _, level = bake_quantized(positions, view_width)
        assert level == expected_level


def test_dash_max_scale_is_validated():
    material = gfx.LineMaterial(dash_pattern=[2, 2])
    assert material.dash_max_scale == 2**0.5
    for bad in [0.9, 2.1]:
        with pytest.raises(ValueError):
            material.dash_max_scale = bad


def test_quantized_rebakes_only_when_the_level_changes():
    """The level changes rarely, so most frames can skip the bake entirely."""
    positions = np.array([[0, 0, 0], [100, 0, 0]], np.float32)
    line = gfx.Line(
        gfx.Geometry(positions=positions),
        gfx.LineMaterial(
            thickness=10,
            dash_pattern=[2, 2],
            thickness_space="screen",
            dash_scaling="quantized",
        ),
    )
    shader = LineShader(line)

    def bake(view_width):
        shader.bake_function(
            line, gfx.OrthographicCamera(view_width, view_width), (1000, 1000)
        )
        return shader._cumdist_hash

    first = bake(200)
    assert bake(190) == first, "a small zoom should not change the level"
    assert bake(100) != first, "a 2x zoom should change the level"


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            func()
