"""Render bundles must never change what is drawn.

Every test renders the same sequence of frames twice, once with render
bundles and once drawing directly, and requires the images to be identical
on every frame. Each frame applies a change that a stale bundle would get
wrong.
"""

import itertools

import numpy as np
import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def _tile(x, y, value, size=8):
    # A pattern (not a flat color), so that interpolation changes are visible
    data = (np.arange(size * size).reshape(size, size) * 7 + value) % 256
    data = data.astype(np.uint8)
    ob = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(data, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255)),
    )
    ob.local.position = (x, y, 0)
    return ob


def _build():
    scene = gfx.Scene()
    tiles = [_tile(10 * (i % 4), 10 * (i // 4), 40 * i) for i in range(6)]
    scene.add(*tiles)
    mesh = gfx.Mesh(gfx.plane_geometry(6, 6), gfx.MeshBasicMaterial(color="#0f0"))
    mesh.local.position = (25, 15, 1)
    scene.add(mesh)
    camera = gfx.OrthographicCamera(50, 30)
    camera.local.position = (20, 10, 10)
    return scene, tiles, mesh, camera


def _changes(scene, tiles, mesh, camera):
    """Each yields after applying one change; frame 0 is the unchanged scene."""
    yield
    yield  # same again: the second identical frame records the bundle
    yield  # bundle is replayed
    camera.local.x += 3  # camera moves
    yield
    tiles[0].geometry.grid.set_data(np.full((8, 8), 255, np.uint8))  # new data
    yield
    mesh.material.color = "#f00"  # uniform change, same pipeline
    yield
    tiles[1].visible = False  # hide
    yield
    yield
    tiles[1].visible = True  # show again
    yield
    yield
    scene.add(_tile(0, 20, 200))  # add an object
    yield
    yield
    mesh.material = gfx.MeshBasicMaterial(color="#00f")  # swap material
    yield
    yield
    tiles[2].geometry.grid = gfx.Texture(np.full((8, 8), 90, np.uint8), dim=2)
    yield  # new texture object -> new bind group
    yield
    tiles[3].material.clim = (0, 100)  # uniform change
    yield
    yield
    scene.remove(tiles[4])  # remove an object
    yield
    yield


def _render_sequence(use_bundles):
    scene, tiles, mesh, camera = _build()
    texture = gfx.Texture(dim=2, size=(100, 60, 1), format="rgba8unorm-srgb")
    renderer = gfx.renderers.WgpuRenderer(texture)
    renderer._use_render_bundles = use_bundles
    overview = gfx.OrthographicCamera(60, 40)
    overview.local.position = (20, 10, 10)
    images = []
    for _ in _changes(scene, tiles, mesh, camera):
        # Two render calls per frame, like a main view plus an overview
        renderer.render(scene, camera, rect=(0, 0, 100, 60), flush=False)
        renderer.render(scene, overview, rect=(0, 0, 30, 20), flush=False)
        renderer.flush()
        images.append(renderer.snapshot().copy())
    return images, renderer


def test_render_bundles_draw_the_same_as_direct_draws():
    direct, _ = _render_sequence(False)
    bundled, renderer = _render_sequence(True)
    assert len(direct) == len(bundled)
    for i, (a, b) in enumerate(zip(direct, bundled, strict=True)):
        assert np.array_equal(a, b), f"frame {i} differs"
    # Bundles were actually used
    assert any(entry[2] is not None for entry in renderer._render_bundles.values())


def test_render_bundles_skip_passes_with_a_stencil_reference():
    # A bundle cannot set the stencil reference, so such a pass must not be
    # bundled (and must still render correctly).
    def run(use_bundles):
        scene = gfx.Scene()
        state = wgpu.StencilFaceState(compare="always", pass_op="replace")
        writer = gfx.Mesh(
            gfx.plane_geometry(1, 2),
            gfx.MeshBasicMaterial(
                color="#000",
                depth_test=False,
                stencil_value=1,
                stencil_front_state=state,
                stencil_back_state=state,
            ),
        )
        writer.local.x = -0.5
        state = wgpu.StencilFaceState(compare="equal")
        reader = gfx.Mesh(
            gfx.plane_geometry(2, 2),
            gfx.MeshBasicMaterial(
                color="#fff",
                depth_test=False,
                stencil_value=1,
                stencil_front_state=state,
                stencil_back_state=state,
            ),
        )
        reader.render_order = 1
        scene.add(writer, reader)
        texture = gfx.Texture(dim=2, size=(64, 48, 1), format="rgba8unorm-srgb")
        renderer = gfx.renderers.WgpuRenderer(texture, enable_stencil=True)
        renderer._use_render_bundles = use_bundles
        renderer.set_clear_value(stencil=0)
        images = []
        for _ in range(4):
            renderer.render(scene, gfx.NDCCamera())
            images.append(renderer.snapshot().copy())
        return images, renderer

    direct, _ = run(False)
    bundled, renderer = run(True)
    for a, b in zip(direct, bundled, strict=True):
        assert np.array_equal(a, b)
    assert all(entry[2] is None for entry in renderer._render_bundles.values())
    h, w = bundled[-1].shape[:2]
    assert bundled[-1][h // 2, w // 4, 0] > 200  # reader drawn where marked
    assert bundled[-1][h // 2, 3 * w // 4, 0] < 50


def _pipeline_changes(scene, tiles, mesh, camera):
    """Visible changes that need new pipelines, shaders or bind groups. Each
    change is followed by an unchanged frame, so the cached paths are used."""
    yield
    yield
    tiles[0].material.interpolation = "linear"  # new shader
    yield
    yield
    mesh.material.wireframe = True  # new shader and topology
    yield
    yield
    tiles[4].material.map = gfx.cm.viridis  # adds a colormap: new shader
    yield
    yield
    tiles[4].material.map = gfx.cm.plasma  # other colormap: new bind group
    yield
    yield
    tiles[1].geometry.grid = gfx.Texture(np.full((8, 8), 30, np.uint8), dim=2)
    yield
    yield
    tiles[2].material = gfx.ImageBasicMaterial(clim=(0, 50))  # material swap
    yield
    yield
    tiles[3].render_order = -1  # draw order
    yield
    yield
    mesh.material.wireframe = False
    tiles[0].material.interpolation = "nearest"  # back again
    yield
    yield


def _render_frame(renderer, scene, camera, overview):
    renderer.render(scene, camera, rect=(0, 0, 100, 60), flush=False)
    renderer.render(scene, overview, rect=(0, 0, 30, 20), flush=False)
    renderer.flush()
    return renderer.snapshot().copy()


def test_cached_frames_match_a_fresh_renderer():
    # A long-lived renderer reuses per-frame results (collected pipelines,
    # pass lists, bundles) while nothing changed. After every change it must
    # draw exactly what an uncached render of the same state draws. The
    # reference for frame i is a brand-new scene, with the first i changes
    # replayed, rendered by a brand-new renderer: no object, pipeline group or
    # cache is shared with the long-lived run.
    def new_renderer():
        texture = gfx.Texture(dim=2, size=(100, 60, 1), format="rgba8unorm-srgb")
        return gfx.renderers.WgpuRenderer(texture)

    def new_overview():
        # Looks from behind, so its depth order (and thus its draw list)
        # differs from the main view: each view has its own cached state.
        overview = gfx.OrthographicCamera(60, 40)
        overview.local.position = (20, 10, -10)
        overview.look_at((20, 10, 0))
        return overview

    # Long-lived run
    scene, tiles, mesh, camera = _build()
    overview = new_overview()
    renderer = new_renderer()
    cached = [
        _render_frame(renderer, scene, camera, overview)
        for _ in _pipeline_changes(scene, tiles, mesh, camera)
    ]

    # Reference: rebuild and replay for every frame
    for i, image in enumerate(cached):
        scene, tiles, mesh, camera = _build()
        changes = _pipeline_changes(scene, tiles, mesh, camera)
        for _ in range(i + 1):
            next(changes)
        reference = _render_frame(new_renderer(), scene, camera, new_overview())
        assert np.array_equal(image, reference), f"frame {i} differs"

    # The changes must be visible, or this test would not detect anything
    n_changed = sum(not np.array_equal(a, b) for a, b in itertools.pairwise(cached))
    assert n_changed >= 7
