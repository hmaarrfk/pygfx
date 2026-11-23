"""Render bundles must never change what is drawn.

Every test renders the same sequence of frames twice, once with render
bundles and once drawing directly, and requires the images to be identical
on every frame. Each frame applies a change that a stale bundle would get
wrong.
"""

import numpy as np
import pytest
import wgpu

import pygfx as gfx

from ..testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def _tile(x, y, value, size=8):
    data = np.full((size, size), value, np.uint8)
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
