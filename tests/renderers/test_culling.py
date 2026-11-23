"""Frustum culling must never change what is drawn.

A long-lived renderer with culling renders a sequence of changes; for every
frame it must match an uncached render of the same state without culling (a
new scene with the changes so far replayed, drawn by a new renderer).
"""

import numpy as np
import pytest

import pygfx as gfx
from pygfx.renderers.wgpu.engine.renderer import FlatScene

from ..testutils import can_use_wgpu_lib

if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


def _image(x, y, value, size=8, scale=1.0):
    data = (np.arange(size * size).reshape(size, size) * 5 + value) % 256
    ob = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(data.astype(np.uint8), dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 255)),
    )
    ob.local.position = (x, y, 0)
    ob.local.scale = (scale, scale, 1)
    return ob


def _build():
    scene = gfx.Scene()
    group = gfx.Group()
    tiles = []
    for i in range(36):
        tile = _image(10 * (i % 6), 10 * (i // 6), 20 * i)
        tiles.append(tile)
        (group if i % 2 else scene).add(tile)
    scene.add(group)
    # Not cullable: always drawn
    mesh = gfx.Mesh(gfx.plane_geometry(4, 4), gfx.MeshBasicMaterial(color="#0f0"))
    mesh.local.position = (25, 25, 1)
    scene.add(mesh)
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(np.full((4, 4, 4), 0.8, np.float32), dim=3)),
        gfx.VolumeSliceMaterial(
            clim=(0, 1)
        ),  # (MIP raycasting fails on Dawn, see pygfx#1348)
    )
    volume.local.position = (40, 0, -5)
    scene.add(volume)
    camera = gfx.OrthographicCamera(60, 60)
    camera.local.position = (27, 27, 50)
    return scene, group, tiles, camera


def _changes(scene, group, tiles, camera):
    yield  # whole grid in view
    camera.width = camera.height = 12  # zoom in on the middle
    yield
    camera.local.x = 5  # pan to the left edge, partially outside
    yield
    camera.local.position = (2, 2, 50)  # corner
    yield
    tiles[35].local.position = (0, 0, 0.5)  # moved into view from outside
    yield
    group.local.x = -8  # moves half the tiles, some into view
    yield
    group.local.x = 0
    tiles[14].geometry.grid = gfx.Texture(np.full((40, 40), 99, np.uint8), dim=2)
    yield  # tile 14 (at 20, 20) now reaches into view? (not yet: far)
    camera.local.position = (55, 55, 50)  # other corner
    yield
    tiles[0].geometry.grid = gfx.Texture(np.full((60, 60), 200, np.uint8), dim=2)
    yield  # tile 0 (at 0, 0) now extends up to (59.5, 59.5): into view
    tiles[0].visible = False
    yield
    tiles[0].visible = True
    yield
    camera.local.position = (40, 0, 50)  # the volume
    yield
    camera.set_view_offset(240, 240, 120, 0, 120, 120)  # a quarter of the view
    yield
    camera.clear_view_offset()
    yield


def _new_renderer(culling):
    texture = gfx.Texture(dim=2, size=(80, 80, 1), format="rgba8unorm-srgb")
    renderer = gfx.renderers.WgpuRenderer(texture)
    renderer._use_frustum_culling = culling
    return renderer


def _render(renderer, scene, camera):
    renderer.render(scene, camera)
    return renderer.snapshot().copy()


def _check_sequence(make_camera_perspective=False):
    def setup():
        scene, group, tiles, camera = _build()
        if make_camera_perspective:
            pcam = gfx.PerspectiveCamera(50, 1)
            pcam.local.position = camera.local.position
            pcam.local.z = 30
            pcam.look_at((20, 25, 0))
            camera = pcam
        return scene, group, tiles, camera

    scene, group, tiles, camera = setup()
    renderer = _new_renderer(True)
    images, n_drawn = [], []
    total = None
    for _ in _changes(scene, group, tiles, camera):
        if make_camera_perspective and isinstance(camera, gfx.PerspectiveCamera):
            camera.look_at((camera.local.x, camera.local.y, 0))
        images.append(_render(renderer, scene, camera))
        flat = FlatScene(scene, None, 10**6, camera.camera_matrix)
        n_drawn.append(len(flat._wobject_wrappers))
        total = len(FlatScene(scene, None, 10**6)._wobject_wrappers)

    for i, image in enumerate(images):
        scene, group, tiles, camera = setup()
        changes = _changes(scene, group, tiles, camera)
        for _ in range(i + 1):
            next(changes)
        if make_camera_perspective and isinstance(camera, gfx.PerspectiveCamera):
            camera.look_at((camera.local.x, camera.local.y, 0))
        reference = _render(_new_renderer(False), scene, camera)
        assert np.array_equal(image, reference), f"frame {i} differs"
    return n_drawn, total


def test_culling_draws_the_same_orthographic():
    n_drawn, total = _check_sequence()
    assert n_drawn[0] == total  # whole grid in view: nothing culled
    assert min(n_drawn) < total // 4  # zoomed in: most tiles culled


def test_culling_draws_the_same_perspective():
    n_drawn, total = _check_sequence(make_camera_perspective=True)
    assert min(n_drawn) < total


def test_culling_keeps_objects_that_touch_the_view():
    # A tile straddling the edge of the view, or only touching a corner,
    # must be kept; one just outside must be culled.
    scene = gfx.Scene()
    inside = _image(0, 0, 0)  # covers -0.5 .. 7.5
    straddle = _image(9, 0, 0)  # covers 8.5 .. 16.5, view ends at 10
    corner = _image(9.9, 9.9, 0)  # covers 9.4 .. 17.4 in x and y
    outside = _image(10.6, 0, 0)  # covers 10.1 .. 18.1
    rotated = _image(-20, 0, 0)  # rotated by 90 degrees about (-20, 0) -> outside
    rotated.local.rotation = (0, 0, 0.7071068, 0.7071068)
    scene.add(inside, straddle, corner, outside, rotated)
    camera = gfx.OrthographicCamera(20, 20)
    camera.local.position = (0, 0, 10)  # view is -10 .. 10
    flat = FlatScene(scene, None, 0, camera.camera_matrix)
    drawn = [w.wobject for w in flat._wobject_wrappers]
    assert drawn == [inside, straddle, corner]
