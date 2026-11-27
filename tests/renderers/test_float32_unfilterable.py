"""Rendering without the 'float32-filterable' feature.

Some GPUs (e.g. the PowerVR of the Pixel 10, some Mali) cannot filter float32
textures. Pygfx then binds them as unfilterable, and interpolates images,
volumes and colormaps in the shader: the result must match what the sampler
gives on a device that has the feature. Texture maps of meshes fall back to
'nearest'. The device is global to a process, so each case renders in a
subprocess of its own.
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from ..testutils import can_use_wgpu_lib


if not can_use_wgpu_lib:
    pytest.skip("Skipping tests that need the wgpu lib", allow_module_level=True)


SCENES = [
    "image_linear",
    "image_nearest",
    "volume_linear",
    "colormap_1d",
    "colormap_2d",
    "colormap_3d",
]


def render_scenes(filterable):
    """Render each scene, return {name: rgba array}. Run in a fresh process."""
    import pygfx as gfx

    if not filterable:
        gfx.renderers.wgpu.enable_wgpu_features("!float32-filterable")

    rng = np.random.default_rng(0)
    images = {}

    def render(scene, camera):
        target = gfx.Texture(dim=2, size=(64, 64, 1), format="rgba8unorm")
        renderer = gfx.WgpuRenderer(target, pixel_scale=1, ppaa="none")
        renderer.render(scene, camera)
        return renderer.snapshot()

    for interpolation in ("linear", "nearest"):
        data = rng.random((8, 8)).astype(np.float32)
        image = gfx.Image(
            gfx.Geometry(grid=gfx.Texture(data, dim=2)),
            gfx.ImageBasicMaterial(
                clim=(0, 1), interpolation=interpolation, map=gfx.cm.viridis
            ),
        )
        camera = gfx.OrthographicCamera(8, 8)
        camera.local.position = (3.5, 3.5, 10)
        images[f"image_{interpolation}"] = render(gfx.Scene().add(image), camera)

    data = rng.random((6, 6, 6)).astype(np.float32)
    volume = gfx.Volume(
        gfx.Geometry(grid=gfx.Texture(data, dim=3)),
        gfx.VolumeMipMaterial(clim=(0, 1), interpolation="linear"),
    )
    camera = gfx.OrthographicCamera(8, 8)
    camera.local.position = (2.5, 2.5, 20)
    images["volume_linear"] = render(gfx.Scene().add(volume), camera)

    # Colormaps of meshes: float32, linear, with each wrap mode. (A 2D map
    # with rgb channels is a texture map, not a colormap: see below.)
    geometry = gfx.plane_geometry(2, 2, 8, 8)
    texcoords = geometry.texcoords.data * 2.5 - 0.75  # beyond 0..1: wraps
    for dim in (1, 2, 3):
        scene = gfx.Scene()
        for i, wrap in enumerate(("clamp", "repeat", "mirror")):
            if dim == 1:
                cmap = rng.random((5, 4)).astype(np.float32)
                cmap[..., 3] = 1
                tc = texcoords[:, 0].copy()
            elif dim == 2:
                cmap = rng.random((4, 5)).astype(np.float32)
                tc = texcoords.copy()
            else:
                cmap = rng.random((3, 4, 5)).astype(np.float32)
                tc = np.column_stack([texcoords, texcoords[:, 0] * 0.6 + 0.3])
                tc = tc.astype(np.float32)
            mesh = gfx.Mesh(
                gfx.Geometry(
                    positions=geometry.positions,
                    indices=geometry.indices,
                    texcoords=tc,
                ),
                gfx.MeshBasicMaterial(
                    map=gfx.TextureMap(
                        gfx.Texture(cmap, dim=dim), filter="linear", wrap=wrap
                    ),
                ),
            )
            mesh.local.x = 2.2 * i
            scene.add(mesh)
        camera = gfx.OrthographicCamera(7, 7)
        camera.local.position = (2.2, 0, 10)
        images[f"colormap_{dim}d"] = render(scene, camera)

    device = gfx.renderers.wgpu.get_shared().device
    return images, "float32-filterable" in device.features


def _run(filterable):
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "images.npz")
        code = f"""
import numpy as np
from tests.renderers.test_float32_unfilterable import render_scenes
images, has_feature = render_scenes({filterable!r})
np.savez({out!r}, has_feature=has_feature, **images)
"""
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with np.load(out) as f:
            return {k: f[k] for k in f.files}, result.stderr


def test_float32_without_filtering_matches():
    with_feature, _ = _run(True)
    without, stderr = _run(False)
    if not with_feature.pop("has_feature"):
        pytest.skip("The adapter has no float32-filterable to compare with")
    assert not without.pop("has_feature")
    assert "samples with 'nearest'" not in stderr  # all interpolated in the shader
    for name in SCENES:
        a = with_feature[name].astype(int)
        b = without[name].astype(int)
        assert a[..., :3].max() > 0, f"{name} rendered nothing"
        diff = np.abs(a - b).max(axis=-1)
        # The sampler computes its weights at lower precision (8 bits of
        # sub-texel position on many GPUs) than the shader does; the maximum
        # projection of a volume can pick another sample for a pixel or so.
        assert (diff > 1).mean() < 0.01, f"{name}: {(diff > 1).mean():.1%} differ"
        assert diff.max() <= 8, f"{name}: max diff {diff.max()}"


def test_float32_texture_map_without_filtering_falls_back_to_nearest():
    code = """
import numpy as np
import pygfx as gfx
gfx.renderers.wgpu.enable_wgpu_features("!float32-filterable")
data = np.random.default_rng(0).random((4, 4, 4)).astype(np.float32)
mesh = gfx.Mesh(
    gfx.plane_geometry(2, 2),
    gfx.MeshBasicMaterial(map=gfx.TextureMap(gfx.Texture(data, dim=2), filter="linear")),
)
target = gfx.Texture(dim=2, size=(32, 32, 1), format="rgba8unorm")
renderer = gfx.WgpuRenderer(target, pixel_scale=1, ppaa="none")
renderer.render(gfx.Scene().add(mesh), gfx.OrthographicCamera(2, 2))
im = renderer.snapshot()
# nearest: the 4x4 texels are blocks of 8x8 pixels
assert im[..., :3].max() > 0
assert len(np.unique(im[..., :3].reshape(-1, 3), axis=0)) == 16, "not nearest"
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "samples with 'nearest'" in result.stderr
