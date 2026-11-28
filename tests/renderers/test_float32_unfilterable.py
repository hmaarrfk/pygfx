"""Rendering without the 'float32-filterable' feature.

Some GPUs (e.g. some mobile GPUs) cannot filter float32 textures. Pygfx then
binds them as unfilterable, and interpolates images, volumes, colormaps and
mipmaps in the shader: the result must match what the sampler gives on a
device that has the feature. Texture maps of meshes fall back to 'nearest'.
The device is global to a process, so each case renders in a subprocess of its
own.
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


# The subprocesses import this module as tests.renderers.test_float32_unfilterable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _in_subprocess(code):
    """Run code in a fresh Python; return its stderr."""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stderr


def _call_in_subprocess(func, filterable):
    """Call func(filterable) of this module in a fresh Python.

    func returns ({name: array}, has_feature). Returns that and the stderr.
    """
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "out.npz")
        stderr = _in_subprocess(f"""
import numpy as np
from tests.renderers.test_float32_unfilterable import {func}
arrays, has_feature = {func}({filterable!r})
np.savez({out!r}, has_feature=has_feature, **arrays)
""")
        with np.load(out) as f:
            arrays = {k: f[k] for k in f.files}
    return arrays, bool(arrays.pop("has_feature")), stderr


# %% Scenes: each one is rendered with and without the feature


def _image(gfx, data, **material):
    image = gfx.Image(
        gfx.Geometry(grid=gfx.Texture(data, dim=2)),
        gfx.ImageBasicMaterial(clim=(0, 1), **material),
    )
    camera = gfx.OrthographicCamera(8, 8)
    camera.local.position = (3.5, 3.5, 10)
    return gfx.Scene().add(image), camera


def _volume(gfx, data, material):
    volume = gfx.Volume(gfx.Geometry(grid=gfx.Texture(data, dim=3)), material)
    camera = gfx.OrthographicCamera(9, 9)
    camera.local.position = (3, 2.5, 20)
    return gfx.Scene().add(volume), camera


def _colormapped_meshes(gfx, dim, maps, repeats=2.5):
    """Meshes side by side, with a float32 colormap each: [(data, kwargs)].

    The texcoords go beyond 0..1, so that the wrap modes show. With many
    repeats the colormap is minified.
    """
    geometry = gfx.plane_geometry(2, 2, 8, 8)
    texcoords = geometry.texcoords.data * repeats - (repeats - 1) / 2
    if dim == 1:
        texcoords = texcoords[:, 0]
    elif dim == 3:  # z varies along y, so that its wrap differs from x's
        texcoords = np.column_stack([texcoords, texcoords[:, 1] * 0.6 + 0.3])
    scene = gfx.Scene()
    for i, (data, kwargs) in enumerate(maps):
        mesh = gfx.Mesh(
            gfx.Geometry(
                positions=geometry.positions,
                indices=geometry.indices,
                texcoords=np.ascontiguousarray(texcoords, np.float32),
            ),
            gfx.MeshBasicMaterial(
                map=gfx.TextureMap(gfx.Texture(data, dim=dim), **kwargs)
            ),
        )
        mesh.local.x = 2.2 * i
        scene.add(mesh)
    camera = gfx.OrthographicCamera(2.2 * len(maps) + 0.2, 7)
    camera.local.position = (1.1 * (len(maps) - 1), 0, 10)
    return scene, camera


WRAPS = ("clamp", "repeat", "mirror")
MIXED_WRAPS = [("clamp", "repeat"), ("mirror", "clamp"), ("repeat", "mirror")]


def _cmap(rng, dim, nchannels=4):
    shape = {1: (5,), 2: (4, 5), 3: (3, 4, 5)}[dim]
    return rng.random((*shape, nchannels)).astype(np.float32)


def scene_image_linear(gfx, rng):
    data = rng.random((8, 8)).astype(np.float32)
    return _image(gfx, data, interpolation="linear", map=gfx.cm.viridis)


def scene_image_nearest(gfx, rng):
    data = rng.random((8, 8)).astype(np.float32)
    return _image(gfx, data, interpolation="nearest", map=gfx.cm.viridis)


def scene_image_rgb8(gfx, rng):  # pygfx stores 3 channels as 4: rgba8unorm
    data = (rng.random((8, 8, 3)) * 255).astype(np.uint8)
    return _image(gfx, data, interpolation="linear")


def scene_image_rgb32(gfx, rng):
    data = rng.random((8, 8, 3)).astype(np.float32)
    return _image(gfx, data, interpolation="linear")


def scene_image_npot_cmap_linear(gfx, rng):  # float32 image and colormap
    cmap = gfx.TextureMap(
        gfx.Texture(rng.random((6, 3)).astype(np.float32), dim=1), filter="linear"
    )
    data = rng.random((7, 5)).astype(np.float32)
    return _image(gfx, data, interpolation="nearest", map=cmap)


def scene_image_yuv444p(gfx, rng):  # YUV is converted from what a sampler gives
    data = rng.random((3, 8, 8)).astype(np.float32)
    texture = gfx.Texture(data, dim=2, size=(8, 8, 3), colorspace="yuv444p")
    image = gfx.Image(
        gfx.Geometry(grid=texture), gfx.ImageBasicMaterial(interpolation="nearest")
    )
    camera = gfx.OrthographicCamera(8, 8)
    camera.local.position = (3.5, 3.5, 10)
    return gfx.Scene().add(image), camera


def scene_image_yuv420p(gfx, rng):  # three grids
    def plane(n):
        return gfx.Texture(
            rng.random((n, n)).astype(np.float32), dim=2, colorspace="yuv420p"
        )

    geometry = gfx.Geometry(grid=plane(8), grid_u=plane(4), grid_v=plane(4))
    image = gfx.Image(geometry, gfx.ImageBasicMaterial(interpolation="nearest"))
    camera = gfx.OrthographicCamera(8, 8)
    camera.local.position = (3.5, 3.5, 10)
    return gfx.Scene().add(image), camera


def scene_volume_mip_linear(gfx, rng):
    data = rng.random((5, 6, 7)).astype(np.float32)
    material = gfx.VolumeMipMaterial(clim=(0, 1), interpolation="linear")
    return _volume(gfx, data, material)


def scene_volume_mip_nearest(gfx, rng):
    data = rng.random((5, 6, 7)).astype(np.float32)
    material = gfx.VolumeMipMaterial(clim=(0, 1), interpolation="nearest")
    return _volume(gfx, data, material)


def scene_volume_slice_linear(gfx, rng):
    data = rng.random((5, 6, 7)).astype(np.float32)
    material = gfx.VolumeSliceMaterial(
        clim=(0, 1), interpolation="linear", plane=(0, 0, -1, 2.3)
    )
    return _volume(gfx, data, material)


def scene_volume_slice_nearest(gfx, rng):
    data = rng.random((5, 6, 7)).astype(np.float32)
    material = gfx.VolumeSliceMaterial(
        clim=(0, 1), interpolation="nearest", plane=(0, 0, -1, 2.3)
    )
    return _volume(gfx, data, material)


def scene_colormap_1d(gfx, rng):  # rgb: stored as rgba
    maps = [(_cmap(rng, 1, 3), dict(filter="linear", wrap=w)) for w in WRAPS]
    return _colormapped_meshes(gfx, 1, maps)


def scene_colormap_1d_nearest(gfx, rng):
    maps = [(_cmap(rng, 1), dict(filter="nearest", wrap=w)) for w in WRAPS]
    return _colormapped_meshes(gfx, 1, maps)


def scene_colormap_1d_one_texel(gfx, rng):
    data = rng.random((1, 3)).astype(np.float32)
    maps = [(data, dict(filter="linear", wrap=w)) for w in WRAPS]
    return _colormapped_meshes(gfx, 1, maps)


def scene_colormap_2d(gfx, rng):  # 1 channel: rgb 2D maps are texture maps
    maps = [(_cmap(rng, 2, 1), dict(filter="linear", wrap=w)) for w in WRAPS]
    return _colormapped_meshes(gfx, 2, maps)


def scene_colormap_3d(gfx, rng):
    maps = [(_cmap(rng, 3, 1), dict(filter="linear", wrap=w)) for w in WRAPS]
    return _colormapped_meshes(gfx, 3, maps)


def scene_colormap_2d_mixed_wrap(gfx, rng):  # wrap_s != wrap_t
    maps = [
        (_cmap(rng, 2, 1), dict(filter="linear", wrap_s=s, wrap_t=t))
        for s, t in MIXED_WRAPS
    ]
    return _colormapped_meshes(gfx, 2, maps)


def scene_colormap_3d_mixed_wrap(gfx, rng):  # and z follows wrap_t
    maps = [
        (_cmap(rng, 3, 1), dict(filter="linear", wrap_s=s, wrap_t=t))
        for s, t in MIXED_WRAPS
    ]
    return _colormapped_meshes(gfx, 3, maps)


def _minified(gfx, rng, dim, nchannels):
    """mag_filter != min_filter, minified: the sampler uses the min filter."""
    maps = [
        (_cmap(rng, dim, nchannels), dict(mag_filter=mag, min_filter=mn, wrap=w))
        for mag, mn in [("linear", "nearest"), ("nearest", "linear")]
        for w in ("repeat", "mirror")
    ]
    return _colormapped_meshes(gfx, dim, maps, repeats=40.3)


def scene_colormap_1d_minified(gfx, rng):
    return _minified(gfx, rng, 1, 4)


def scene_colormap_2d_minified(gfx, rng):
    return _minified(gfx, rng, 2, 1)


SCENES = {
    name.removeprefix("scene_"): func
    for name, func in list(globals().items())
    if name.startswith("scene_")
}


def render_scenes(filterable):
    """Render each scene, return ({name: rgba array}, has_feature)."""
    import pygfx as gfx

    if not filterable:
        gfx.renderers.wgpu.enable_wgpu_features("!float32-filterable")
    images = {}
    for i, (name, build) in enumerate(SCENES.items()):
        scene, camera = build(gfx, np.random.default_rng(i))
        target = gfx.Texture(dim=2, size=(64, 64, 1), format="rgba8unorm")
        renderer = gfx.WgpuRenderer(target, pixel_scale=1, ppaa="none")
        renderer.render(scene, camera)
        images[name] = renderer.snapshot()
    device = gfx.renderers.wgpu.get_shared().device
    return images, "float32-filterable" in device.features


@pytest.fixture(scope="module")
def renders():
    with_feature, has_feature, stderr_with = _call_in_subprocess("render_scenes", True)
    if not has_feature:
        pytest.skip("The adapter has no float32-filterable to compare with")
    without, has_feature, stderr_without = _call_in_subprocess("render_scenes", False)
    assert not has_feature
    return with_feature, without, stderr_with + stderr_without


def test_without_filtering_no_warnings(renders):
    *_, stderr = renders
    assert "samples with 'nearest'" not in stderr  # all interpolated in the shader
    assert "unknown to pygfx" not in stderr


@pytest.mark.parametrize("name", list(SCENES))
def test_without_filtering_matches(renders, name):
    with_feature, without, _ = renders
    a = with_feature[name].astype(int)
    b = without[name].astype(int)
    assert a[..., :3].max() > 0, f"{name} rendered nothing"
    diff = np.abs(a - b).max(axis=-1)
    # The sampler computes its weights at lower precision (8 bits of sub-texel
    # position on many GPUs) than the shader does; the maximum projection of a
    # volume can pick another sample for a pixel or so.
    assert (diff > 1).mean() < 0.01, f"{(diff > 1).mean():.1%} differ"
    assert diff.max() <= 8, f"max diff {diff.max()}"


# %% Fallbacks to 'nearest'


def test_float32_texture_map_without_filtering_falls_back_to_nearest():
    stderr = _in_subprocess("""
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
""")
    assert "s_map: the device cannot filter rgba32float" in stderr
    assert "samples with 'nearest'" in stderr


def test_float32_yuv_linear_without_filtering_falls_back_to_nearest():
    stderr = _in_subprocess("""
import numpy as np
import pygfx as gfx
gfx.renderers.wgpu.enable_wgpu_features("!float32-filterable")
data = np.random.default_rng(0).random((3, 8, 8)).astype(np.float32)
texture = gfx.Texture(data, dim=2, size=(8, 8, 3), colorspace="yuv444p")
image = gfx.Image(
    gfx.Geometry(grid=texture), gfx.ImageBasicMaterial(interpolation="linear")
)
target = gfx.Texture(dim=2, size=(32, 32, 1), format="rgba8unorm")
renderer = gfx.WgpuRenderer(target, pixel_scale=1, ppaa="none")
camera = gfx.OrthographicCamera(8, 8)
camera.local.position = (3.5, 3.5, 10)
renderer.render(gfx.Scene().add(image), camera)
assert renderer.snapshot()[..., :3].max() > 0
""")
    assert "s_img: the device cannot filter r32float" in stderr


# %% Mipmaps


MIP_SHAPES = [(4, 4), (5, 5), (3, 7), (6, 10), (1, 8), (7, 2)]


def render_mipmaps(filterable):
    """Level 1 of mipmapped float32 textures with 2 layers, and the data."""
    import wgpu
    import pygfx as gfx
    from pygfx.renderers.wgpu.engine.update import ensure_wgpu_object, update_resource

    if not filterable:
        gfx.renderers.wgpu.enable_wgpu_features("!float32-filterable")
    device = gfx.renderers.wgpu.get_shared().device
    rng = np.random.default_rng(2)
    out = {}
    for h, w in MIP_SHAPES:
        data = rng.random((2, h, w)).astype(np.float32)
        tex = gfx.Texture(data, dim=2, size=(w, h, 2), generate_mipmaps=True)
        tex._wgpu_usage |= wgpu.TextureUsage.TEXTURE_BINDING
        tex._wgpu_usage |= wgpu.TextureUsage.COPY_SRC
        ensure_wgpu_object(tex)
        update_resource(tex)
        w1, h1 = max(1, w // 2), max(1, h // 2)
        for layer in range(2):
            raw = device.queue.read_texture(
                {"texture": tex._wgpu_object, "mip_level": 1, "origin": (0, 0, layer)},
                {"offset": 0, "bytes_per_row": 256},
                (w1, h1, 1),
            )
            level = np.frombuffer(raw, np.uint8).reshape(h1, 256)[:, : w1 * 4]
            out[f"{h}x{w}_{layer}"] = level.copy().view(np.float32)
            out[f"{h}x{w}_{layer}_src"] = data[layer]
    return out, "float32-filterable" in device.features


def _bilinear_downsample(src, h1, w1):
    """What a linear clamp-to-edge sampler gives at the texel centers of level 1."""
    h, w = src.shape
    y = (np.arange(h1) + 0.5) * h / h1 - 0.5
    x = (np.arange(w1) + 0.5) * w / w1 - 0.5
    y0, x0 = np.floor(y).astype(int), np.floor(x).astype(int)
    ty, tx = (y - y0)[:, None], (x - x0)[None, :]

    def texel(yy, xx):
        return src[np.clip(yy, 0, h - 1)[:, None], np.clip(xx, 0, w - 1)[None, :]]

    top = (1 - tx) * texel(y0, x0) + tx * texel(y0, x0 + 1)
    bottom = (1 - tx) * texel(y0 + 1, x0) + tx * texel(y0 + 1, x0 + 1)
    return (1 - ty) * top + ty * bottom


@pytest.mark.parametrize("filterable", [True, False])
def test_float32_mipmaps_match_the_linear_sampler(filterable):
    out, has_feature, _ = _call_in_subprocess("render_mipmaps", filterable)
    if filterable and not has_feature:
        pytest.skip("The adapter has no float32-filterable")
    for key in [k for k in out if not k.endswith("_src")]:
        expected = _bilinear_downsample(out[key + "_src"], *out[key].shape)
        # 8 bits of sub-texel precision in the sampler: 1/256 of a texel step
        np.testing.assert_allclose(out[key], expected, atol=1 / 128, err_msg=key)


# %% The classification of texture formats


@pytest.fixture
def device_features(monkeypatch):
    """The features that the classification sees on the device (a set to fill)."""
    from types import SimpleNamespace

    from pygfx.renderers.wgpu.engine import shared

    features = set()
    fake = SimpleNamespace(device=SimpleNamespace(features=features))
    monkeypatch.setattr(shared, "get_shared", lambda: fake)
    return features


def test_every_texture_format_is_classified():
    import wgpu
    from pygfx.renderers.wgpu.engine import shared

    tables = [
        shared.TEXTURE_FORMATS_FILTERABLE,
        set(shared.TEXTURE_FORMATS_FILTERABLE_WITH_FEATURE),
        shared.TEXTURE_FORMATS_UNFILTERABLE,
    ]
    # Each format of wgpu is in exactly one table, and the tables have no others
    assert sorted(f for t in tables for f in t) == sorted(wgpu.TextureFormat)
    # The features named exist: in WebGPU, Dawn, or wgpu-native
    features = set(wgpu.FeatureName) | {
        "unorm16-filterable",  # Dawn
        "texture-format16bit-norm",  # wgpu-native
    }
    for needed in shared.TEXTURE_FORMATS_FILTERABLE_WITH_FEATURE.values():
        assert set(needed) <= features


def test_unknown_texture_format_is_unfilterable_with_warning(caplog):
    from pygfx.renderers.wgpu.engine.shared import texture_format_is_filterable

    assert texture_format_is_filterable("rgba8unorm")
    assert not texture_format_is_filterable("r32uint")
    assert not caplog.records
    assert not texture_format_is_filterable("r64fancyfloat")
    assert "unknown to pygfx" in caplog.text


@pytest.mark.parametrize(
    "format, needs",
    [
        ("rgba8unorm", None),
        ("rgb8unorm", None),  # pygfx's 3-channel format, stored as rgba8unorm
        ("r16float", None),
        ("r32float", "float32-filterable"),
        ("rgb32float", "float32-filterable"),
        ("r16unorm", "unorm16-filterable"),  # Dawn
        ("r16unorm", "texture-format16bit-norm"),  # wgpu-native
        ("r16snorm", "texture-format16bit-norm"),
        ("bc1-rgba-unorm", "texture-compression-bc"),
    ],
)
def test_texture_format_is_filterable(device_features, format, needs):
    from pygfx.renderers.wgpu.engine.shared import texture_format_is_filterable

    if needs:
        assert not texture_format_is_filterable(format)
        device_features.add(needs)
    assert texture_format_is_filterable(format)


def test_16bit_norm_is_not_filterable_with_tier1_only(device_features):
    # Dawn binds them as unfilterable-float with 'texture-formats-tier1' only
    from pygfx.renderers.wgpu.engine.shared import texture_sample_type

    device_features.add("texture-formats-tier1")
    assert texture_sample_type("r16unorm") == "unfilterable-float"


@pytest.mark.parametrize(
    "format, features, sample_type",
    [
        ("rgba8unorm", (), "float"),
        ("rgb8unorm", (), "float"),
        ("r32float", ("float32-filterable",), "float"),
        ("r32float", (), "unfilterable-float"),
        ("rgb32float", (), "unfilterable-float"),
        ("r8uint", (), "uint"),
        ("rgb16sint", (), "sint"),
        ("stencil8", (), "uint"),
        ("depth32float", (), "depth"),
        ("depth16unorm", (), "depth"),
    ],
)
def test_texture_sample_type(device_features, format, features, sample_type):
    from pygfx.renderers.wgpu.engine.shared import texture_sample_type

    device_features.update(features)
    assert texture_sample_type(format) == sample_type


@pytest.mark.parametrize(
    "format, interpolation, expected",
    [
        ("rgb8unorm", "linear", True),
        ("rgb8unorm", "nearest", True),
        ("rgb8unorm", "cubic", False),  # only the shader does cubic
        ("r8uint", "nearest", False),  # images bind a filtering sampler
        ("r32float", "linear", False),  # without the feature
    ],
)
def test_sampler_can_interpolate(device_features, format, interpolation, expected):
    from pygfx.renderers.wgpu.engine.shared import sampler_can_interpolate

    assert sampler_can_interpolate(format, interpolation) == expected
