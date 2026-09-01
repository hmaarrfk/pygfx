"""Round 3: bisect the real pygfx pipeline.

The synthetic probes all agree across vendors, yet the real renders differ on
~1% of values. So bisect actual pygfx scenes from the simplest possible content
upward, and hash each. The simplest scene that differs is the answer.
"""
import hashlib, os, sys
os.environ["WGPU_FORCE_OFFSCREEN"] = "true"
os.environ["PYGFX_DEFAULT_PPAA"] = "none"

import numpy as np
from rendercanvas.offscreen import RenderCanvas
import pygfx as gfx

W = H = 256


def render(build):
    canvas = RenderCanvas(size=(W, H))
    renderer = gfx.WgpuRenderer(canvas, bitexact_srgb=BITEXACT)
    scene = gfx.Scene()
    camera = build(scene)
    canvas.request_draw(lambda: renderer.render(scene, camera))
    return np.asarray(renderer.target.draw())


def bg_solid(scene):
    scene.add(gfx.Background.from_color("#808080"))
    return gfx.OrthographicCamera(2, 2)


def bg_gradient(scene):
    scene.add(gfx.Background.from_color("#fff", "#000"))
    return gfx.OrthographicCamera(2, 2)


def mesh_solid(scene):
    scene.add(gfx.Mesh(gfx.plane_geometry(1, 1),
                       gfx.MeshBasicMaterial(color="#ff0000")))
    return gfx.OrthographicCamera(2, 2)


def mesh_blend(scene):
    scene.add(gfx.Background.from_color("#fff", "#000"))
    scene.add(gfx.Mesh(gfx.plane_geometry(1, 1),
                       gfx.MeshBasicMaterial(color="#ff0000", alpha_mode="blend",
                                             opacity=0.7)))
    return gfx.OrthographicCamera(2, 2)


def line_aa(scene):
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pos = np.stack([np.sin(t), np.cos(t), np.zeros_like(t)], 1).astype("f4")
    scene.add(gfx.Line(gfx.Geometry(positions=pos),
                       gfx.LineMaterial(thickness=14, color="red", aa=True)))
    return gfx.OrthographicCamera(3, 3)


def line_blend_aa(scene):
    scene.add(gfx.Background.from_color("#fff", "#000"))
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pos = np.stack([np.sin(t), np.cos(t), np.zeros_like(t)], 1).astype("f4")
    scene.add(gfx.Line(gfx.Geometry(positions=pos),
                       gfx.LineMaterial(thickness=14, color="red", aa=True,
                                        alpha_mode="blend", opacity=0.7)))
    return gfx.OrthographicCamera(3, 3)


def image_cmap(scene):
    im = np.repeat(np.linspace(0, 1, 100).reshape(1, -1), 24, 0).astype(np.float32)
    cd = np.zeros((256, 3), np.float32)
    cd[0:128, 0] = np.linspace(1, 0, 128); cd[0:128, 1] = np.linspace(0, 1, 128)
    cd[128:, 1] = np.linspace(1, 0, 128);  cd[128:, 2] = np.linspace(0, 1, 128)
    scene.add(gfx.Image(gfx.Geometry(grid=gfx.Texture(im, dim=2)),
                        gfx.ImageBasicMaterial(clim=(0, 1),
                            map=gfx.TextureMap(gfx.Texture(cd, dim=1),
                                               filter="nearest", wrap="clamp"))))
    cam = gfx.OrthographicCamera()
    cam.show_rect(0, 100, 0, 24)
    return cam


SCENES = [bg_solid, bg_gradient, mesh_solid, mesh_blend, line_aa, line_blend_aa, image_cmap]


def main():
    from examples.tests.testutils import adapter
    print("bitexact_srgb:", BITEXACT)
    print("adapter:", adapter.info["device"], "|", adapter.info["description"], flush=True)
    out = {}
    for fn in SCENES:
        img = render(fn)
        out[fn.__name__] = img
        print(f"  {fn.__name__:14s} sha={hashlib.sha256(np.ascontiguousarray(img)).hexdigest()[:16]}",
              flush=True)
    if len(sys.argv) > 1:
        np.savez_compressed(sys.argv[1], **out)
        print("saved", sys.argv[1])

main()
