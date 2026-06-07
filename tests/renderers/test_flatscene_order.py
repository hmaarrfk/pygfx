"""FlatScene must present renderables (and lights/shadow casters) in the same
order the historical depth-first walk produced. This guards the incremental
index and the vectorized depth-sort against ordering regressions.

The reference here is computed independently (a from-scratch DFS + the
renderer's sort-key formula), so it validates whatever FlatScene does
internally. Pure CPU; no GPU/adapter needed.
"""
import numpy as np
import pylinalg as la
import pygfx as gfx
from pygfx.renderers.wgpu.engine.renderer import FlatScene


def _is_group(o):
    return getattr(type(o), "_subtree_is_group", False)


def _reference(scene, view_matrix):
    renderables = []

    def sort_key(o, group_ro):
        m = o._material
        rq = m.render_queue
        sign = 1 if 1500 < rq <= 2500 else -1
        if m.alpha_method == "weighted":
            df = 0
        elif view_matrix is None:
            df = -1
        else:
            rel = la.vec_transform(o.world.position, view_matrix, projection=False)
            df = float(-rel[2]) * sign
        return (rq, group_ro, o.render_order, df)

    def visit(o, group_ro):
        if o._material is not None:
            renderables.append((o, sort_key(o, group_ro)))
        child_ro = o.render_order if _is_group(o) else group_ro
        for c in o._children:
            if c._store.visible:
                visit(c, child_ro)

    if scene._store.visible:
        sgr = scene.render_order if _is_group(scene) else 0
        for c in scene._children:
            if c._store.visible:
                visit(c, sgr)
    renderables.sort(key=lambda t: t[1])  # stable
    return [o for o, _ in renderables]


def _pts(**kw):
    p = gfx.Points(
        gfx.Geometry(positions=np.random.rand(1, 3).astype("float32")),
        gfx.PointsMaterial(),
    )
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def _cameras():
    a = gfx.PerspectiveCamera(70, 1)
    a.local.position = (3, 4, 12)
    a.show_pos((0, 0, 0))
    b = gfx.OrthographicCamera(10, 10)
    b.local.position = (-2, 1, 8)
    b.show_pos((0, 0, 0))
    return {"none": None, "persp": a.view_matrix, "ortho": b.view_matrix}


def _scenes():
    rng = np.random.default_rng(1)
    scenes = {}

    s = gfx.Scene()
    s.add(gfx.AmbientLight(), gfx.DirectionalLight())
    for i in range(40):
        p = _pts(render_order=i % 3)
        p.local.position = rng.uniform(-5, 5, 3)
        s.add(p)
    s.children[10].visible = False
    scenes["flat_varied"] = s

    s = gfx.Scene()
    g1, g2, g3 = gfx.Group(), gfx.Group(), gfx.Group()
    g1.render_order, g2.render_order, g3.render_order = 5, 1, 3
    s.add(g1, g2, g3)
    for g in (g2, g1, g3):  # fill out of DFS-tail order
        for _ in range(8):
            p = _pts()
            p.local.position = rng.uniform(-5, 5, 3)
            g.add(p)
    g2.visible = False
    scenes["grouped_reordered"] = s

    s = gfx.Scene()
    s.add(gfx.AmbientLight())
    for i in range(30):
        p = _pts(render_order=i % 2)
        p.local.position = rng.uniform(-5, 5, 3)
        if i % 2 == 0:
            p.material.alpha_mode = "weighted_blend"
        s.add(p)
    scenes["weighted_mix"] = s

    return scenes


def test_flatscene_renderable_order_matches_reference():
    for sname, scene in _scenes().items():
        for cname, vm in _cameras().items():
            flat = FlatScene(scene, vm)
            flat.sort()
            got = [w.wobject for w in flat._wobject_wrappers]
            assert got == _reference(scene, vm), f"{sname}/{cname}"
