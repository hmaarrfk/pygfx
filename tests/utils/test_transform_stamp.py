"""The cache of the transforms must not depend on the resolution of the clock."""

import numpy as np
import pygfx as gfx
from pygfx.utils import transform


def test_changes_within_one_tick_of_the_clock(monkeypatch):
    # A clock that does not advance, as in a browser (Pyodide), where the
    # resolution of timers is limited to 0.1 ms or more
    frozen = transform.perf_counter_ns()
    monkeypatch.setattr(transform, "perf_counter_ns", lambda: frozen)

    ob = gfx.WorldObject()
    ob.world.x = 3
    assert ob.world.x == 3
    ob.world.y = 5
    assert ob.world.y == 5
    ob.local.z = 7
    assert np.allclose(ob.world.position, (3, 5, 7))
    assert np.allclose(ob.world.matrix[:3, 3], (3, 5, 7))

    child = gfx.WorldObject()
    ob.add(child)
    child.local.x = 1
    assert np.allclose(child.world.position, (4, 5, 7))
    ob.local.x = 10
    assert np.allclose(child.world.position, (11, 5, 7))

    camera = gfx.PerspectiveCamera(50)
    before = camera.last_modified
    camera.flag_update()
    assert camera.last_modified > before


def test_modification_stamp_increases(monkeypatch):
    monkeypatch.setattr(transform, "perf_counter_ns", lambda: 42)
    stamps = [transform.modification_stamp() for _ in range(5)]
    assert stamps == sorted(set(stamps))
