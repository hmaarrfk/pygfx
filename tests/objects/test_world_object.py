from math import pi
from unittest.mock import Mock, call
from weakref import ref
import pylinalg as la
import numpy as np
import numpy.testing as npt
import pytest

from pygfx import WorldObject
from pygfx.utils.transform import AffineTransform, RecursiveTransform
import pygfx as gfx


def test_traverse():
    root = WorldObject()

    layer1_child1 = WorldObject()
    root.add(layer1_child1)
    layer1_child2 = WorldObject()
    root.add(layer1_child2)

    layer2_child1 = WorldObject()
    layer1_child2.add(layer2_child1)
    layer2_child2 = WorldObject()
    layer1_child2.add(layer2_child2)

    mock = Mock()
    root.traverse(mock)
    mock.assert_has_calls(
        [
            call(root),
            call(layer1_child1),
            call(layer1_child2),
            call(layer2_child1),
            call(layer2_child2),
        ]
    )
    assert len(mock.mock_calls) == 5


def test_remove():
    root = WorldObject()

    layer1_child1 = WorldObject()
    root.add(layer1_child1)
    layer1_child2 = WorldObject()
    root.add(layer1_child2)

    layer2_child1 = WorldObject()
    layer1_child2.add(layer2_child1)
    layer2_child2 = WorldObject()
    layer1_child2.add(layer2_child2)

    root.remove(layer1_child2)
    # layer1_child2 removed
    assert layer1_child2.parent is None
    assert layer1_child2 not in root.children
    # layer1_child1 not removed
    assert layer1_child1.parent is root
    assert layer1_child1 in root.children


def test_update_matrix():
    root = WorldObject()
    root.local.position = (3, 6, 8)
    root.local.scale = (1, 1.2, 1)
    root.local.rotation = la.quat_from_euler((pi / 2, 0, 0))

    pos, rot, scale = la.mat_decompose(root.local.matrix)
    assert np.allclose(pos, root.local.position)
    assert np.allclose(rot, root.local.rotation)
    assert np.allclose(scale, root.local.scale)


def test_update_matrix_world():
    root = WorldObject()
    root.local.position = (-5, 8, 0)
    root.local.rotation = la.quat_from_euler((pi / 4, 0, 0))

    child1 = WorldObject()
    child1.local.position = (0, 0, 5)
    root.add(child1)

    child2 = WorldObject()
    child2.local.rotation = la.quat_from_euler((0, -pi / 4, 0))
    child1.add(child2)

    expected = (
        root.local @ child1.local @ child2.local @ la.vec_homogeneous((10, 10, 10))
    )

    actual = child2.world @ la.vec_homogeneous((10, 10, 10))

    # if there is a difference it's a floating point error
    assert np.allclose(actual, expected)


def test_reparenting():
    root = WorldObject()
    child1 = WorldObject()
    child2 = WorldObject()
    root.add(child1, child2)

    obj = WorldObject()
    child1.add(obj)

    assert obj.parent is child1
    assert obj in child1.children

    child2.add(obj)

    assert obj.parent is child2
    assert obj not in child1.children
    assert obj in child2.children


def test_no_cyclic_references():
    parent = WorldObject()
    child = WorldObject()
    parent.add(child)
    # Add object without creating a strong reference
    child.add(WorldObject())

    # Create a weak ref to the added child
    grandchild_ref = ref(child.children[0])
    # Calling the ref should retrieve the grandchild
    assert grandchild_ref() == child.children[0]

    # When the child is removed (and deleted), the grandchild
    # should also be garbage collected as there should be no
    # direct references anymore
    parent.remove(child)
    del child

    assert not grandchild_ref()


def test_adjust_children_order():
    root = WorldObject()
    child1 = WorldObject()
    child2 = WorldObject()

    root.add(child1, child2)

    expected_children = (child1, child2)
    assert len(root.children) == len(expected_children)
    for actual, expected in zip(root.children, expected_children, strict=True):
        assert actual is expected

    root.add(child2, before=child1)

    expected_children = (child2, child1)
    assert len(root.children) == len(expected_children)
    for actual, expected in zip(root.children, expected_children, strict=True):
        assert actual is expected

    child3 = WorldObject()
    root.add(child3, child1, before=child2)

    expected_children = (child3, child1, child2)
    assert len(root.children) == len(expected_children)
    for actual, expected in zip(root.children, expected_children, strict=True):
        assert actual is expected


def _ref_dfs(root):
    """Reference depth-first pre-order of root's descendants (excludes root)."""
    out = []

    def walk(o):
        for c in o._children:
            out.append(c)
            walk(c)

    walk(root)
    return out


def _index_order(root):
    root._ensure_subtree_index_order()
    return list(root._subtree_index.all_objects)


def test_subtree_index_dfs_order():
    # The incremental scene-graph index must present descendants in the same
    # depth-first pre-order the old full-walk produced, because that order
    # drives renderer-id assignment and the stable tiebreak for equal sort
    # keys. See pygfx issue #1298.

    # Adding to a group that already has a later sibling (not the DFS tail).
    root = WorldObject()
    g1, g2 = WorldObject(), WorldObject()
    root.add(g1, g2)
    x = WorldObject()
    g1.add(x)
    assert _index_order(root) == _ref_dfs(root)
    assert _index_order(root) == [g1, x, g2]

    # before= insertion.
    root = WorldObject()
    a, b, c = WorldObject(), WorldObject(), WorldObject()
    root.add(a, b)
    root.add(c, before=a)
    assert _index_order(root) == _ref_dfs(root) == [c, a, b]

    # Reparenting a subtree to an earlier sibling.
    root = WorldObject()
    g1, g2 = WorldObject(), WorldObject()
    root.add(g1, g2)
    leaf = WorldObject()
    g2.add(leaf)
    g1.add(leaf)
    assert _index_order(root) == _ref_dfs(root) == [g1, leaf, g2]


def test_subtree_index_dfs_order_hard_cases():
    # Move a *populated* subtree to an earlier position: its whole block must
    # relocate in DFS order, not just its root.
    root = WorldObject()
    a, b = WorldObject(), WorldObject()
    root.add(a, b)
    sub = WorldObject()
    s1, s2 = WorldObject(), WorldObject()
    sub.add(s1, s2)
    b.add(sub)  # sub currently under b (the tail) -> clean
    assert _index_order(root) == _ref_dfs(root)
    a.add(sub)  # move populated sub under a (earlier sibling) -> dirty
    assert _index_order(root) == _ref_dfs(root) == [a, sub, s1, s2, b]

    # Several non-tail mutations before a single render coalesce into one
    # correct reorder.
    root = WorldObject()
    groups = [WorldObject() for _ in range(4)]
    root.add(*groups)
    leaves = [WorldObject() for _ in range(4)]
    for g, leaf in zip(groups, leaves, strict=True):
        g.add(leaf)  # each adds to a non-tail group (except the last)
    assert _index_order(root) == _ref_dfs(root)

    # The element whose insertion dirtied the index is removed again before
    # the render: the reorder must still produce valid DFS order.
    root = WorldObject()
    g1, g2 = WorldObject(), WorldObject()
    root.add(g1, g2)
    tmp = WorldObject()
    g1.add(tmp)  # dirties (g1 is not the tail)
    g1.remove(tmp)  # remove the offending node before any render
    assert _index_order(root) == _ref_dfs(root) == [g1, g2]

    # clear() then rebuild.
    root = WorldObject()
    g1, g2 = WorldObject(), WorldObject()
    root.add(g1, g2)
    g1.add(WorldObject())
    root.clear()
    assert _index_order(root) == _ref_dfs(root) == []
    c1, c2 = WorldObject(), WorldObject()
    root.add(c1)
    c1.add(c2)
    assert _index_order(root) == _ref_dfs(root) == [c1, c2]


def test_subtree_index_renderable_and_light_order_follow_dfs():
    # renderable order is the stable tiebreak for equal sort keys, and light
    # order maps to uniform slots; both must follow DFS pre-order after a
    # non-tail insertion.
    import numpy as np

    def _point(name):
        p = gfx.Points(
            gfx.Geometry(positions=np.zeros((1, 3), "float32")),
            gfx.PointsMaterial(),
        )
        p.name = name
        return p

    scene = gfx.Scene()
    g1, g2 = gfx.Group(), gfx.Group()
    scene.add(g1, g2)
    p_late = _point("late")
    g2.add(p_late)
    p_early = _point("early")
    g1.add(p_early)  # non-tail insert -> dirty
    scene._ensure_subtree_index_order()

    ref = _ref_dfs(scene)
    renderable = list(scene._subtree_index.renderable)
    # renderable is the DFS order filtered to objects-with-material.
    assert renderable == [o for o in ref if o in scene._subtree_index.renderable]
    assert [o.name for o in renderable] == ["early", "late"]

    # Lights likewise follow DFS order.
    scene = gfx.Scene()
    g1, g2 = gfx.Group(), gfx.Group()
    scene.add(g1, g2)
    l_late = gfx.PointLight()
    l_late.name = "late"
    g2.add(l_late)
    l_early = gfx.PointLight()
    l_early.name = "early"
    g1.add(l_early)  # non-tail -> dirty
    scene._ensure_subtree_index_order()
    assert [o.name for o in scene._subtree_index.lights_point] == ["early", "late"]


def test_subtree_index_tail_append_stays_clean():
    # The common build patterns graft at the depth-first tail, so they must
    # NOT dirty the index (this is what keeps render a no-op walk-free path).
    root = WorldObject()
    for _ in range(10):
        root.add(WorldObject())  # flat append
    assert root._subtree_index._dfs_dirty is False

    root = WorldObject()
    for _ in range(4):
        g = WorldObject()
        root.add(g)
        for _ in range(3):
            g.add(WorldObject())  # fill each group before moving on
    assert root._subtree_index._dfs_dirty is False
    assert _index_order(root) == _ref_dfs(root)

    root = WorldObject()  # deep rightmost spine
    parent = root
    for _ in range(8):
        g = WorldObject()
        parent.add(g)
        parent = g
    assert root._subtree_index._dfs_dirty is False


def test_iter():
    class Foo(WorldObject):
        pass

    root = WorldObject()
    root.add(Foo(), Foo())
    assert len(list(root.iter(lambda x: isinstance(x, Foo)))) == 2

    root = Foo()
    root.add(Foo(), Foo())
    assert len(list(root.iter(lambda x: isinstance(x, Foo)))) == 3

    root = WorldObject()
    root.add(Foo(), WorldObject(), Foo())
    assert len(list(root.iter(lambda x: isinstance(x, Foo)))) == 2


def test_setting_world_transform():
    root = gfx.WorldObject()
    child = gfx.WorldObject()
    child2 = gfx.WorldObject()

    root.add(child)
    child.add(child2)

    root.local.position = (1, 2, 3)
    child.local.position = (4, 4, 4)
    child2.local.position = (10, 0, 0)

    assert np.allclose(child.local.position, (4, 4, 4))
    assert np.allclose(child.world.position, (5, 6, 7))

    assert np.allclose(child2.local.position, (10, 0, 0))
    assert np.allclose(child2.world.position, (15, 6, 7))

    child.world.position = (1, 2, 3)

    assert np.allclose(child.local.position, (0, 0, 0))
    assert np.allclose(child.world.position, (1, 2, 3))

    assert np.allclose(child2.local.position, (10, 0, 0))
    assert np.allclose(child2.world.position, (11, 2, 3))


def test_complex_multi_insert():
    children = []
    for _ in range(5):
        children.append(gfx.WorldObject())

    root = gfx.WorldObject()
    for _ in range(3):
        root.add(gfx.WorldObject())

    first = root.children[0]
    reference = root.children[1]
    root.add(*children, before=reference)

    # ensure children were inserted in order before `before`
    for from_root, child in zip(root.children[1:6], children, strict=True):
        assert from_root is child
    assert root.children[6] is reference

    root.add(first, before=reference)
    assert root.children[0] is not first
    assert root.children[5] is first
    assert root.children[6] is reference


def test_axis_getters():
    # "normal" euclidean space
    obj = gfx.WorldObject()
    assert np.allclose(obj.world.forward, (0, 0, 1))
    assert np.allclose(obj.world.up, (0, 1, 0))
    assert np.allclose(obj.world.right, (-1, 0, 0))

    # camera space
    camera = gfx.PerspectiveCamera()
    assert np.allclose(camera.world.forward, (0, 0, -1))
    assert np.allclose(camera.world.up, (0, 1, 0))
    assert np.allclose(camera.world.right, (1, 0, 0))

    # position offsets shouldn't matter
    obj.world.position = (0, 0, -10)
    assert np.allclose(obj.world.forward, (0, 0, 1))
    assert np.allclose(obj.world.up, (0, 1, 0))
    assert np.allclose(obj.world.right, (-1, 0, 0))

    # rotations should influence local orientations
    obj.world.rotation = la.quat_from_euler(np.pi / 2, order="Y")
    assert np.allclose(obj.local.up, (0, 1, 0))
    assert np.allclose(obj.local.right, (0, 0, 1))
    assert np.allclose(obj.local.forward, (1, 0, 0))

    obj.world.rotation = la.quat_from_euler((np.pi / 4, np.pi / 4), order="XZ")
    assert np.allclose(obj.local.forward, (0, -np.cos(np.pi / 4), np.sin(np.pi / 4)))

    print("")


def test_axis_setters():
    obj = gfx.WorldObject()

    obj.world.forward = (1, 0, 0)
    assert np.allclose(obj.world.forward, (1, 0, 0))
    assert np.allclose(
        obj.world.rotation, (0, np.sin((np.pi / 2) / 2), 0, np.cos((np.pi / 2) / 2))
    )

    obj.world.right = (-1, 0, 0)
    assert np.allclose(obj.world.right, (-1, 0, 0))
    assert np.allclose(obj.world.rotation, (0, 0, 0, 1))

    obj.world.right = (1, 0, 0)
    assert np.allclose(obj.world.right, (1, 0, 0))
    assert np.allclose(obj.world.up, (0, 1, 0))
    assert np.allclose(obj.world.forward, (0, 0, -1))

    # these vectors are unit

    obj.world.forward = (2, 0, 0)
    assert np.allclose(obj.world.forward, (1, 0, 0))

    obj.world.up = (1, 1, 1)
    assert np.allclose(obj.world.up, (1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)))


def test_reference_up():
    group = gfx.WorldObject()
    assert np.allclose(group.world.reference_up, (0, 1, 0))
    assert np.allclose(group.local.reference_up, (0, 1, 0))

    # reference_up is given in parent frame, so it is independent of the transform
    group.world.forward = (1, 1, 1)
    group.world.position = (1, 42, 13)
    assert np.allclose(group.world.reference_up, (0, 1, 0))
    assert np.allclose(group.local.reference_up, (0, 1, 0))

    # local reference_up does change if there is a parent since the parent frame may have
    # a transform relative to world
    obj1 = gfx.WorldObject()
    obj1.world.position = (0, 4, 9)
    group.add(obj1, keep_world_matrix=True)
    assert np.allclose(obj1.world.position, (0, 4, 9))
    reference_up = la.vec_transform(obj1.world.reference_up, group.world.inverse_matrix)
    world_origin = la.vec_transform((0, 0, 0), group.world.inverse_matrix)
    reference_up = reference_up - world_origin
    assert np.allclose(obj1.local.reference_up, reference_up)

    # but the parent remains unaffected by its children
    # as does the world reference
    assert np.allclose(group.local.reference_up, (0, 1, 0))
    assert np.allclose(group.world.reference_up, (0, 1, 0))
    assert np.allclose(obj1.world.reference_up, (0, 1, 0))

    # (world) reference_up is independent between objects
    obj2 = gfx.WorldObject()
    obj2.world.rotation = (0, 0, 1, 0)
    group.add(obj1, keep_world_matrix=True)

    obj3 = gfx.WorldObject()

    obj1.world.reference_up = (1, 2, 3)
    obj2.world.reference_up = (1, 0, 1)
    obj3.world.reference_up = (0, 42, 0)
    isqrt2 = 1 / np.sqrt(2)

    assert np.allclose(group.local.reference_up, (0, 1, 0))
    assert np.allclose(obj1.world.reference_up, la.vec_normalize((1, 2, 3)))
    assert np.allclose(obj2.world.reference_up, (isqrt2, 0, isqrt2))
    assert np.allclose(obj3.world.reference_up, (0, 1, 0))


def test_geometry_bounding_box():
    pos = np.array([(0, 0, 0), (1, 1, 1), (3, 3, 3)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box().tolist() == [[0, 0, 0], [3, 3, 3]]

    pos = np.array([(0, 1, 3), (3, 0, 1), (1, 3, 0)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box().tolist() == [[0, 0, 0], [3, 3, 3]]

    pos = np.array([(1, 1, 1), (1, 1, 1), (1, 1, 1)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box().tolist() == [[1, 1, 1], [1, 1, 1]]

    pos = np.array([(0, 1, 2), (0, 1, 2), (0, 1, 2)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box().tolist() == [[0, 1, 2], [0, 1, 2]]

    pos = np.array([(0, 0, np.nan), (1, 1, 1), (2, 2, 2)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box().tolist() == [[1, 1, 1], [2, 2, 2]]

    pos = np.array([(0, np.inf, 0), (1, 1, 1), (2, 2, 2)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box().tolist() == [[1, 1, 1], [2, 2, 2]]

    pos = np.array([(-np.inf, 0, 0), (1, 1, 1), (2, 2, 2)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box().tolist() == [[1, 1, 1], [2, 2, 2]]

    # Empty buffer is not allowed
    # pos = np.zeros((0, 3), np.float32)
    # ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    # assert ob.get_geometry_bounding_box() is None

    pos = np.array([(-np.inf, 0, 0), (1, np.nan, 1)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    assert ob.get_geometry_bounding_box() is None


def test_geometry_bounding_sphere():
    pos = np.array([(0, 0, -8), (1, 1, 1), (2, 2, 10)], np.float32)
    ob = gfx.WorldObject(gfx.Geometry(positions=pos), None)
    bsphere = ob.get_bounding_sphere()
    bsphere_via_aabb = la.aabb_to_sphere(ob.get_geometry_bounding_box())

    assert np.allclose(bsphere, [1, 1, 1, 9.1104])
    assert np.allclose(bsphere_via_aabb, [1, 1, 1, 9.1104])


def test_bounding_box():
    scene = gfx.Group()

    # Start out without a bounding box

    assert scene.get_bounding_box() is None

    # Add a camera, still no box
    scene.add(gfx.PerspectiveCamera())
    assert scene.get_bounding_box() is None

    # Add a light, still no box
    scene.add(gfx.DirectionalLight())
    assert scene.get_bounding_box() is None

    # Add a point, we've got a bbox with no volume
    ob = gfx.Points(
        gfx.Geometry(positions=np.array([(0, 0, 0)], np.float32)),
        gfx.PointsMaterial(),
    )
    scene.add(ob)
    assert scene.get_bounding_box().tolist() == [[0, 0, 0], [0, 0, 0]]

    # Add another point to get a larger bbox
    ob = gfx.Points(
        gfx.Geometry(positions=np.array([(0, 3, 1)], np.float32)),
        gfx.PointsMaterial(),
    )
    scene.add(ob)
    assert scene.get_bounding_box().tolist() == [[0, 0, 0], [0, 3, 1]]

    # Adding a point with no valid positions ... has no effect
    ob = gfx.Points(
        gfx.Geometry(positions=np.array([(99, np.nan, 99)], np.float32)),
        gfx.PointsMaterial(),
    )
    scene.add(ob)
    assert ob.get_bounding_box() is None
    assert ob.get_world_bounding_box() is None
    assert scene.get_bounding_box().tolist() == [[0, 0, 0], [0, 3, 1]]

    # Add a point that is transformed, to make sure that is taken into account
    ob = gfx.Points(
        gfx.Geometry(positions=np.array([(-1, 0, 2)], np.float32)),
        gfx.PointsMaterial(),
    )
    ob.local.scale = 3, 3, 3
    ob.local.x = -2
    scene.add(ob)
    assert scene.get_bounding_box().tolist() == [[-5, 0, 0], [0, 3, 6]]

    # Create a point with all of the above ... to make sure the own geo is taken into account
    point_with_children = gfx.Points(
        gfx.Geometry(positions=np.array([(9, 1, 0)], np.float32)),
        gfx.PointsMaterial(),
    )
    point_with_children.add(scene)
    assert point_with_children.get_bounding_box().tolist() == [[-5, 0, 0], [9, 3, 6]]


def test_scale_preservation():
    """Test that the original scaling component is preserved through
    matrix composition roundtrips"""
    ob = gfx.WorldObject()
    s = (1, -2, 3)
    ob.local.scale = s
    # without scale preservation in matrix compose -> decompose roundtrip
    # ob.local.scale becomes (-1, 2, 3)
    npt.assert_array_almost_equal(ob.local.scale, s)

    child = gfx.WorldObject()
    ob.add(child)
    npt.assert_array_almost_equal(child.local.scale, [1, 1, 1])
    npt.assert_array_almost_equal(child.world.scale, s)

    s2 = (-4, -4, 4)
    child.local.scale = s2
    npt.assert_array_almost_equal(child.local.scale, s2)
    npt.assert_array_almost_equal(child.world.scale, [-4, 8, 12])


def test_scaling_signs_manual_matrix():
    """Test that if the matrix is set directly, everything still works
    and we do not manage to reconstruct the original signs."""
    ob = gfx.WorldObject()
    s = (1, -2, 3)
    expected = (-1, 2, 3)
    t = la.mat_compose(
        (10, 0, 0),
        la.quat_from_axis_angle((0, 0, 1), np.pi / 2),
        s,
    )
    ob.local.matrix = t

    with pytest.raises(AssertionError):
        npt.assert_array_almost_equal(ob.local.scale, s)
    npt.assert_array_almost_equal(ob.local.scale, expected)

    child = gfx.WorldObject()
    ob.add(child)
    npt.assert_array_almost_equal(child.local.scale, [1, 1, 1])
    npt.assert_array_almost_equal(child.world.scale, expected)

    s2 = (-4, -4, 4)
    expected2 = np.array(expected) * np.array(s2)
    child.local.scale = s2
    npt.assert_array_almost_equal(child.local.scale, s2)
    npt.assert_array_almost_equal(child.world.scale, expected2)


def test_rotation_derived():
    obj = gfx.WorldObject()
    e1 = np.array([0, np.pi / 2, 0])
    q1 = la.quat_from_euler(e1, order="XYZ")
    m1 = la.mat_from_quat(q1)

    obj.local.rotation = q1
    npt.assert_array_almost_equal(obj.local.rotation, q1)
    npt.assert_array_almost_equal(obj.local.rotation_matrix, m1)
    npt.assert_array_almost_equal(obj.local.forward, [1, 0, 0])
    npt.assert_array_almost_equal(obj.local.right, [0, 0, 1])
    npt.assert_array_almost_equal(obj.local.up, [0, 1, 0])

    obj.local.rotation_matrix = m1
    npt.assert_array_almost_equal(obj.local.rotation, q1)
    npt.assert_array_almost_equal(obj.local.rotation_matrix, m1)
    npt.assert_array_almost_equal(obj.local.forward, [1, 0, 0])
    npt.assert_array_almost_equal(obj.local.right, [0, 0, 1])
    npt.assert_array_almost_equal(obj.local.up, [0, 1, 0])

    obj.local.euler = e1
    npt.assert_array_almost_equal(obj.local.rotation, q1)
    npt.assert_array_almost_equal(obj.local.rotation_matrix, m1)
    npt.assert_array_almost_equal(obj.local.forward, [1, 0, 0])
    npt.assert_array_almost_equal(obj.local.right, [0, 0, 1])
    npt.assert_array_almost_equal(obj.local.up, [0, 1, 0])

    obj.local.euler = [0, 0, 0]
    npt.assert_array_almost_equal(obj.local.rotation, [0, 0, 0, 1])
    npt.assert_array_almost_equal(obj.local.rotation_matrix, np.eye(4))
    npt.assert_array_almost_equal(obj.local.forward, [0, 0, 1])
    npt.assert_array_almost_equal(obj.local.right, [-1, 0, 0])
    npt.assert_array_almost_equal(obj.local.up, [0, 1, 0])

    obj.local.euler_x = np.pi / 2
    npt.assert_array_almost_equal(
        obj.local.rotation, la.quat_from_euler(np.pi / 2, order="X")
    )

    obj.local.euler_y = np.pi / 2
    npt.assert_array_almost_equal(
        obj.local.rotation, la.quat_from_euler(np.pi / 2, order="Y")
    )

    obj.local.euler_z = np.pi / 2
    npt.assert_array_almost_equal(
        obj.local.rotation, la.quat_from_euler(np.pi / 2, order="Z")
    )


def test_update_laziness():
    """Assert that world matrices (and other transform derived properties)
    are evaluated lazily.

    An important distinction is that pygfx API methods such as
    WorldObject.add should not trigger transform evaluations, as it
    defeats the purpose of lazy evaluation.

    In other words, by default, the world matrix should not be computed
    until just before an object is rendered. The only exception is when
    user code (e.g. picking, collision) requires the world matrix earlier.
    """
    # this is admittedly a very awkward line
    # but at least it will fail if/when the caching mechanism is
    # ever changed, invalidating this unit test's implementation
    cache_attr = RecursiveTransform.__dict__["_matrix"].name

    a = gfx.WorldObject()
    b = gfx.WorldObject()

    assert not hasattr(a.world, cache_attr)
    assert not hasattr(b.world, cache_attr)

    a.add(b)

    assert not hasattr(a.world, cache_attr)
    assert not hasattr(b.world, cache_attr)

    a.local.position = (1, 2, 3)

    assert not hasattr(a.world, cache_attr)
    assert not hasattr(b.world, cache_attr)


def test_update_propagation():
    """Simple test to check that transform invalidation propagates
    through multiple layers of the scene graph
    """
    root = gfx.WorldObject()
    level1 = gfx.WorldObject()
    level2 = gfx.WorldObject()
    level3 = gfx.WorldObject()

    root.add(level1)
    level1.add(level2)
    level2.add(level3)

    l3wlm = level3.world.last_modified
    l3llm = level3.local.last_modified

    root.world.position = (10, 10, 10)

    assert level3.world.last_modified > l3wlm
    assert level3.local.last_modified == l3llm


def test_update_propagation_reference_up():
    """Test to check that reference_up invalidation propagates
    through multiple layers of the scene graph
    """
    root = gfx.WorldObject()
    level1 = gfx.WorldObject()
    level2 = gfx.WorldObject()
    level3 = gfx.WorldObject()

    root.add(level1)
    level1.add(level2)
    level2.add(level3)

    # the local reference up setter
    # interestingly should not affect local transform
    # but it should affect world transform
    # since it determines the parent frame's reference up
    # for children
    l1wlm = level1.world.last_modified
    l1llm = level1.local.last_modified
    l3wlm = level3.world.last_modified
    l3llm = level3.local.last_modified

    level1.local.reference_up = (0, 0, 1)

    assert level1.world.last_modified > l1wlm
    assert level1.local.last_modified == l1llm
    assert level3.world.last_modified > l3wlm
    # this one is counter-intuitive
    # because the local reference_up IS affected
    # by the parent's frame being changed
    # but the property is only used in setters
    # so there is no local cache to invalidate
    assert level3.local.last_modified == l3llm


def test_transform_state_basis():
    """Test that state_basis=="matrix" works properly."""
    ob = gfx.WorldObject()

    pos = (5, 7, 8)
    ob.local.position = pos

    ob.local.state_basis = "matrix"
    # check that the position is maintained after toggling
    npt.assert_allclose(ob.local.position, pos)

    mat = la.mat_compose(
        pos,
        la.quat_from_axis_angle((0, 0, 1), np.pi / 2),
        (1, -2, 3),
    )
    ob.local.matrix = mat
    npt.assert_array_equal(ob.local.matrix, mat)
    npt.assert_allclose(ob.local.scale, (-1, 2, 3))
    npt.assert_allclose(ob.local.scaling_signs, (1, 1, 1))  # bypassed in matrix mode

    ob.local.state_basis = "components"
    npt.assert_allclose(ob.local.position, pos)
    npt.assert_allclose(ob.local.scale, (-1, 2, 3))
    npt.assert_allclose(
        ob.local.scaling_signs, (-1, 1, 1)
    )  # derived in components mode


def test_transform_multiply():
    for state_basis in ["matrix", "components"]:
        for transforms in [
            ("position", (0, 1, 0), "scale", (1.0, 1.5, 1.0)),
            ("scale", (1.0, 1.5, 1.0), "position", (0, 1, 0)),
            ("position", (0, 1, 0), "euler_z", 0.5),
            ("euler_z", 0.5, "position", (0, 1, 0)),
            ("euler_z", 0.5, "scale", (1.0, 1.5, 1.0)),
            ("scale", (1.0, 1.5, 1.0), "euler_z", 0.5),  # shear here :)
        ]:
            ttype1, value1, ttype2, value2 = transforms

            t1 = AffineTransform(state_basis=state_basis)
            t2 = AffineTransform(state_basis=state_basis)
            setattr(t1, ttype1, value1)
            setattr(t2, ttype2, value2)

            # Get matrix in two ways. The result must be the same.
            ma = t1.matrix @ t2.matrix
            mb = (t1 @ t2).matrix
            assert np.allclose(ma, mb)


def test_shear_support():
    t1 = AffineTransform(state_basis="components")
    t1.scale_y = 1.5

    t2 = AffineTransform(state_basis="components")
    t2.euler_z = 0.5

    # Get matrix in two ways. The result must be the same.
    ma = t1.matrix @ t2.matrix
    mb = (t1 @ t2).matrix
    assert np.allclose(ma, mb)
