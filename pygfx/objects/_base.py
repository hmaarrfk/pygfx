from __future__ import annotations

import random
import weakref
import threading
from typing import Any, Callable, ClassVar, Iterator, List, Tuple
import pylinalg as la
from time import perf_counter_ns

import numpy as np

from ..resources import Buffer, Texture
from ..utils import array_from_shadertype, logger
from ..utils.trackable import Trackable
from ..utils.bounds import Bounds
from ._events import EventTarget
from ..utils.transform import (
    AffineTransform,
    RecursiveTransform,
)
from ..geometries import Geometry
from ..materials import Material


class IdProvider:
    """Object for internal use to manage world object id's."""

    def __init__(self):
        self._ids_in_use = set([0])
        self._map = weakref.WeakValueDictionary()
        self._lock = threading.RLock()

    def claim_id(self, wobject: WorldObject) -> int:
        """Used by wobjects to claim an id."""
        # We don't simply count up, but keep a pool of ids. This is
        # because an application *could* create and discard objects at
        # a high rate, so we want to be able to re-use these ids.
        #
        # Some numbers:
        # * 4_294_967_296 (2**32) max number for u32
        # * 2_147_483_647 (2**31 -1) max number for i32.
        # *    16_777_216 max integer that can be stored exactly in f32
        # *     4_000_000 max integer that survives being passed as a varying (in my tests)
        # *     1_048_575 is ~1M is 2**20 seems like a good max scene objects.
        # *    67_108_864 is ~50M is 2**26 seems like a good max vertex count.
        #                 which leaves 64-20-26=18 bits for any other picking info.

        # Max allowed id, inclusive
        id_max = 1_048_575  # 2*20-1

        # The max number of ids. This is a bit less to avoid choking
        # when there are few free id's left.
        max_items = 1_000_000

        with self._lock:
            if len(self._ids_in_use) >= max_items:
                raise RuntimeError("Max number of objects reached.")
            id = 0
            while id in self._ids_in_use:
                id = random.randint(1, id_max)
            self._ids_in_use.add(id)
            self._map[id] = wobject

        return id

    def release_id(self, wobject: WorldObject, id: int) -> None:
        """Release an id associated with a wobject."""
        if id > 0:
            with self._lock:
                self._ids_in_use.discard(id)
                self._map.pop(id, None)

    def get_object_from_id(self, id: int) -> WorldObject | None:
        """Return the wobject associated with an id, or None."""
        return self._map.get(id)


id_provider = IdProvider()


class _SubtreeIndex:
    """Categorized flat lists of *descendants* of a world object.

    Each ``WorldObject`` owns one of these. ``self`` is **not** an entry of
    its own index (only its strict descendants are) — this keeps the index
    free of reference cycles, so removed subtrees become unreachable as
    soon as their parents drop them, without needing cyclic GC.

    The renderer iterates these flat lists at render time instead of
    walking the scene graph; the lists are kept in sync incrementally by
    ``WorldObject.add``/``remove`` and by the relevant property setters
    (``cast_shadow``, ``geometry``, ``material``).

    The lists must be in scene-graph (DFS pre-order) order, because:
    - ``all_objects`` order drives renderer-id assignment, and
    - ``renderable`` order is the stable tiebreak for objects with equal
      sort keys (e.g. coplanar transparent objects), so it must match what
      the old depth-first walk produced.

    Appending a freshly-added subtree keeps DFS order **as long as it is
    grafted at the depth-first tail** (the common case: building a scene by
    successive ``add``s, or filling a group before moving on). A non-tail
    insertion — adding to a group that already has later siblings, or using
    ``add(..., before=...)`` — would put entries in the wrong place, so it
    instead sets ``_dfs_dirty`` and the order is rebuilt lazily on the next
    render (see ``WorldObject._ensure_subtree_index_order``). This keeps the
    no-traversal fast path for steady-state render loops while staying
    correct under arbitrary mutation.

    All stored values are ``None``; the dicts act as ordered sets.
    """

    __slots__ = (
        "_dfs_dirty",
        "all_objects",
        "lights_ambient",
        "lights_directional",
        "lights_point",
        "lights_spot",
        "renderable",
        "shadow_casters",
    )

    def __init__(self):
        self.all_objects = {}
        self.lights_point = {}
        self.lights_directional = {}
        self.lights_spot = {}
        self.lights_ambient = {}
        self.shadow_casters = {}
        self.renderable = {}
        # Set when a non-DFS-tail insertion leaves the lists out of
        # scene-graph order; cleared by a lazy reorder at render time.
        self._dfs_dirty = False

    def add_object(self, obj: "WorldObject") -> None:
        """Insert ``obj`` into the relevant categorized lists."""
        self.all_objects[obj] = None
        kind = obj._subtree_light_kind
        if kind == "point":
            self.lights_point[obj] = None
        elif kind == "directional":
            self.lights_directional[obj] = None
        elif kind == "spot":
            self.lights_spot[obj] = None
        elif kind == "ambient":
            self.lights_ambient[obj] = None
        if obj._caster_in_index:
            self.shadow_casters[obj] = None
        if obj._renderable_in_index:
            self.renderable[obj] = None

    def remove_object(self, obj: "WorldObject") -> None:
        """Remove ``obj`` from all categorized lists (no-op if absent)."""
        self.all_objects.pop(obj, None)
        kind = obj._subtree_light_kind
        if kind == "point":
            self.lights_point.pop(obj, None)
        elif kind == "directional":
            self.lights_directional.pop(obj, None)
        elif kind == "spot":
            self.lights_spot.pop(obj, None)
        elif kind == "ambient":
            self.lights_ambient.pop(obj, None)
        self.shadow_casters.pop(obj, None)
        self.renderable.pop(obj, None)

    def merge_in(self, other: "_SubtreeIndex") -> None:
        """Merge all of ``other``'s entries into ``self`` (preserving order)."""
        self.all_objects.update(other.all_objects)
        self.lights_point.update(other.lights_point)
        self.lights_directional.update(other.lights_directional)
        self.lights_spot.update(other.lights_spot)
        self.lights_ambient.update(other.lights_ambient)
        self.shadow_casters.update(other.shadow_casters)
        self.renderable.update(other.renderable)

    def remove_in(self, other: "_SubtreeIndex") -> None:
        """Remove all of ``other``'s entries from ``self``."""
        for ob in other.all_objects:
            self.all_objects.pop(ob, None)
        for ob in other.lights_point:
            self.lights_point.pop(ob, None)
        for ob in other.lights_directional:
            self.lights_directional.pop(ob, None)
        for ob in other.lights_spot:
            self.lights_spot.pop(ob, None)
        for ob in other.lights_ambient:
            self.lights_ambient.pop(ob, None)
        for ob in other.shadow_casters:
            self.shadow_casters.pop(ob, None)
        for ob in other.renderable:
            self.renderable.pop(ob, None)


class WorldObject(EventTarget, Trackable):
    """Base class for objects.

    This class represents objects in the world, i.e., the scene graph.Each
    WorldObject has geometry to define it's data, and material to define its
    appearance. The object itself is only responsible for defining object
    hierarchies (parent / children) and its position and orientation in the
    world.

    Parameters
    ----------
    geometry : Geometry
        The data defining the shape of the object. See the documentation
        on the different WorldObject subclasses for what attributes the
        geometry should and may have.
    material : Material
        The data defining the appearance of the object.
    visible : bool
        Whether the object is visible.
    render_order : float
        Value that helps controls the order in which objects are rendered.
    name : str
        The name of the object.

    Notes
    -----
    Use :class:`Group` to collect multiple world objects into a single empty
    world object.

    See Also
    --------
    pygfx.utils.transform.AffineBase
        Various getters and setters defined on ``obj.local`` and ``obj.world``.
    pygfx.utils.transform.AffineTransform
        The class used to implement ``obj.local``.
    pygfx.utils.transform.RecursiveTransform
        The class used to implement ``obj.world``.

    """

    _FORWARD_IS_MINUS_Z = False  # Default is +Z (lights and cameras use -Z)

    # Marker used by the incremental scene-graph index. Light subclasses set
    # this to one of "point", "directional", "spot", "ambient" so the index
    # can categorize them without isinstance checks (and without circular
    # imports between _base.py and _lights.py).
    _subtree_light_kind: ClassVar[str | None] = None

    # Marker used by the index for ``Group``-aware ``render_order``
    # propagation. ``Group`` (and its subclasses, e.g. ``Scene``) sets this
    # to True so child objects can resolve the closest Group ancestor
    # without isinstance checks.
    _subtree_is_group: ClassVar[bool] = False

    _id = 0

    # The uniform type describes the structured info for this object, which represents
    # every "property" that a renderer would need to know in order to visualize it.
    # Put larger items first for alignment, also note that host-sharable structs
    # align at power-of-two only, so e.g. vec3 needs padding.
    # todo: rename uniform to info or something?

    uniform_type: ClassVar[dict[str, str]] = dict(
        world_transform="4x4xf4",
        world_transform_inv="4x4xf4",
        global_id="u4",
        renderer_id="u4",
    )

    def __init__(
        self,
        geometry: Geometry | None = None,
        material: Material | None = None,
        *,
        visible: bool = True,
        render_order: float = 0,
        name: str = "",
        cast_shadow: bool = False,
        receive_shadow: bool = False,
        nonlinear_transform: str | None = None,
    ) -> None:
        super().__init__()
        self._parent: weakref.ReferenceType[WorldObject] | None = None

        #: Subtrees of the scene graph that depend on this object.
        self._children: List[WorldObject] = []

        # Incremental scene-graph index state. Initialised lazily after
        # all property setters have run (some setters update the index, so
        # we need them to be no-ops until ``_subtree_index`` is created).
        self._subtree_index: _SubtreeIndex | None = None
        self._caster_in_index = False
        self._renderable_in_index = False
        self._effective_visible = True
        self._closest_group_ancestor_ref: weakref.ReferenceType[WorldObject] | None = (
            None
        )

        self.geometry = geometry
        self.material = material

        self.name = name

        # Compose complete uniform type
        buffer = Buffer(array_from_shadertype(self.uniform_type), force_contiguous=True)
        buffer.data["world_transform"] = np.eye(4)
        buffer.data["world_transform_inv"] = np.eye(4)

        self._world_last_modified = perf_counter_ns()

        #: The object's transform expressed in parent space.
        self.local = AffineTransform(is_camera_space=self._FORWARD_IS_MINUS_Z)
        #: The object's transform expressed in world space.
        self.world = RecursiveTransform(
            self.local, is_camera_space=self._FORWARD_IS_MINUS_Z, reference_up=(0, 1, 0)
        )

        # Set id that is global to the process. For identification of the object
        self._global_id = id_provider.claim_id(self)
        buffer.data["global_id"] = self._global_id

        # Init id assigned by the renderer. This id will be set once, by the first
        # renderer to render this object. That way, within one "visualization"
        # (consisting of a renderer and a scene with fresh objects) these id's are
        # reproducable. I.e. validation examples have reproducable locally unique id's.
        self._renderer_id = 0

        # Bounds
        self._bounds_geometry = None
        self._bounds_geometry_rev = 0

        #: The GPU data of this WorldObject.
        self.uniform_buffer = buffer

        # Init visibility and render props
        self.visible = visible
        self.render_order = render_order
        self.cast_shadow = cast_shadow
        self.receive_shadow = receive_shadow
        self.nonlinear_transform = nonlinear_transform

        self.name = name

        # Now bootstrap the subtree index. By convention our index holds
        # only our *descendants*, not ourselves; ``self`` is registered
        # into ancestor indices when added to a parent. Keeping self out
        # of its own index prevents reference cycles.
        self._subtree_index = _SubtreeIndex()
        self._caster_in_index = self._compute_is_shadow_caster()
        self._renderable_in_index = self._material is not None
        self._effective_visible = bool(self._store.visible)

    def _assign_renderer_id(self, id):
        if self._renderer_id == 0:
            assert id > 0
            self._renderer_id = id
            self.uniform_buffer.data["renderer_id"] = id
            self.uniform_buffer.update_full()
            return 1
        return 0

    def _update_object(self):
        """This gets called (by the renderer) right before being drawn. Good time for lazy updates."""
        world_last_modified = self.world.last_modified
        if world_last_modified > self._world_last_modified:
            self._world_last_modified = world_last_modified
            self._update_world_transform()

    def _update_world_transform(self):
        """This gets called right before being drawn, when the world transform has changed."""
        np.copyto(
            self.uniform_buffer.data["world_transform"],
            self.world.matrix.T,
            casting="unsafe",
        )
        np.copyto(
            self.uniform_buffer.data["world_transform_inv"],
            self.world.inverse_matrix.T,
            casting="unsafe",
        )
        self.uniform_buffer.update_full()

    # ----- Incremental scene-graph index helpers -----

    def _compute_is_shadow_caster(self) -> bool:
        """Whether self qualifies as a shadow caster right now."""
        return bool(self.cast_shadow) and self._store.geometry is not None

    def _iter_self_and_ancestors(self):
        """Yield self, then each ancestor walking up the parent chain."""
        node = self
        while node is not None:
            yield node
            parent_ref = node._parent
            node = parent_ref() if parent_ref is not None else None

    def _iter_ancestors(self):
        """Yield each strict ancestor (parent, grandparent, ...)."""
        parent_ref = self._parent
        node = parent_ref() if parent_ref is not None else None
        while node is not None:
            yield node
            parent_ref = node._parent
            node = parent_ref() if parent_ref is not None else None

    def _refresh_caster_status(self) -> None:
        """Recompute self's shadow-caster status and propagate to every
        ancestor's ``shadow_casters`` list."""
        if self._subtree_index is None:
            return  # still in __init__, nothing to register against
        new_caster = self._compute_is_shadow_caster()
        if new_caster == self._caster_in_index:
            return
        self._caster_in_index = new_caster
        for ancestor in self._iter_ancestors():
            casters = ancestor._subtree_index.shadow_casters
            if new_caster:
                casters[self] = None
            else:
                casters.pop(self, None)

    def _refresh_renderable_status(self) -> None:
        """Recompute self's renderable status and propagate to every
        ancestor's ``renderable`` list."""
        if self._subtree_index is None:
            return
        new_renderable = self._material is not None
        if new_renderable == self._renderable_in_index:
            return
        self._renderable_in_index = new_renderable
        for ancestor in self._iter_ancestors():
            renderable = ancestor._subtree_index.renderable
            if new_renderable:
                renderable[self] = None
            else:
                renderable.pop(self, None)

    def _refresh_visibility_subtree(self) -> None:
        """Recompute ``_effective_visible`` for self and, if it changed,
        cascade to descendants. Called from the ``visible`` setter."""
        if self._subtree_index is None:
            return
        parent = self.parent
        parent_eff = parent._effective_visible if parent is not None else True
        new_eff = parent_eff and bool(self._store.visible)
        if new_eff == self._effective_visible:
            return
        self._effective_visible = new_eff
        for child in self._children:
            child._refresh_visibility_subtree()

    def _cascade_topology_change(
        self,
        parent_eff_visible: bool,
        parent_closest_group_ref: "weakref.ReferenceType[WorldObject] | None",
    ) -> None:
        """Recompute ``_effective_visible`` and ``_closest_group_ancestor_ref``
        for self and all descendants.

        Called when a subtree is attached to a new parent, or detached.
        The caller passes the parent's effective-visibility and the
        weakref to the parent's closest-Group-ancestor *for self*. (If
        the new parent is itself a Group, the parent is self's closest
        Group ancestor.)
        """
        self._effective_visible = parent_eff_visible and bool(self._store.visible)
        self._closest_group_ancestor_ref = parent_closest_group_ref

        if self._subtree_is_group:
            child_group_ref = weakref.ref(self)
        else:
            child_group_ref = parent_closest_group_ref

        for child in self._children:
            child._cascade_topology_change(self._effective_visible, child_group_ref)

    def _group_ref_for_my_children(
        self,
    ) -> "weakref.ReferenceType[WorldObject] | None":
        """The closest-Group-ancestor weakref to give to a child being
        attached under self.

        If self is a Group, that's a weakref to self; otherwise it's
        whatever self's own closest-Group-ancestor is.
        """
        if self._subtree_is_group:
            return weakref.ref(self)
        return self._closest_group_ancestor_ref

    def _attach_subtree_under(self, parent: "WorldObject") -> None:
        """Apply the index/visibility/group bookkeeping for adopting
        ``self`` (with its existing subtree) as a freshly-added child of
        ``parent``.

        Precondition: ``self._parent`` already points to ``parent`` and
        ``self`` is in ``parent._children``.
        """
        # Push self + its descendants up through every ancestor's index.
        # ``self`` is registered explicitly because it is not present in
        # its own ``_subtree_index`` (descendants only).
        #
        # Appending keeps DFS pre-order only while the new subtree is
        # grafted at the depth-first tail, i.e. every node on the path from
        # the ancestor down to ``self`` is the last child of its parent.
        # ``at_tail`` tracks that; once it fails for an ancestor, that
        # ancestor (and every higher one) is marked dirty for a lazy
        # reorder at render time.
        own_index = self._subtree_index
        at_tail = True
        child = self
        for ancestor in parent._iter_self_and_ancestors():
            anc_index = ancestor._subtree_index
            anc_index.add_object(self)
            anc_index.merge_in(own_index)
            if not (at_tail and ancestor._children[-1] is child):
                at_tail = False
                anc_index._dfs_dirty = True
            child = ancestor

        # Cascade visibility + group ancestor down through the new subtree.
        self._cascade_topology_change(
            parent._effective_visible, parent._group_ref_for_my_children()
        )

    def _detach_subtree_from(self, parent: "WorldObject") -> None:
        """Apply the index/visibility/group bookkeeping for removing
        ``self`` (with its subtree) from ``parent``.

        Precondition: ``self`` has been removed from ``parent._children``
        but ``self._parent`` has not yet been cleared.
        """
        own_index = self._subtree_index
        for ancestor in parent._iter_self_and_ancestors():
            anc_index = ancestor._subtree_index
            anc_index.remove_object(self)
            anc_index.remove_in(own_index)

        # Cascade visibility + group ancestor as if self is now a root.
        self._cascade_topology_change(True, None)

    def _ensure_subtree_index_order(self) -> None:
        """Rebuild this object's index lists in scene-graph (DFS pre-order)
        order if a non-tail insertion left them dirty.

        This is a no-op in the common case — appends at the depth-first tail
        keep the lists ordered, so ``_dfs_dirty`` stays clear and a
        steady-state render loop never walks the graph. Removals preserve the
        relative order of the remaining entries, so only insertions can dirty
        the index. When dirty, this costs one O(N) walk of the subtree, paid
        once on the first render after the mutation.
        """
        index = self._subtree_index
        if index is None or not index._dfs_dirty:
            return

        # Depth-first pre-order over the descendants (self is excluded from
        # its own index by design).
        ordered = {}
        stack = self._children[::-1]
        while stack:
            node = stack.pop()
            ordered[node] = None
            children = node._children
            if children:
                stack.extend(children[::-1])

        # Reorder each categorized list to follow ``ordered`` while keeping
        # its existing membership (built incrementally and already correct).
        index.all_objects = ordered
        index.renderable = {o: None for o in ordered if o in index.renderable}
        index.shadow_casters = {o: None for o in ordered if o in index.shadow_casters}
        index.lights_point = {o: None for o in ordered if o in index.lights_point}
        index.lights_directional = {
            o: None for o in ordered if o in index.lights_directional
        }
        index.lights_spot = {o: None for o in ordered if o in index.lights_spot}
        index.lights_ambient = {o: None for o in ordered if o in index.lights_ambient}
        index._dfs_dirty = False

    def __repr__(self):
        return f"<pygfx.{self.__class__.__name__} {self.name} at {hex(id(self))}>"

    def __del__(self):
        id_provider.release_id(self, self.id)
        self.local._set_wrapper(
            None
        )  # break the circular reference so GC has it a little easier

    def _self(self):
        """Get self. This looks silly, but in case the WorldObject is
        wrapped in a weakproxy, like FastPlotLib does, this gets the
        real wobject, so we can hash it. You cannot dereference a weak.proxy.
        """
        return self

    @property
    def up(self) -> np.ndarray:
        """
        Relic of old WorldObjects that aliases with the new ``transform.up``
        direction. Prefer `obj.world.reference_up` instead.

        """

        logger.warning(
            "`WorldObject.up` is deprecated. Use `WorldObject.world.reference_up` instead.",
        )

        return self.world.reference_up

    @up.setter
    def up(self, value: np.ndarray) -> None:
        logger.warning(
            "`WorldObject.up` is deprecated. Use `WorldObject.world.reference_up` instead.",
        )

        self.world.reference_up = np.asarray(value)

    @property
    def id(self) -> int:
        """An integer id smaller than 2**31 (read-only)."""
        return self._global_id

    @property
    def visible(self) -> bool:
        """Whether is object is rendered or not. Default True."""
        return self._store.visible

    @visible.setter
    def visible(self, visible: bool) -> None:
        self._store.visible = bool(visible)
        self._refresh_visibility_subtree()

    @property
    def render_order(self) -> float:
        """Per-object rendering priority used to fine-tune the draw order within a render queue.

        Objects with higher render_order values are rendered later than those with lower values.
        This affects both opaque and transparent objects and can be used to resolve z-fighting,
        or control draw order beyond automatic depth sorting.

        If the object is part of a Group, the group's render_order is considered first (that is the group_order of the object).

        The order in wich objects are rendered is:
            1. the ``material.render_queue``.
            2. the ``ob.parent.render_order`` (if ``isinstance(ob.parent, gfx.Group)``).
            3. the ``ob.render_order``.
            4. the distance to camera (if ``renderer.sort_objects==True``).
            5. the position of the object in the scene graph.

        Also see ``material.render_queue``.
        """
        # Note: the render order is on the object, not the material, because it affects
        # a specific object, and materials are often shared between multiple objects.
        return self._store.render_order

    @render_order.setter
    def render_order(self, value: float) -> None:
        self._store.render_order = float(value)

    @property
    def render_mask(self):
        return None

    @render_mask.setter
    def render_mask(self, value):
        raise DeprecationWarning(
            "render_mask is deprecated, see material.alpha_mode to control how the renderer should treat an object."
        )

    @property
    def geometry(self) -> Geometry | None:
        """The object's geometry, the data that defines (the shape of) this object."""
        return self._store.geometry

    @geometry.setter
    def geometry(self, geometry: Geometry | None):
        if not (geometry is None or isinstance(geometry, Geometry)):
            raise TypeError(
                f"WorldObject.geometry must be a Geometry object or None, not {geometry!r}"
            )
        self._store.geometry = geometry
        # Shadow-caster eligibility depends on geometry presence.
        self._refresh_caster_status()

    @property
    def material(self) -> Material | None:
        """The object's material, the data that defines the appearance of this object."""
        # In contrast to the geometry, the material is not stored on self._store,
        # because it should not be tracked, because pipeline-containers are unique
        # for each combi of (wobject, material, renderstate).
        return self._material

    @material.setter
    def material(self, material: Material | None) -> None:
        if not (material is None or isinstance(material, Material)):
            raise TypeError(
                f"WorldObject.geometry must be a Geometry object or None, not {material!r}"
            )
        self._material = material
        # Renderable eligibility depends on having a material.
        self._refresh_renderable_status()

    @property
    def cast_shadow(self) -> bool:
        """Whether this object casts shadows, i.e. whether it is rendered into
        a shadow map. Default False."""
        return self._cast_shadow  # does not affect any shaders

    @cast_shadow.setter
    def cast_shadow(self, value: bool) -> None:
        self._cast_shadow = bool(value)
        self._refresh_caster_status()

    @property
    def receive_shadow(self) -> bool:
        """Whether this object receives shadows. Default False."""
        return self._store.receive_shadow

    @receive_shadow.setter
    def receive_shadow(self, value: bool) -> None:
        self._store.receive_shadow = bool(value)

    @property
    def nonlinear_transform(self) -> str | None:
        """An optional nonlinear transform, expressed as WGSL shader code, applied to the raw vertex positions.

        The WGSL must define a function ``fn nonlinear_transform(pos: vec3f) -> vec3f { ... }``.
        Alternatively, the given WGSL can be a single expression (without ';')
        using ``pos`` (a vec3f) as an existing variable, in which case the code is
        wrapped in a function. E.g. ``vec3f(pos.x, pos.y*1.5, pos.z)``.

        Note that the transform is applied to the raw geometry's vertex
        positions of this object, and does not affect child objects.

        Note that the transform is not taken into account in the calculation of
        bounding boxes.
        """
        return self._store.nonlinear_transform

    @nonlinear_transform.setter
    def nonlinear_transform(self, value: str | None):
        if not (value is None or isinstance(value, str)):
            raise TypeError("nonlinear_transform must be str or None")
        if value is not None:
            value = value.strip() or None
        if value is not None and not any(x in value for x in ("fn n", "return ", ";")):
            value = f"fn nonlinear_transform(pos: vec3f) -> vec3f {{ return {value}; }}"
        self._store.nonlinear_transform = value

    @property
    def parent(self) -> WorldObject | None:
        """Object's parent in the scene graph (read-only).
        An object can have at most one parent.
        """
        if self._parent is None:
            return None
        else:
            return self._parent()

    @property
    def children(self) -> Tuple[WorldObject, ...]:
        """tuple of children of this object. (read-only)"""
        return tuple(self._children)

    def add(
        self,
        *objects: WorldObject,
        before: WorldObject | None = None,
        keep_world_matrix: bool = False,
    ) -> WorldObject:
        """Add child objects.

        Any number of objects may be added. Any current parent on an object
        passed in here will be removed, since an object can have at most one
        parent. If ``before`` argument is given, then the items are inserted
        before the given element.

        Parameters
        ----------
        *objects : WorldObject
            The world objects to add as children.
        before : WorldObject
            If not None, insert the objects before this child object.
        keep_world_matrix : bool
            If True, the child will keep it's world transform. It moves in the
            scene graph but will visually remain in the same place. If False,
            the child will keep it's parent transform.

        """
        for obj in objects:
            if obj.parent is not None:
                obj.parent.remove(obj, keep_world_matrix=keep_world_matrix)

            if before is not None:
                idx = self._children.index(before)
            else:
                idx = len(self._children)

            if keep_world_matrix:
                transform_matrix = obj.world.matrix

            obj._parent = weakref.ref(self)
            obj.world.parent = self.world
            self._children.insert(idx, obj)
            self.world.children.insert(idx, obj.world)

            if keep_world_matrix:
                obj.world.matrix = transform_matrix

            # Update the incremental scene-graph index. ``obj`` arrived as a
            # root (its prior parent, if any, was removed above), so its
            # ``_subtree_index`` already describes exactly the subtree to
            # graft in.
            obj._attach_subtree_under(self)

        return self

    def remove(self, *objects: WorldObject, keep_world_matrix: bool = False) -> None:
        """Removes object as child of this object. Any number of objects may be removed."""
        for obj in objects:
            try:
                self._children.remove(obj)
                self.world.children.remove(obj.world)
            except ValueError:
                logger.warning("Attempting to remove object that was not a child.")
                continue
            else:
                # Detach from the index BEFORE clearing the parent ref, so
                # we can still walk the ancestor chain.
                obj._detach_subtree_from(self)
                obj._reset_parent(keep_world_matrix=keep_world_matrix)

    def clear(self, *, keep_world_matrix: bool = False) -> None:
        """Removes all children."""

        for child in self._children:
            child._detach_subtree_from(self)
            child._reset_parent(keep_world_matrix=keep_world_matrix)

        self._children.clear()
        self.world.children.clear()

    def _reset_parent(self, *, keep_world_matrix=False):
        """Sets the parent to None.

        xref: https://github.com/pygfx/pygfx/pull/482#discussion_r1135670771
        """

        if keep_world_matrix:
            transform_matrix = self.world.matrix

        self._parent = None
        self.world.parent = None

        if keep_world_matrix:
            self.world.matrix = transform_matrix

    def traverse(
        self, callback: Callable[[WorldObject], Any], skip_invisible: bool = False
    ):
        """Executes the callback on this object and all descendants.

        If ``skip_invisible`` is given and True, objects whose
        ``visible`` property is False - and their children - are
        skipped. Note that modifying the scene graph inside the callback
        is discouraged.
        """

        for child in self.iter(skip_invisible=skip_invisible):
            callback(child)

    def iter(
        self,
        filter_fn: Callable[[WorldObject], bool] | None = None,
        skip_invisible: bool = False,
    ) -> Iterator[WorldObject]:
        """Create a generator that iterates over this objects and its children.
        If ``filter_fn`` is given, only objects for which it returns ``True``
        are included.
        """
        if skip_invisible and not self.visible:
            return

        if filter_fn is None:
            yield self
        elif filter_fn(self):
            yield self

        for child in self._children:
            yield from child.iter(filter_fn, skip_invisible)

    def _get_bounds_from_geometry(self):
        geometry = self.geometry
        if geometry is None:
            # Note: would be good to have a way to disable the geometry-from-bounds, e.g. when using an extreme nonlinear transform.
            # Once we have the new bounds logic a user could fo e.g. ``ob.set_local_bounds(None, None)``, see https://github.com/pygfx/pygfx/pull/1049
            self._bounds_geometry = None
        elif isinstance(positions_buf := getattr(geometry, "positions", None), Buffer):
            if self._bounds_geometry_rev == positions_buf.rev:
                return self._bounds_geometry
            self._bounds_geometry = None
            # Get array and check expected shape
            positions_array = positions_buf.data
            if (
                positions_array is not None
                and positions_array.ndim == 2
                and positions_array.shape[1] in (2, 3)
            ):
                self._bounds_geometry = Bounds.from_points(positions_array)
                self._bounds_geometry_rev = positions_buf.rev
        elif isinstance(grid_buf := getattr(geometry, "grid", None), Texture):
            if self._bounds_geometry_rev == grid_buf.rev:
                return self._bounds_geometry
            # account for multi-channel image data
            grid_shape = tuple(reversed(grid_buf.size[: grid_buf.dim]))
            # create aabb in index/data space
            aabb = np.array([np.zeros_like(grid_shape), grid_shape[::-1]], dtype="f8")
            # convert to local image space by aligning
            # center of voxel index (0, 0, 0) with origin (0, 0, 0)
            aabb -= 0.5
            # ensure coordinates are 3D
            # NOTE: important we do this last, we don't want to apply
            # the -0.5 offset to the z-coordinate of 2D images
            if aabb.shape[1] == 2:
                aabb = np.hstack([aabb, [[0], [0]]])
            self._bounds_geometry = Bounds(aabb, None)
            self._bounds_geometry_rev = grid_buf.rev
        else:
            self._bounds_geometry = None
        return self._bounds_geometry

    def get_geometry_bounding_box(self) -> np.ndarray | None:
        bounds = self._get_bounds_from_geometry()
        if bounds is not None:
            return bounds.aabb

    def get_bounding_box(self) -> np.ndarray | None:
        """Axis-aligned bounding box in local model space.

        Returns
        -------
        aabb : ndarray, [2, 3] or None
            An axis-aligned bounding box, or None when the object does
            not take up a particular space.
        """

        # Collect bounding boxes
        _aabbs = []
        for child in self._children:
            aabb = child.get_bounding_box()
            if aabb is not None:
                trafo = child.local.matrix
                _aabbs.append(la.aabb_transform(aabb, trafo))
        bounds = self._get_bounds_from_geometry()
        if bounds is not None:
            _aabbs.append(bounds.aabb)

        # Combine
        if _aabbs:
            aabbs = np.stack(_aabbs)
            final_aabb = np.empty((2, 3), dtype=float)
            final_aabb[0] = np.min(aabbs[:, 0, :], axis=0)
            final_aabb[1] = np.max(aabbs[:, 1, :], axis=0)
        else:
            final_aabb = None

        return final_aabb

    def get_bounding_sphere(self) -> np.ndarray | None:
        """Bounding Sphere in local model space.

        Returns
        -------
        bounding_shere : ndarray, [4] or None
            A sphere (x, y, z, radius), or None when the object does
            not take up a particular space.

        """
        # NOTE: this currently does not even use the sphere-data from the geometry!
        aabb = self.get_bounding_box()
        return None if aabb is None else la.aabb_to_sphere(aabb)

    def get_world_bounding_box(self) -> np.ndarray | None:
        """Axis aligned bounding box in world space.

        Returns
        -------
        aabb : ndarray, [2, 3] or None
            The transformed axis-aligned bounding box, or None when the
            object does not take up a particular space.

        """
        aabb = self.get_bounding_box()
        return None if aabb is None else la.aabb_transform(aabb, self.world.matrix)

    def get_world_bounding_sphere(self) -> np.ndarray | None:
        """Bounding Sphere in world space.

        Returns
        -------
        bounding_shere : ndarray, [4] or None
            A sphere (x, y, z, radius), or None when the object does
            not take up a particular space.

        """
        aabb = self.get_world_bounding_box()
        return None if aabb is None else la.aabb_to_sphere(aabb)

    def _wgpu_get_pick_info(self, pick_value) -> dict:
        # In most cases the material handles this.
        return self.material._wgpu_get_pick_info(pick_value)

    def look_at(self, target: WorldObject) -> None:
        """Orient the object so it looks at the given position.

        This sets the object's rotation such that its ``forward`` direction
        points towards ``target`` (given in world space). This rotation takes
        reference_up into account, i.e., the rotation is chosen in such a way that a
        camera looking ``forward`` follows the rotation of a human head looking
        around without tilting the head sideways.

        """

        self.world.forward = target - self.world.position
