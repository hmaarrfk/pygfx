"""Test that the shaders do not need more bind groups than WebGPU guarantees."""

import numpy as np
import pygfx as gfx
from pygfx.renderers.wgpu import get_shared
from pygfx.renderers.wgpu.shaders.meshshader import MeshPhysicalShader


def test_physical_material_bind_group_count():
    # WebGPU guarantees 4 bind groups (maxBindGroups). The last one is for
    # the render state (lights and shadows), which leaves 3 for the shader.
    tex = gfx.Texture(np.zeros((4, 4, 4), np.uint8), dim=2)
    material = gfx.MeshPhysicalMaterial(
        map=gfx.TextureMap(tex),
        roughness_map=gfx.TextureMap(tex),
        clearcoat=1.0,
        clearcoat_map=gfx.TextureMap(tex),
        iridescence=1.0,
        iridescence_map=gfx.TextureMap(tex),
    )
    mesh = gfx.Mesh(gfx.box_geometry(), material)
    scene = gfx.Scene()
    scene.add(mesh)

    shader = MeshPhysicalShader(mesh)
    bindings = shader.get_bindings_info(mesh, get_shared(), scene)

    assert len(bindings) <= 3
    assert set(bindings) <= {0, 1, 2}

    # The bindings of the physical material follow those of the standard one
    names = [b.name for b in bindings[2].values()]
    assert "t_roughness_map" in names
    assert "t_clearcoat_map" in names
    assert "t_iridescence_map" in names
    assert sorted(bindings[2]) == list(range(len(names)))


def _physical_mesh_with_all_maps():
    tex = gfx.Texture(np.zeros((4, 4, 4), np.uint8), dim=2)
    maps = [
        "map",
        "normal_map",
        "ao_map",
        "light_map",
        "emissive_map",
        "roughness_map",
        "metalness_map",
        "specular_intensity_map",
        "clearcoat_map",
        "clearcoat_roughness_map",
        "clearcoat_normal_map",
        "iridescence_map",
        "iridescence_thickness_map",
        "sheen_color_map",
        "sheen_roughness_map",
        "anisotropy_map",
    ]
    material = gfx.MeshPhysicalMaterial(
        clearcoat=1.0, iridescence=1.0, sheen=1.0, anisotropy=1.0
    )
    for name in maps:
        setattr(material, name, gfx.TextureMap(tex))
    mesh = gfx.Mesh(gfx.box_geometry(), material)
    return mesh, maps


def test_texture_map_transforms_share_one_uniform_buffer():
    # WebGPU guarantees 12 uniform buffers per shader stage. The renderer
    # adds up to 4 for the lights, so the shader of a material with all its
    # maps must stay well below that.
    mesh, maps = _physical_mesh_with_all_maps()
    scene = gfx.Scene()
    scene.add(mesh)

    shader = MeshPhysicalShader(mesh)
    bindings = shader.get_bindings_info(mesh, get_shared(), scene)

    uniforms = [
        b.name
        for group in bindings.values()
        for b in group.values()
        if b.type == "buffer/uniform"
    ]
    assert sorted(uniforms) == [
        "u_map_transforms",
        "u_material",
        "u_stdinfo",
        "u_wobject",
    ]
    buffer = next(
        b.resource for b in bindings[0].values() if b.name == "u_map_transforms"
    )
    fields = {n for n in buffer.data.dtype.names if not n.startswith("__")}
    assert fields == set(maps)


def test_texture_map_transform_reaches_the_shared_buffer():
    mesh, _ = _physical_mesh_with_all_maps()
    scene = gfx.Scene()
    scene.add(mesh)

    shader = MeshPhysicalShader(mesh)
    bindings = shader.get_bindings_info(mesh, get_shared(), scene)
    buffer = next(
        b.resource for b in bindings[0].values() if b.name == "u_map_transforms"
    )

    # The transform at the time the bindings were made
    assert np.all(buffer.data["normal_map"][:, :3] == np.eye(3))

    # A later change of the map is copied
    mesh.material.normal_map.update_matrix_components(offset=(0.25, 0.5))
    assert np.all(
        buffer.data["normal_map"]
        == mesh.material.normal_map.uniform_buffer.data["transform"]
    )
    assert buffer.data["normal_map"][2, 0] == 0.25
    assert buffer.data["normal_map"][2, 1] == 0.5
    assert np.all(buffer.data["map"][:, :3] == np.eye(3))
