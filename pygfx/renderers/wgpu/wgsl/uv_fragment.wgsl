$$ if map_uv is defined
    let map_uv = (u_map_transforms.map * vec3<f32>(varyings.texcoord{{map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_normal_map is defined
    let normal_map_uv = (u_map_transforms.normal_map * vec3<f32>(varyings.texcoord{{normal_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_specular_map is defined
    let specular_map_uv = (u_map_transforms.specular_map * vec3<f32>(varyings.texcoord{{specular_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_specular_intensity_map is defined
    let specular_intensity_map_uv = (u_map_transforms.specular_intensity_map * vec3<f32>(varyings.texcoord{{specular_intensity_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_ao_map is defined
    let ao_map_uv = (u_map_transforms.ao_map * vec3<f32>(varyings.texcoord{{ao_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_light_map is defined
    let light_map_uv = (u_map_transforms.light_map * vec3<f32>(varyings.texcoord{{light_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_emissive_map is defined
    let emissive_map_uv = (u_map_transforms.emissive_map * vec3<f32>(varyings.texcoord{{emissive_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_metalness_map is defined
    let metalness_map_uv = (u_map_transforms.metalness_map * vec3<f32>(varyings.texcoord{{metalness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_roughness_map is defined
    let roughness_map_uv = (u_map_transforms.roughness_map * vec3<f32>(varyings.texcoord{{roughness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_map is defined
    let clearcoat_map_uv = (u_map_transforms.clearcoat_map * vec3<f32>(varyings.texcoord{{clearcoat_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_normal_map is defined
    let clearcoat_normal_map_uv = (u_map_transforms.clearcoat_normal_map * vec3<f32>(varyings.texcoord{{clearcoat_normal_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_clearcoat_roughness_map is defined
    let clearcoat_roughness_map_uv = (u_map_transforms.clearcoat_roughness_map * vec3<f32>(varyings.texcoord{{clearcoat_roughness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_iridescence_map is defined
    let iridescence_map_uv = (u_map_transforms.iridescence_map * vec3<f32>(varyings.texcoord{{iridescence_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_iridescence_thickness_map is defined
    let iridescence_thickness_map_uv = (u_map_transforms.iridescence_thickness_map * vec3<f32>(varyings.texcoord{{iridescence_thickness_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_anisotropy_map is defined
    let anisotropy_map_uv = (u_map_transforms.anisotropy_map * vec3<f32>(varyings.texcoord{{anisotropy_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_sheen_color_map is defined
    let sheen_color_map_uv = (u_map_transforms.sheen_color_map * vec3<f32>(varyings.texcoord{{sheen_color_map_uv or ''}}, 1.0)).xy;
$$ endif

$$ if use_sheen_roughness_map is defined
    let sheen_roughness_map_uv = (u_map_transforms.sheen_roughness_map * vec3<f32>(varyings.texcoord{{sheen_roughness_map_uv or ''}}, 1.0)).xy;
$$ endif

