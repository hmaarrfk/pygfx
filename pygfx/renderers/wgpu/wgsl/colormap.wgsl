// Colormap support

fn sample_colormap(texcoord: {{ colormap_coord_type }}) -> vec4<f32> {
    return sample_colormap_nomaprange(
        (texcoord - u_material.maprange[0]) / (u_material.maprange[1] - u_material.maprange[0])
    );
}

fn sample_colormap_nomaprange(texcoord: {{ colormap_coord_type }}) -> vec4<f32> {

    $$ if use_colormap is defined and use_colormap
        // apply the colormap transform
        $$ if colormap_coord_type == "f32"
            let map_uv = (u_colormap.transform * vec3<f32>(texcoord, 1.0, 1.0)).x;
        $$ elif colormap_coord_type == "vec2<f32>"
            let map_uv = (u_colormap.transform * vec3<f32>(texcoord, 1.0)).xy;
        $$ elif colormap_coord_type == "vec3<f32>"
            let transformed_uv = (u_colormap.transform * vec3<f32>(texcoord.xy, 1.0));
            let map_uv = vec3f(transformed_uv.xy, texcoord.z);
        $$ endif
    $$ else
        let map_uv = texcoord;
    $$ endif

    // Sample in the colormap. We get a vec4 color, but not all channels may be used.
    $$ if not colormap_dim
        let color_value = vec4<f32>(0.0);
    $$ elif colormap_format == 'f32'
        let color_value = textureSample(t_colormap, s_colormap, map_uv);
    $$ elif colormap_format == 'f32-load'
        let color_value = sample_colormap_f32_load(map_uv);
    $$ else
        $$ if colormap_dim == '1d'
        let texcoords_dim = f32(textureDimensions(t_colormap));
        let ti = i32(map_uv * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '2d'
        let texcoords_dim = vec2<f32>(textureDimensions(t_colormap));
        let ti = vec2<i32>(map_uv * texcoords_dim % texcoords_dim);
        $$ elif colormap_dim == '3d'
        let texcoords_dim = vec3<f32>(textureDimensions(t_colormap));
        let ti = vec3<i32>(map_uv * texcoords_dim % texcoords_dim);
        $$ endif
        let color_value = vec4<f32>(textureLoad(t_colormap, ti, 0));
    $$ endif

    // Depending on the number of channels we makeGfxTextureView grayscale, rgb, etc.
    $$ if colormap_nchannels == 1
        let color = vec4<f32>(color_value.rrr, 1.0);
    $$ elif colormap_nchannels == 2
        let color = vec4<f32>(color_value.rrr, color_value.g);
    $$ elif colormap_nchannels == 3
        let color = vec4<f32>(color_value.rgb, 1.0);
    $$ else
        let color = vec4<f32>(color_value.rgb, color_value.a);
    $$ endif
    return color;
}


$$ if colormap_format == 'f32-load'
// A float32 colormap on a device without the 'float32-filterable' feature: the
// texture cannot be sampled with a filtering sampler, so read the texels with
// textureLoad, and wrap and interpolate as the sampler would.

$$ for axis, wrap in [('x', colormap_wrap[0]), ('y', colormap_wrap[1]), ('z', colormap_wrap[2])]
fn colormap_wrap_{{ axis }}(i: i32, n: i32) -> i32 {
    $$ if wrap == 'repeat'
    return ((i % n) + n) % n;
    $$ elif wrap == 'mirror-repeat'
    let p = 2 * n;
    let m = ((i % p) + p) % p;
    return select(m, p - 1 - m, m >= n);
    $$ else
    return clamp(i, 0, n - 1);  // 'clamp-to-edge'
    $$ endif
}
$$ endfor

$$ if colormap_dim == '1d'
fn colormap_load(i: i32, n: i32) -> vec4<f32> {
    return textureLoad(t_colormap, colormap_wrap_x(i, n), 0);
}
fn sample_colormap_f32_load(uv: f32) -> vec4<f32> {
    let n = i32(textureDimensions(t_colormap));
    $$ if colormap_filter == 'linear'
    let pos = uv * f32(n) - 0.5;
    let i0 = i32(floor(pos));
    let t = pos - floor(pos);
    return mix(colormap_load(i0, n), colormap_load(i0 + 1, n), t);
    $$ else
    return colormap_load(i32(floor(uv * f32(n))), n);
    $$ endif
}
$$ elif colormap_dim == '2d'
fn colormap_load(i: vec2<i32>, n: vec2<i32>) -> vec4<f32> {
    return textureLoad(t_colormap, vec2<i32>(colormap_wrap_x(i.x, n.x), colormap_wrap_y(i.y, n.y)), 0);
}
fn sample_colormap_f32_load(uv: vec2<f32>) -> vec4<f32> {
    let n = vec2<i32>(textureDimensions(t_colormap));
    $$ if colormap_filter == 'linear'
    let pos = uv * vec2<f32>(n) - 0.5;
    let i0 = vec2<i32>(floor(pos));
    let t = pos - floor(pos);
    let c0 = mix(colormap_load(i0, n), colormap_load(i0 + vec2<i32>(1, 0), n), t.x);
    let c1 = mix(colormap_load(i0 + vec2<i32>(0, 1), n), colormap_load(i0 + vec2<i32>(1, 1), n), t.x);
    return mix(c0, c1, t.y);
    $$ else
    return colormap_load(vec2<i32>(floor(uv * vec2<f32>(n))), n);
    $$ endif
}
$$ elif colormap_dim == '3d'
fn colormap_load(i: vec3<i32>, n: vec3<i32>) -> vec4<f32> {
    return textureLoad(t_colormap, vec3<i32>(colormap_wrap_x(i.x, n.x), colormap_wrap_y(i.y, n.y), colormap_wrap_z(i.z, n.z)), 0);
}
fn sample_colormap_f32_load(uv: vec3<f32>) -> vec4<f32> {
    let n = vec3<i32>(textureDimensions(t_colormap));
    $$ if colormap_filter == 'linear'
    let pos = uv * vec3<f32>(n) - 0.5;
    let i0 = vec3<i32>(floor(pos));
    let t = pos - floor(pos);
    var c: array<vec4<f32>, 2>;
    for (var z = 0; z < 2; z++) {
        let c0 = mix(colormap_load(i0 + vec3<i32>(0, 0, z), n), colormap_load(i0 + vec3<i32>(1, 0, z), n), t.x);
        let c1 = mix(colormap_load(i0 + vec3<i32>(0, 1, z), n), colormap_load(i0 + vec3<i32>(1, 1, z), n), t.x);
        c[z] = mix(c0, c1, t.y);
    }
    return mix(c[0], c[1], t.z);
    $$ else
    return colormap_load(vec3<i32>(floor(uv * vec3<f32>(n))), n);
    $$ endif
}
$$ endif
$$ endif
