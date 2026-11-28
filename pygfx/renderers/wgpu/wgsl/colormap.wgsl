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
// A float32 colormap on a device without the 'float32-filterable' feature: it
// cannot be sampled with a filtering sampler, so read the texels with
// textureLoad, and wrap and filter as the sampler would.

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
    $$ set ivec = 'i32'
    $$ set fvec = 'f32'
$$ elif colormap_dim == '2d'
    $$ set ivec = 'vec2<i32>'
    $$ set fvec = 'vec2<f32>'
$$ else
    $$ set ivec = 'vec3<i32>'
    $$ set fvec = 'vec3<f32>'
$$ endif

fn colormap_load(i: {{ ivec }}, n: {{ ivec }}) -> vec4<f32> {
    $$ if colormap_dim == '1d'
    let wrapped = colormap_wrap_x(i, n);
    $$ elif colormap_dim == '2d'
    let wrapped = vec2<i32>(colormap_wrap_x(i.x, n.x), colormap_wrap_y(i.y, n.y));
    $$ else
    let wrapped = vec3<i32>(
        colormap_wrap_x(i.x, n.x), colormap_wrap_y(i.y, n.y), colormap_wrap_z(i.z, n.z)
    );
    $$ endif
    return textureLoad(t_colormap, wrapped, 0);
}

fn colormap_nearest(uv: {{ fvec }}) -> vec4<f32> {
    let n = {{ ivec }}(textureDimensions(t_colormap));
    return colormap_load({{ ivec }}(floor(uv * {{ fvec }}(n))), n);
}

fn colormap_linear(uv: {{ fvec }}) -> vec4<f32> {
    // Weigh the 2, 4 or 8 texels around uv; texel centers are at +0.5
    let n = {{ ivec }}(textureDimensions(t_colormap));
    let pos = uv * {{ fvec }}(n) - 0.5;
    let i0 = {{ ivec }}(floor(pos));
    let t = fract(pos);
    var color = vec4<f32>(0.0);
    $$ for dz in ([0, 1] if colormap_dim == '3d' else [0])
    $$ for dy in ([0, 1] if colormap_dim != '1d' else [0])
    $$ for dx in [0, 1]
        $$ if colormap_dim == '1d'
    color += {{ 't' if dx else '(1.0 - t)' }} * colormap_load(i0 + {{ dx }}, n);
        $$ else
            $$ set wx = 't.x' if dx else '(1.0 - t.x)'
            $$ set wy = 't.y' if dy else '(1.0 - t.y)'
            $$ set wz = 't.z' if dz else '(1.0 - t.z)'
            $$ if colormap_dim == '2d'
    color += {{ wx }} * {{ wy }} * colormap_load(i0 + vec2<i32>({{ dx }}, {{ dy }}), n);
            $$ else
    color += {{ wx }} * {{ wy }} * {{ wz }} * colormap_load(i0 + vec3<i32>({{ dx }}, {{ dy }}, {{ dz }}), n);
            $$ endif
        $$ endif
    $$ endfor
    $$ endfor
    $$ endfor
    return color;
}

fn sample_colormap_f32_load(uv: {{ fvec }}) -> vec4<f32> {
    $$ if colormap_mag_filter == colormap_min_filter
    return colormap_{{ colormap_mag_filter }}(uv);
    $$ else
    // As the sampler: the min filter where the colormap is minified, i.e.
    // where a pixel covers more than a texel (level of detail > 0)
    let texels = uv * {{ fvec }}(textureDimensions(t_colormap));
    let lod = log2(max(length(dpdx(texels)), length(dpdy(texels))));
    let magnified = colormap_{{ colormap_mag_filter }}(uv);
    let minified = colormap_{{ colormap_min_filter }}(uv);
    return select(magnified, minified, lod > 0.0);
    $$ endif
}
$$ endif
