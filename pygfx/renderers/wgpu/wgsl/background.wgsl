// Main shader for rendering backgrounds.

{# Includes #}
{$ include 'pygfx.std.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    // Define positions at the four corners of the viewport, at the largest depth
    var positions = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0),
    );
    // Get the current ndc position, and the "virtual" ndc pos. Usually the same except when camera.set_view_offset() is used.
    let pos = positions[i32(in.index)];
    let virtual_pos = pos * u_stdinfo.ndc_offset.xy + u_stdinfo.ndc_offset.zw;

    $$ if texture_dim == "cube"
        // Get view direction. Since the texcoord is based on the view direction, the virtual_pos is not needed (it is included in the camera projection matrix)
        let ndc_pos1 = vec4<f32>(pos, 0.9999999, 1.0);
        let ndc_pos2 = vec4<f32>(pos, 1.1000000, 1.0);
        let wpos1 = ndc_to_world_pos(ndc_pos1);
        let wpos2 = ndc_to_world_pos(ndc_pos2);
        let d = wpos1.xyz - wpos2.xyz;  // view direction in world space.

        // Transform the view direction to background object space, it's also the cubemap texcoord, so we can use it to sample the cubemap.
        var texcoord = vec3<f32>((u_wobject.world_transform_inv * vec4<f32>(d, 0.0)).xyz);

        // By convention, cube maps are specified in a coordinate system in which positive-x is to the right when looking at the positive-z axis,
        // that is, it using a left-handed coordinate system.
        // See https://www.khronos.org/opengl/wiki/Cubemap_Texture#Cubemap_coordinate_system
        // Since pygfx uses a right-handed coordinate system, we need to flip the x coordinate when sampling from the cubemap.
        texcoord = vec3<f32>(-texcoord.x, texcoord.y, texcoord.z);
    $$ else
        // Store positions and the view direction in the world
        let texcoord = vec3<f32>(virtual_pos * 0.5 + 0.5, 0.0);
    $$ endif

    var varyings: Varyings;
    varyings.position = vec4<f32>(pos, 0.9999999, 1.0);
    varyings.texcoord = vec3<f32>(texcoord);
    //varyings.world_pos = vec3<f32>(ndc_to_world_pos(vec4<f32>(virtual_pos, 0.9999999, 1.0);));
    return varyings;
}


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    var final_color : vec4<f32>;
    $$ if texture_dim
        $$ if texture_dim == '2d'
            let color = textureSample(r_tex, r_sampler, varyings.texcoord.xy);
        $$ elif texture_dim == 'cube'
            let color = textureSample(r_tex, r_sampler, varyings.texcoord.xyz);
        $$ endif
        $$ if texture_nchannels == 1
            final_color = vec4<f32>(color.rrr, 1.0);
        $$ elif texture_nchannels == 2
            final_color = vec4<f32>(color.rrr, color.g);
        $$ else
            final_color = color;
        $$ endif
    $$ else
        let f = varyings.texcoord.xy;
        final_color = (
            u_material.color_bottom_left * (1.0 - f.x) * (1.0 - f.y)
            + u_material.color_bottom_right * f.x * (1.0 - f.y)
            + u_material.color_top_left * (1.0 - f.x) * f.y
            + u_material.color_top_right * f.x * f.y
        );
    $$ endif

    // Make physical color with combined alpha
    let physical_color = srgb2physical(final_color.rgb);
    let opacity = final_color.a * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    // We can apply clipping planes, but maybe a background should not be clipped?
    // apply_clipping_planes(in.world_pos);
    // Same for alpha test
    // do_alpha_test(opacity);

    // Force opaque. We may undo this when the ordered2 mechanics are removed.
    var out: FragmentOutput;
    out.color = vec4<f32>(out_color.rgb, 1.0);
    $$ if write_pick
        // We omit any extra information in the pick info
        // While we figure out exactly how best to return it.
        // Much of this information may soon be redundant.
        // https://github.com/pygfx/pygfx/pull/700
        out.pick = pick_pack(u32(u_wobject.global_id), 20);
    $$ endif
    return out;
}