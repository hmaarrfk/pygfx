// # Line shader
//
// ## References:
//
// * https://almarklein.org/line_rendering.html -> for a high level understaning of what this shader does.
// * https://almarklein.org/triangletricks.html -> for an explanation of some of the referred tricks.
// * https://jcgt.org/published/0002/02/08/paper.pdf -> Nicolas Rougier's paper that was a great inspiration.
//
// ## Summary
//
// The vertex shader uses VertexId and storage buffers instead of vertex buffers.
// It creates 6 vertices for each point on the line, and a triangle-strip topology.
// That gives 6 faces, of which 4 are used: 2 for the rectangular segment to the
// previous join, and two for the join or cap(s). In each configuration 2 faces
// are dropped.
//
// The resulting shapes are made up of triangles. In the fragment shader we discard
// fragments depending on join and cap shapes, and we use aa for crisp edges.
//
// ## Definitions
//
// - node: the positions that define the line. In other contexts these
//   may be called vertices or points.
// - vertex: the "virtual vertices" generated in the vertex shader,
//   in order to create a thick line with nice joins and caps.
// - segment: the rectangular piece of the line between two nodes.
// - join: the piece of the line to connect two segments.
// - broken join: joins with too sharp corners are rendered as two
//   separate segments with caps.
// - cap: the beginning/end of the line and dashes. It typically extends
//   a bit beyond the node (or dash end). There are multiple cap shapes.
// - stroke: when dashing is enabled, the stoke represents the "on" piece.
//   This is the visible piece to which caps are added. Can go over a
//   join, i.e. is not always straight. The gap is the off-piece.
//
// ## Basic algorithm
//
// - We read the positions of three nodes, the previous, current, and next.
// - These are converted to logical pixel screen space.
// - We define six coordinate vectors which represent the (virtual) vertices.
//   The first two end the previous segment, the last two start the next
//   segment, the two in the middle help define the join/caps.
// - To obtain the positions, the above coordinates are rotated, added to the
//   node positions, and then converted to ndc.
// - To some degree these calculations are done for all 6 vertices, and the
//   one corresponding to the current vertex_index is selected.
//
//            /  o     node 3
//           /  /  /
//        5 /  /  /
//   - - - 1  /  /     segment-vertices 1, 2, 5, 6
//   o-------o  6      the vertices 3 and 4 are both in the outer corner.
//   - - - 2 - 34
//                node 2
//  node 1
//


{# Includes #}
{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif



// -------------------- functions --------------------


fn is_finite_vec(v:vec3<f32>) -> bool {
    return is_finite(v.x) && is_finite(v.y) && is_finite(v.z);
}

// Naga has removed isNan checks, because backends may be using fast-math, in
// which case nan is assumed not to happen, and isNan would always be false. If
// we assume that some nan mechanics still work, we can still detect it.
// See https://github.com/pygfx/wgpu-py/blob/main/tests/test_not_finite.py
// NOTE: Other option is loading as i32, checking bitmask, and then bitcasting to float.
//       -> This might be faster, but we need a benchmark to make sure.
fn is_nan(v:f32) -> bool {
    return min(v, 1.0) == 1.0 && max(v, -1.0) == -1.0;
}
fn is_inf(v:f32) -> bool {
    return v != 0.0 && v * 2.0 == v;
}
fn is_finite(v:f32) -> bool {
    return !is_nan(v) && !is_inf(v);
}

fn rotate_vec2(v:vec2<f32>, angle:f32) -> vec2<f32> {
    return vec2<f32>(cos(angle) * v.x - sin(angle) * v.y, sin(angle) * v.x + cos(angle) * v.y);
}


// -------------------- vertex shader --------------------


struct VertexInput {
    @builtin(vertex_index) index : u32,
};

@vertex
fn vs_main(in: VertexInput) -> Varyings {

    let i0 = i32(in.index);

    let raw_pos = load_s_positions(i0);
    let wpos = u_wobject.world_transform * vec4<f32>(raw_pos.xyz, 1.0);
    let npos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * wpos;

    var varyings: Varyings;
    varyings.position = vec4<f32>(npos);
    varyings.world_pos = vec3<f32>(ndc_to_world_pos(npos));

    // per-vertex or per-face coloring
    $$ if color_mode == 'vertex'
        let color_index = i0;
        $$ if color_buffer_channels == 1
            let cvalue = load_s_colors(color_index);
            varyings.color = vec4<f32>(cvalue, cvalue, cvalue, 1.0);
        $$ elif color_buffer_channels == 2
            let cvalue = load_s_colors(color_index);
            varyings.color = vec4<f32>(cvalue.r, cvalue.r, cvalue.r, cvalue.g);
        $$ elif color_buffer_channels == 3
            varyings.color = vec4<f32>(load_s_colors(color_index), 1.0);
        $$ elif color_buffer_channels == 4
            varyings.color = vec4<f32>(load_s_colors(color_index));
        $$ endif
    $$ endif

    // Set texture coords
    let tex_coord_index = i0;
    $$ if colormap_dim == '1d'
    varyings.texcoord = f32(load_s_texcoords(tex_coord_index));
    $$ elif colormap_dim == '2d'
    varyings.texcoord = vec2<f32>(load_s_texcoords(tex_coord_index));
    $$ elif colormap_dim == '3d'
    varyings.texcoord = vec3<f32>(load_s_texcoords(tex_coord_index));
    $$ endif

    return varyings;
}


// --------------------  fragment shader --------------------


$$ if dashing
// Constant to help compiler create fixed-size arrays and loops.
const dash_count = {{dash_count}};
$$ endif


@fragment
fn fs_main(varyings: Varyings, @builtin(front_facing) is_front: bool) -> FragmentOutput {

    // Get the half-thickness in physical coordinates. This is the reference thickness.
    // If aa is used, the line is actually a bit thicker, leaving space to do aa.
    let half_thickness_p = 0.5 * varyings.thickness_pw / varyings.w;

    // Discard invalid faces. These are faces for which *all* 3 verts are set to zero. (trick 5b)
    if (varyings.valid_if_nonzero == 0.0) {
        discard;
    }

    // Determine whether we are at a join (i.e. an unbroken corner).
    // These are faces for which *any* vert is nonzero. (trick 5a)
    $$ if line_type == 'quickline'
    let is_join = false;  // hard-coded to false. I'm assuming the Naga optimizer will eliminate dead code.
    $$ else
    let is_join = varyings.join_coord != 0.0;
    $$ endif

    // Obtain the join coordinates. It comes in two flavours, linear and fan-shaped,
    // which each serve a different purpose. These represent trick 3 and 4, respectively.
    //
    // join_coord_lin      join_coord_fan
    //
    // | | | | |-          | | | / / ╱
    // | | | |- -          | | | / ╱ ⟋
    // | | |- - -          | | | ╱ ⟋ ⟋
    //      - - -                - - -
    //      - - -                - - -
    //
    let join_coord_lin = varyings.join_coord;
    let join_coord_fan = join_coord_lin / varyings.is_outer_corner;

    // Get the line coord in physical pixels.
    // For joins, the outer vertices are inset, and we need to take that into account,
    // so that the origin is at the node (i.e. the pivot point).
    var segment_coord_p = varyings.segment_coord_pw / varyings.w;
    if (is_join) {
        let dist_from_segment = abs(join_coord_lin);
        let a = segment_coord_p.x / dist_from_segment;
        segment_coord_p = vec2<f32>(max(0.0, dist_from_segment - 0.5) * a, segment_coord_p.y);
    }

    // Calculate the distance to the stroke's edge. Negative means inside, positive means outside. Just like SDF.
    let dist_to_center_p = length(segment_coord_p);
    var dist_to_stroke_p = length(segment_coord_p) - half_thickness_p;

    $$ if dashing

        // Calculate the cumulative distance along the line. We need a continuous value to parametrize
        // the dash (and its cap). Going around the corner, it will compress on the inside, and expand
        // on the outer side, deforming dashes as they move around the corner, appropriately.
        var cumdist_continuous : f32;
        var cumdist_per_pixel : f32;
        if (is_join) {
            // First calculate the cumdist at the edge where segment and join meet.
            // Note that cumdist_vertex == cumdist_node at the outer-corner-vertex.
            let cumdist_segment = varyings.cumdist_node - (varyings.cumdist_node - varyings.cumdist_vertex) / (1.0 - abs(join_coord_lin));
            // Calculate the continous cumdist, by interpolating using join_coord_fan
            cumdist_continuous = mix(cumdist_segment, varyings.cumdist_node, abs(join_coord_fan));
        } else {
            // In a segment everything is straight.
            cumdist_continuous = varyings.cumdist_vertex;
        }
        $$ if thickness_space == 'screen'
            cumdist_continuous = cumdist_continuous / varyings.w;
            cumdist_per_pixel = varyings.cumdist_per_pixel;
        $$ else
            cumdist_per_pixel = varyings.cumdist_per_pixel * varyings.w;
        $$ endif

        // Define dash pattern, scale with (uniform) thickness.
        // Note how the pattern is templated (triggering recompilation when it changes), wheras the thickness is a uniform.
        var stroke_sizes = array<f32,dash_count>{{dash_pattern[::2]}};
        var gap_sizes = array<f32,dash_count>{{dash_pattern[1::2]}};
        for (var i=0; i<dash_count; i+=1) {
            stroke_sizes[i] = stroke_sizes[i];
            gap_sizes[i] = gap_sizes[i];
        }

        // Calculate the total dash size, and the size of the last gap. The dash_count is a const
        var dash_size = 0.0;
        var last_gap = 0.0;
        for (var i=0; i<dash_count; i+=1) {
            dash_size += stroke_sizes[i];
            last_gap = gap_sizes[i];
            dash_size += last_gap;
        }

        // Calculate dash_progress, a number 0..dash_size, indicating the fase of the dash.
        // Except that we shift it, so that half of the final gap gets in front (as a negative number).
        let cumdist_corrected = cumdist_continuous / u_material.thickness + u_material.dash_offset % dash_size;
        let dash_progress = (cumdist_corrected + 0.5 * last_gap) % dash_size - 0.5 * last_gap;

        // Its looks a bit like this. Now we select the nearest stroke, and calculate the
        // distance to the beginning and end of that stroke.
        //
        //  -0.5*last_gap      0                                              dash_size-0.5*last_gap     dash_size
        //     |               |                                                          |               |
        //     |---------------|XXXXXXXXXXXXXXX|-------|-------|XXXXXXXXXX|---------------|...............|
        //     |               |               |       |
        //  gap_begin    stroke_begin    stroke_end    gap_end (i.e. begin of next stroke)
        //
        var dist_to_begin = 0.0;
        var dist_to_end = 0.0;
        var gap_begin = -0.5 * last_gap;
        var stroke_begin = 0.0;
        for (var i=0; i<dash_count; i+=1) {
            let half_gap_size = 0.5 * gap_sizes[i];
            let stroke_end = stroke_begin + stroke_sizes[i];
            let gap_end = stroke_end + half_gap_size;
            if (dash_progress >= gap_begin && dash_progress <= gap_end) {
                dist_to_begin = stroke_begin - dash_progress;
                dist_to_end = dash_progress - stroke_end;
                break;
            }
            // Next
            gap_begin = gap_end;
            stroke_begin = gap_end + half_gap_size;
        }

        // The distance to the dash's stoke is now easy to calculate.
        // Note that it's also possible to calculate dist_to_dash without dist_to_begin
        // and dist_to_end, but we need these for the trick in broken joins below.
        let dist_to_dash = max(0.0, max(dist_to_begin, dist_to_end));

        // Convert to (physical) pixel units
        let dashdist_to_physical = u_material.thickness / cumdist_per_pixel;
        let dist_to_begin_p = dist_to_begin * dashdist_to_physical;
        let dist_to_end_p = dist_to_end * dashdist_to_physical;
        let dist_to_dash_p = dist_to_dash * dashdist_to_physical;

        // At broken joins there is overlapping cumdist in both caps. The code below
        // avoids (not 100% prevents) the begin or end of a cap to be drawn twice.
        // The logic is basically: if we are in the cap (of a broken join), and if the
        // current dash would not be drawn in the segment attached to this cap, we
        // don't draw it here either.
        let is_broken_join = !is_join && segment_coord_p.x != 0.0;
        if (is_broken_join){
            let dist_at_segment_p = select(dist_to_end_p, dist_to_begin_p, segment_coord_p.x > 0.0) + abs(segment_coord_p.x);
            if (dist_at_segment_p > half_thickness_p) {
                discard;
            }
        }

        // The vector to the stoke (at the line-center)
        var yy = length(segment_coord_p.y);
        if (abs(join_coord_lin) > 0.5) {
            yy = length(segment_coord_p);  // smoother dash-turns
        }
        let vec_to_dash_p = vec2<f32>(dist_to_dash_p, yy);

        // Apply cap
        // let dist_to_stroke_dash_p = vec_to_dash_p.x;  // Butt caps
        let dist_to_stroke_dash_p = length(vec_to_dash_p) - half_thickness_p; // Round caps

        // Update dist_to_stroke_p with dash info
        dist_to_stroke_p = max(dist_to_stroke_p, dist_to_stroke_dash_p);

        // end dashing
    $$ endif

    $$ if debug
        // In debug-mode, use barycentric coords to draw the edges of the faces.
        dist_to_stroke_p = -1.0;
        if (min(varyings.bary.x, min(varyings.bary.y, varyings.bary.z)) > 0.1) {
            dist_to_stroke_p = 1.0;
        }
    $$ endif

    $$ if line_type == 'arrow'
        // Arrow shape. Use pick_zigzag, because it has exactly what we need.
        let arrow_head_factor = varyings.pick_zigzag;
        let arrow_tail_factor = varyings.pick_zigzag * 3.0 - 2.0;
        dist_to_stroke_p = max(
            abs(segment_coord_p.y) - half_thickness_p * arrow_head_factor,
            half_thickness_p * arrow_tail_factor- abs(segment_coord_p.y)
        );
        // Ignore caps
        dist_to_stroke_p = select(dist_to_stroke_p, 9999999.0, segment_coord_p.x != 0.0);
    $$ endif

    // Anti-aliasing.
    // By default, the renderer uses SSAA (super-sampling), but if we apply AA for the edges
    // here this will help the end result. Because this produces semitransparent fragments,
    // it relies on a good blend method, and the object gets drawn twice.
    var alpha: f32 = 1.0;
    $$ if aa
        if (half_thickness_p > 0.5) {
            alpha = clamp(0.5 - dist_to_stroke_p, 0.0, 1.0);
        } else {
            // Thin lines, factor based on dist_to_center_p, scaled by the size (with a max)
            alpha = (1.0 - dist_to_center_p) * max(0.01, half_thickness_p * 2.0);
        }
        alpha = sqrt(alpha);  // this prevents aa lines from looking thinner
        if (alpha <= 0.0) { discard; }
    $$ else
        if (dist_to_stroke_p > 0.0) { discard; }
    $$ endif

    // Determine srgb color
    $$ if color_mode == 'vertex'
        var color = varyings.color_vert;
        if (is_join) {
            let color_segment = varyings.color_node - (varyings.color_node - varyings.color_vert) / (1.0 - abs(join_coord_lin));
            color = mix(color_segment, varyings.color_node, abs(join_coord_fan));
        }
    $$ elif color_mode == 'face'
        let color = varyings.color_vert;
    $$ elif color_mode == 'vertex_map'
        var texcoord = varyings.texcoord_vert;
        if (is_join) {
            let texcoord_segment = varyings.texcoord_node - (varyings.texcoord_node - varyings.texcoord_vert) / (1.0 - abs(join_coord_lin));
            texcoord = mix(texcoord_segment, varyings.texcoord_node, abs(join_coord_fan));
        }
        let color = sample_colormap(texcoord);
    $$ elif color_mode == 'face_map'
        let color = sample_colormap(varyings.texcoord_vert);
    $$ else
        let color = u_material.color;
    $$ endif
    var physical_color = srgb2physical(color.rgb);

    $$ if false
        // Alternative debug options during dev.
        physical_color = vec3<f32>(abs(dist_to_dash_p) / 20.0, 0.0, 0.0);
    $$ endif

    // Determine final rgba value
    let opacity = min(1.0, color.a) * alpha * u_material.opacity;
    let out_color = vec4<f32>(physical_color, opacity);

    // Wrap up
    apply_clipping_planes(varyings.world_pos);
    var out = get_fragment_output(varyings.position.z, out_color);

    // Set picking info.
    $$ if write_pick
    // The wobject-id must be 20 bits. In total it must not exceed 64 bits.
    // The pick_idx is int-truncated, so going from a to b, it still has the value of a
    // even right up to b. The pick_zigzag alternates between 0 (even indices) and 1 (odd indices).
    // Here we decode that. The result is that we can support vertex indices of ~32 bits if we want.
    let is_even = varyings.pick_idx % 2u == 0u;
    var coord = select(varyings.pick_zigzag, 1.0 - varyings.pick_zigzag, is_even);
    coord = select(coord, coord - 1.0, coord > 0.5);
    let idx = varyings.pick_idx + select(0u, 1u, coord < 0.0);
    out.pick = (
        pick_pack(u32(u_wobject.id), 20) +
        pick_pack(u32(idx), 26) +
        pick_pack(u32(coord * 100000.0 + 100000.0), 18)
    );
    $$ endif

    // The outer edges with lower alpha for aa are pushed a bit back to avoid artifacts.
    // This is only necessary for blend method "ordered1"
    //out.depth = varyings.position.z + 0.0001 * (0.8 - min(0.8, alpha));

    return out;
}
