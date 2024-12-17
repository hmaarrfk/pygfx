"""
Points with different markers
=============================

* All available marker shapes are shown.
* Shows red, green and blue faces. Then a semi-transparent face, and finally a fully-transparent face.

By default the edge is painted on center of the marker.
However, this can be customized in order to be painted on the
inner or outer edge only by setting the ``edge_mode`` property of
the ``PointsMarkerMaterial``.
To this end, we repeat the pattern with the inner and outer edge painted.
"""

# sphinx_gallery_pygfx_docs = 'screenshot'
# sphinx_gallery_pygfx_test = 'compare'

#   import os
# os.environ["WGPU_FORCE_OFFSCREEN"] = "true"

import numpy as np
from wgpu.gui.auto import WgpuCanvas, run
from wgpu.gui.offscreen import WgpuManualOffscreenCanvas as OffscreenCanvas
import pygfx as gfx


canvas = WgpuCanvas(size=(1600, 800))
# canvas = WgpuCanvas(size=(16384, 16384))
renderer = gfx.WgpuRenderer(canvas, pixel_ratio=1)

colors = np.array(
    [
        [1.0, 0.5, 0.5, 1.0],
        [0.5, 1.0, 0.5, 1.0],
        [0.5, 0.5, 1.0, 1.0],
        [0.5, 0.5, 1.0, 0.3],
        [0.0, 0.0, 0.0, 0.0],
    ],
    np.float32,
)

npoints = len(colors)

positions = np.zeros((npoints, 3), np.float32)
positions[:, 0] = np.arange(npoints) * 2
geometry = gfx.Geometry(positions=positions, colors=colors)


scene = gfx.Scene()
scene.add(gfx.Background.from_color("#bbb", "#777"))

pygfx_sdf = """
    let m_sqrt_3 = 1.7320508075688772;

    // coords uses WGSL coordinates.
    // we shift it so that the center of the triangle is at the origin.
    // for ease of calculations.
    var coord_for_sdf = coord / size + vec2<f32>(0.5, -0.5);

    // https://math.stackexchange.com/a/4073070
    // equilateral triangle has length of size
    //    sqrt(3) - 1
    let triangle_x = m_sqrt_3 - 1.;
    let one_minus_triangle_x = 2. - m_sqrt_3;
    let triangle_length = SQRT_2 * triangle_x;

    let pygfx_width = 0.10;

    let v1 = normalize(vec2<f32>(one_minus_triangle_x, 1));
    let r1_out = dot(coord_for_sdf, v1);
    let r1_in  = r1_out + pygfx_width;

    let v2 = normalize(vec2<f32>(-1, -one_minus_triangle_x));
    let r2_out = dot(coord_for_sdf, v2);
    let r2_in  = r2_out + pygfx_width;

    let v3 = normalize(vec2<f32>(triangle_x, -triangle_x));
    let r3_out = dot(coord_for_sdf - vec2(1, -one_minus_triangle_x), v3);
    let r3_in  = r3_out + pygfx_width;

    let inner_offset = 0.5 * (triangle_length - pygfx_width / 2.);
    let r1_out_blue = -r1_out - inner_offset;
    let r1_in_blue = r1_out_blue + pygfx_width;
    let r1_blue = max(
        max(r2_out, r3_in),
        max(r1_out_blue, -r1_in_blue)
    );

    let r2_out_blue = -r2_out - inner_offset;
    let r2_in_blue = r2_out_blue + pygfx_width;
    let r2_blue = max(
        max(r3_out, r1_in),
        max(r2_out_blue, -r2_in_blue)
    );

    let r3_out_blue = -r3_out - inner_offset;
    let r3_in_blue = r3_out_blue + pygfx_width;
    let r3_blue = max(
        max(r1_out, r2_in),
        max(r3_out_blue, -r3_in_blue)
    );

    let inner_triangle = min(r1_blue, min(r2_blue, r3_blue));

    let outer_triangle = max(
        max(r1_out, max(r2_out, r3_out)),
        -max(r1_in, max(r2_in, r3_in))
    );

    return min(inner_triangle, outer_triangle) * size;
"""


y = 0
text = gfx.Text(
    gfx.TextGeometry("centered", anchor="middle-middle", font_size=1),
    gfx.TextMaterial("#000"),
)
text.local.y = y
text.local.x = npoints
scene.add(text)

text = gfx.Text(
    gfx.TextGeometry("inner", anchor="middle-middle", font_size=1),
    gfx.TextMaterial("#000"),
)
text.local.y = y
text.local.x = 2 * npoints + npoints
scene.add(text)

text = gfx.Text(
    gfx.TextGeometry("outer", anchor="middle-middle", font_size=1),
    gfx.TextMaterial("#000"),
)
text.local.y = y
text.local.x = 4 * npoints + npoints
scene.add(text)

all_lines = []
for marker in gfx.MarkerShape:
    y += 2
    line = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=1,
            size_space="world",
            color_mode="vertex",
            # color_mode='debug',
            marker=marker,
            edge_color="#000",
            edge_width=0.1 if not marker == "custom" else 0.033333,
            custom_sdf=pygfx_sdf if marker == "custom" else None,
        ),
    )
    line.local.y = -y
    line.local.x = 1
    scene.add(line)
    all_lines.append(line)

    line_inner = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=1,
            size_space="world",
            color_mode="vertex",
            marker=marker,
            edge_color="#000",
            edge_width=0.1 if not marker == "custom" else 0.033333,
            edge_mode="inner",
            custom_sdf=pygfx_sdf if marker == "custom" else None,
        ),
    )

    line_inner.local.y = -y
    line_inner.local.x = 1 + 2 * npoints

    scene.add(line_inner)
    all_lines.append(line_inner)

    line_outer = gfx.Points(
        geometry,
        gfx.PointsMarkerMaterial(
            size=1,
            size_space="world",
            color_mode="vertex",
            marker=marker,
            edge_color="#000",
            edge_width=0.1 if not marker == "custom" else 0.033333,
            edge_mode="outer",
            custom_sdf=pygfx_sdf if marker == "custom" else None,
        ),
    )

    line_outer.local.y = -y
    line_outer.local.x = 1 + 4 * npoints

    scene.add(line_outer)
    all_lines.append(line_outer)

    text = gfx.Text(
        gfx.TextGeometry(marker, anchor="middle-right", font_size=1),
        gfx.TextMaterial("#000"),
    )
    text.local.y = -y
    text.local.x = 0
    scene.add(text)

camera = gfx.OrthographicCamera()
camera.show_object(scene, scale=0.7)
controller = gfx.PanZoomController(camera, register_events=renderer)


canvas.request_draw(lambda: renderer.render(scene, camera))


@renderer.add_event_handler("key_down")
def handle_event(event):
    if event.key == "d":
        color_mode = "debug"
        for line in all_lines:
            line.material.color_mode = color_mode
        print(f"color_mode {line.material.color_mode}")
    elif event.key == "v":
        color_mode = "vertex"
        for line in all_lines:
            line.material.color_mode = color_mode
        print(f"color_mode {line.material.color_mode}")
    elif event.key == "j":
        for line in all_lines:
            line.material.edge_width /= 1.1
        print(f"edge_width {line.material.edge_width}")
    elif event.key == "k":
        for line in all_lines:
            line.material.edge_width *= 1.1
        print(f"edge_width {line.material.edge_width}")

    canvas.update()


def offscreen_render_high_res(
    canvas, camera, scene,
    upscale=2,
    offscreen_side=400,
):
    import imageio
    from copy import deepcopy
    from tqdm import tqdm
    camera_state = camera.get_state()
    if camera_state['fov'] != 0.0:
        raise ValueError("This is only implemented for orthographic cameras")

    # Lets keep it simple, and render to a square...
    offscreen_canvas = OffscreenCanvas(size=(offscreen_side, offscreen_side))
    # Some users might be using their own renderer....
    # let them shoot themselves in the foot...
    offscreen_renderer = renderer.__class__(
        offscreen_canvas,
        # Want to copy over many of the parameters from the renderer
        # to ensure we get as close to the same rendering as possible
        pixel_ratio=renderer.pixel_ratio,
        pixel_filter=renderer.pixel_filter,
        show_fps=False,  # this is not really meaningful, is it???
        blend_mode=renderer.blend_mode,
        sort_objects=renderer.sort_objects,
        enable_events=False,
        gamma_correction=renderer.gamma_correction,
        # An addition to help get the same rendering for the background
        # and potentially other objects that use NDC coordinates
        main_camera=camera,
        # Can we copy over the device?
    )

    offscreen_camera = camera.__class__()
    offscreen_camera_state = deepcopy(camera_state)
    # del offscreen_camera_state['position']
    offscreen_canvas.request_draw(lambda: offscreen_renderer.render(scene, offscreen_camera))

    logical_size = canvas.get_logical_size()
    logical_size = (logical_size[0] * upscale, logical_size[1] * upscale)
    offscreen_logical_size = offscreen_canvas.get_logical_size()
    tiles = (
        int((logical_size[0] + offscreen_logical_size[0] - 1) / offscreen_logical_size[0]),
        int((logical_size[1] + offscreen_logical_size[1] - 1) / offscreen_logical_size[1]),
    )
    final_image = np.zeros((tiles[1] * offscreen_side, tiles[0] * offscreen_side, 3), dtype='uint8')
    for i in tqdm(np.ndindex(tiles), miniters=0, mininterval=0, total=tiles[0] * tiles[1]):
        offscreen_camera_side = max(
            (offscreen_logical_size[1] / (logical_size[1])) * camera_state['height'],
            (offscreen_logical_size[0] / (logical_size[0])) * camera_state['width'],
        )
        offscreen_camera_state['height'] = offscreen_camera_side
        offscreen_camera_state['width'] = offscreen_camera_side
        # Don't touch z???
        # offscreen_camera_state['position'][2] = camera_state['position'][2]

        offset_x = (i[0] - 0.5 * (tiles[0] - 1)) * offscreen_camera_side
        offset_y = (i[1] - 0.5 * (tiles[1] - 1)) * offscreen_camera_side

        # TODO: Take into account "rotation" and "reference_up"????

        offscreen_camera_state['position'][0] = camera_state['position'][0] + offset_x
        # GPU coordinates typically flip the y direction, so flip it here too in the offset
        offscreen_camera_state['position'][1] = camera_state['position'][1] - offset_y

        offscreen_camera.set_state(offscreen_camera_state)
        img = np.asarray(offscreen_canvas.draw())[..., :3]
        final_image[
            offscreen_side * i[1]:offscreen_side * (i[1] + 1),
            offscreen_side * i[0]:offscreen_side * (i[0] + 1)
        ] = img
        imageio.v3.imwrite(f"test_offscreen_rect_{i[1]:03d}_{i[0]:03d}.png", img)

    imageio.v3.imwrite("final_image.png", final_image)


if __name__ == "__main__":
    # import ipdb; ipdb.set_trace()

    renderer.render(scene, camera)
    snapshot = renderer.snapshot()[..., :3]
    # Mark -- I think it is important to have the
    # renderer.render
    # be invoked before the offscreen renderer
    # so that the main camera is correctly initialized.
    # There is a call to camera.update_projection_matrix
    # That I'm not too sure how to handle since it is seems
    # related to the viewport.
    offscreen_render_high_res(canvas, camera, scene)
    print(__doc__)
    import imageio
    imageio.v3.imwrite("snapshot.png", snapshot)
    # run()
