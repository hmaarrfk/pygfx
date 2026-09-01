// Conversion between srgb and physical (linear) color.
//
// Both directions deliberately avoid pow(). Vulkan defines pow(x, y) as
// exp2(y * log2(x)), and allows exp2 and log2 3 ULP outside [0.5, 2.0], so a
// pow() result is not reproducible between implementations. Of the float ops,
// only OpFAdd, OpFSub and OpFMul are required to be correctly rounded.
//
// That is not academic. Rendering to an 'rgba8unorm-srgb' target hands the
// conversion to the driver instead, and llvmpipe's version of it is built on
// rsqrtps, the ~12 bit approximate reciprocal square root whose low bits x86
// leaves implementation defined. Intel and AMD return different bits for it,
// so the same scene renders to a different 8-bit image on the two vendors --
// about 2% of values land on the other side of a rounding boundary. That is
// why the renderer now targets plain 'rgba8unorm' and calls physical2srgb()
// below, rather than letting the driver do it.
//
// In mesa, pinned at 8e12d6000b247715b7c9bfaac67bca565dd8b9d8:
//   - the approximation, and mesa's own comment on it ("the constants are
//     magic values. They were found empirically ... This function has an error
//     of max +-0.17. Not sure this is actually enough"):
//     src/gallium/auxiliary/gallivm/lp_bld_format_srgb.c:244-292
//   - what lp_build_fast_rsqrt() lowers to, llvm.x86.sse.rsqrt.ps and
//     llvm.x86.avx.rsqrt.ps.256: src/gallium/auxiliary/gallivm/lp_bld_arit.c:2616-2639
//   - mesa's own deterministic CPU-side encode, an integer LUT that the JIT
//     path does not use: src/util/format_srgb.h:92-126

fn srgb2physical(color: vec3<f32>) -> vec3<f32> {
    // In Python, the below reads as
    // c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    //
    // The simple version of the curved part, kept for reference, was:
    //     let f = pow((color + 0.055) / 1.055, vec3<f32>(2.4));
    //
    // Instead: a degree-6 minimax polynomial in x over the curved part's
    // domain [0.0813, 1], by Horner, so multiplies and adds only. Max absolute
    // error 6.0e-6. The coefficients sum to exactly 1, so white stays white.
    let x = (color + 0.055) / 1.055;
    let f = (((((-0.073726393 * x
        + 0.315543950) * x
        + -0.614526184) * x
        + 0.960801971) * x
        + 0.428587764) * x
        + -0.017192921) * x
        + 0.000511814;
    let t = color / 12.92;
    return select(f, t, color <= vec3<f32>(0.04045));
}


fn physical2srgb(color: vec3<f32>) -> vec3<f32> {
    // The inverse of srgb2physical. In Python, the below reads as
    // c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1 / 2.4) - 0.055
    //
    // The simple version, kept for reference, was:
    //     let f = 1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055;
    //
    // A plain polynomial is no good in this direction: the curve is
    // near-vertical at the low end, and even degree 12 only reaches ~0.94 of an
    // 8-bit level. So this uses the basis mesa uses for the same job, powers of
    // a square root, with two extra terms and refitted coefficients. Max error
    // 0.0013 of an 8-bit level, about 130x better than mesa's own version. The
    // coefficients sum to exactly 1, so white stays white, and every one of the
    // 256 8-bit srgb values round-trips exactly through srgb2physical() --
    // see tests/renderers/test_colorspace.py.
    let s = sqrt(color);
    let s38 = sqrt(sqrt(s * color));
    let s25 = sqrt(s);
    let f = 0.279086994 * s
        + 0.876495078 * s38
        - 0.099148823 * s25
        - 0.003900440 * color
        - 0.052532808;
    let t = color * 12.92;
    return select(f, t, color <= vec3<f32>(0.0031308));
}
