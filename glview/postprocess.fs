#version 300 es

const int MAX_KERNEL_WIDTH = 25;

precision highp float;

uniform sampler2D img;
uniform ivec2 resolution;
uniform float magnification;
uniform int mirror;
uniform bool sharpen;
uniform float kernel[MAX_KERNEL_WIDTH * MAX_KERNEL_WIDTH];
uniform int kernw;
uniform bool autoexpose;
uniform float ae_gain;
uniform int tonemap;
uniform int gamma;
uniform int debug;

in vec2 texcoords;
out vec4 color;


/**************************************************************************************/
/*
/*    U T I L I T I E S
/*
/**************************************************************************************/


const vec3 ones = vec3(1.0);
const vec3 zeros = vec3(0.0);
const vec3 eps = vec3(5.0 / 256.0);


float min3(vec3 v) {
  /**
   * Returns the minimum value of the given vector. Annoyingly, there is no
   * built-in function in GLSL for this.
   */
  return min(min(v.x, v.y), v.z);
}


float max3(vec3 v) {
  /**
   * Returns the maximum value of the given vector. Annoyingly, there is no
   * built-in function in GLSL for this.
   */
  return max(max(v.x, v.y), v.z);
}


vec3 gamut_distance(vec3 rgb) {
  /**
   * Returns component-wise relative distances to gamut boundary; >1.0 means out
   * of gamut.  At least one of the input colors is always non-negative by
   * definition of homogeneous coordinates -- it's not possible for a point to
   * be outside of all three sides of a triangle at the same time.
   */
  vec3 max_rgb = vec3(max3(rgb));  // [0, maxval]
  vec3 abs_dist = max_rgb - rgb;  // [0, maxdiff]
  vec3 rel_dist = abs_dist / max_rgb;  // [0, >1]
  return rel_dist;
}


/**************************************************************************************/
/*
/*    I M A G E   O R I E N T A T I O N
/*
/**************************************************************************************/


vec2 flip(vec2 tc, int mirror) {
  /**
   * Flips the given texture coordinates in the horizontal or vertical or both
   * directions.
   */
  if ((mirror & 3) == 1) {  // horizontal only
    tc = vec2(1.0 - tc.x, tc.y);
  }
  if ((mirror & 3) == 2) {  // both directions
    tc = vec2(1.0 - tc.x, 1.0 - tc.y);
  }
  if ((mirror & 3) == 3) {  // vertical only
    tc = vec2(tc.x, 1.0 - tc.y);
  }
  return tc;
}


/**************************************************************************************/
/*
/*    G A M M A
/*
/**************************************************************************************/


vec3 srgb_gamma(vec3 rgb) {
  /**
   * Returns the given color with standard sRGB gamma applied. Input color
   * components are clamped to [0, 1].
   */
  rgb = clamp(rgb, 0.0, 1.0);
  bvec3 is_hi = greaterThan(rgb, vec3(0.0031308));
  vec3 hi = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
  vec3 lo = rgb * vec3(12.92);
  return mix(lo, hi, is_hi);
}


vec3 srgb_degamma(vec3 rgb) {
  /**
   * Returns the given color with standard sRGB inverse gamma applied. Input
   * color components are assumed to be in [0, 1].
   */
  bvec3 is_hi = greaterThan(rgb, vec3(0.04045));
  vec3 hi = pow(((rgb + vec3(0.055)) / vec3(1.055)), vec3(2.4));
  vec3 lo = rgb / vec3(12.92);
  return mix(lo, hi, is_hi);
}


vec3 st2084_gamma(vec3 rgb) {
  /**
   * Applies the standard SMPTE ST 2084 Perceptual Quantizer (PQ) on the given
   * color, compressing linear luminances from a nominal range of [0, 10000] cd/m²
   * into a perceptually uniform [0, 1] scale. Negative input values are clamped
   * to zero. This function is the equivalent of sRGB gamma for HDR monitors; see
   * https://en.wikipedia.org/wiki/Perceptual_quantizer.
   */
  rgb = max(rgb, 0.0);
  float m1 = 2610.0 / 16384.0;
  float m2 = 2523.0 / 4096.0 * 128.0;
  float c1 = 3424.0 / 4096.0;
  float c2 = 2413.0 / 4096.0 * 32.0;
  float c3 = 2392.0 / 4096.0 * 32.0;
  vec3 y = rgb / 10000.0;
  y = pow(y, vec3(m1));
  rgb = (c1 + c2 * y) / (1.0 + c3 * y);
  rgb = pow(rgb, vec3(m2));
  return rgb;
}


vec3 hlg(vec3 rgb) {
  /**
   * Applies the standard Hybrid Log-Gamma (HLG) electro-optical transfer function
   * on the given frame, as specified in ITU Rec. 2100. Negative input values are
   * clamped to zero. This function is an extension of sRGB gamma for HDR displays;
   * see https://en.wikipedia.org/wiki/Hybrid_log-gamma.
   */
  rgb = max(rgb, 0.0);
  vec3 a = vec3(0.17883277);
  vec3 b = vec3(1.0) - vec3(4.0) * a;
  vec3 c = vec3(0.5) - a * log(vec3(4.0) * a);
  vec3 lo = sqrt(vec3(3.0) * rgb);
  vec3 hi = a * log(vec3(12.0) * rgb - b) + c;
  bvec3 is_hi = greaterThan(rgb, vec3(1.0 / 12.0));
  return mix(lo, hi, is_hi);
}


vec3 apply_gamma(vec3 rgb) {
  /**
   * Applies the selected electro-optical transfer function (EOTF), aka. gamma,
   * on the given RGB color. The following functions are available:
   *
   *   0 - none
   *   1 - sRGB gamma
   *   2 - HLG (Hybrid Log-Gamma), max display luminance 1000 nits (cd/m²)
   *   3 - ST2084 (HDR10), assumed max display luminance 1000 nits (cd/m²)
   */
  switch (gamma) {
    case 1:
      rgb = srgb_gamma(rgb);
      break;
    case 2:
      rgb = hlg(rgb);
      break;
    case 3:
      rgb = st2084_gamma(rgb * 1000.0);
      break;
  }
  return rgb;
}


/**************************************************************************************/
/*
/*    C O L O R   S P A C E   C O N V E R S I O N S
/*
/**************************************************************************************/


vec3 xy_to_xyz(vec2 xy) {
  /**
   * Transforms the given color coordinates from CIE xy to CIE XYZ.
   */
  vec3 xyz;
  xyz.x = xy.x / xy.y;
  xyz.y = 1.0;
  xyz.z = (1.0 - xy.x - xy.y) / xy.y;
  return xyz;
}


mat3 rgb_to_xyz_mtx(vec2 xy_r, vec2 xy_g, vec2 xy_b, vec2 xy_w) {
  /**
   * Returns a 3 x 3 conversion matrix from RGB to XYZ, given the (x, y) chromaticity
   * coordinates of the RGB primaries and the reference white. Conversion formula taken
   * from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html.
   */
  vec3 XYZ_r = xy_to_xyz(xy_r);
  vec3 XYZ_g = xy_to_xyz(xy_g);
  vec3 XYZ_b = xy_to_xyz(xy_b);
  vec3 XYZ_w = xy_to_xyz(xy_w);
  mat3 M = mat3(XYZ_r, XYZ_g, XYZ_b);

  // Scale each column of the RGB-to-XYZ matrix with a scalar such
  // that [1, 1, 1] gets transformed to the given whitepoint (XYZ_w);
  // for example, M * [1, 1, 1] = [0.9504, 1.0, 1.0888] in case of D65.

  vec3 S = inverse(M) * XYZ_w;  // whitepoint in RGB
  M[0] *= S[0];  // R column vector scale
  M[1] *= S[1];  // G column vector scale
  M[2] *= S[2];  // B column vector scale
  return M;
}


mat3 xyz_to_rgb_mtx(vec2 xy_r, vec2 xy_g, vec2 xy_b, vec2 xy_w) {
  /**
   * Returns a 3 x 3 conversion matrix from XYZ to RGB, given the (x, y) chromaticity
   * coordinates of the RGB primaries and the reference white. Conversion formula taken
   * from http://www.brucelindbloom.com/index.html?Eqn_RGB_XYZ_Matrix.html.
   */
  mat3 M = rgb_to_xyz_mtx(xy_r, xy_g, xy_b, xy_w);
  M = inverse(M);
  return M;
}


mat3 p3_to_xyz_mtx() {
  /**
   * Returns the exact DCI-P3 to XYZ conversion matrix defined by the P3 specification.
   * Note that the matrix is computed from limited-precision primaries and whitepoint,
   * as per the specification.
   */
  vec2 D65_WP = vec2(0.3127, 0.3290);
  vec2 xy_r = vec2(0.680, 0.320);
  vec2 xy_g = vec2(0.265, 0.690);
  vec2 xy_b = vec2(0.150, 0.060);
  mat3 M = rgb_to_xyz_mtx(xy_r, xy_g, xy_b, D65_WP);
  return M;
}


mat3 xyz_to_p3_mtx() {
  /**
   * Returns the exact XYZ to DCI-P3 conversion matrix defined by the P3 specification.
   * Note that the matrix is computed from limited-precision primaries and whitepoint,
   * as per the specification.
   */
  mat3 M = inverse(p3_to_xyz_mtx());
  return M;
}


vec3 p3_to_xyz(vec3 rgb) {
  /**
   * Transforms the given DCI-P3 color to CIE XYZ.
   */
  vec3 xyz = p3_to_xyz_mtx() * rgb;
  return xyz;
}


vec3 xyz_to_p3(vec3 xyz) {
  /**
   * Transforms the given CIE XYZ color to DCI-P3.
   */
  vec3 rgb = xyz_to_p3_mtx() * xyz;
  return rgb;
}


/**************************************************************************************/
/*
/*    S H A R P E N I N G
/*
/**************************************************************************************/


vec4 conv2d(sampler2D img, vec2 tc) {
  /**
   * Convolves the given texture with a kernel defined in a uniform array. The color
   * of each pixel is multiplied by a factor proportional to the sum of its weighted
   * neighbors. The factor is clamped to an asymmetric [0.5, 5] range, so the bright
   * side of an edge is likely to be boosted more than the darker side.
   *
   * The convolution kernel is computed in screen space if the texture is minified,
   * and in texture space if it's magnified. If it's rendered at 1:1 scale (neither
   * minified nor magnified), screen space and texture space are the same.
   *
   * :param img: the input texture to convolve
   * :param tc: texture coordinates of the center pixel
   * :uniform ivec2 resolution: target viewport width and height, in pixels
   * :uniform float magnification: ratio of screen pixels to texture pixels
   * :uniform float[] kernel: array of per-pixel weights defining the kernel
   * :uniform int kernw: width and height of the kernel, in pixels
   * :returns: the original pixel color boosted by a 2D convolution kernel
   */
  float sum = 0.0f;
  vec2 xy_step = vec2(1.0) / vec2(resolution);
  xy_step *= max(magnification, 1.0);
  vec2 tc_base = tc - floor(float(kernw) / 2.0) * xy_step;
  for (int x = 0; x < kernw; x++) {
    for (int y = 0; y < kernw; y++) {
      float weight = kernel[y * kernw + x];
      vec2 tc = tc_base + vec2(x, y) * xy_step;
      vec4 pixel = texture(img, tc);
      float grayscale = pixel.r + pixel.g + pixel.b;
      grayscale = max(grayscale, 0.0f);
      sum += weight * grayscale;
    }
  }
  vec4 org = texture(img, tc);
  float org_gray = max(org.r + org.g + org.b, 0.0f);
  float boost = clamp(sum / org_gray, 0.5f, 5.0f);
  org.rgb = org.rgb * boost;
  return org;
}


/**************************************************************************************/
/*
/*    G L O B A L   T O N E   M A P P I N G
/*
/**************************************************************************************/


vec3 gtm_neutral(vec3 rgb, float ymax) {
  /**
   * Compresses RGB colors from [0, inf] to [0, ymax]. See apply_gtm() for
   * documentation.
   */
  const float desaturate = 0.1;
  const float ystart = 0.5;
  float d = ymax - ystart;

  float peak = max3(rgb);
  if (peak >= ystart) {
    // Compress the peak value from [ystart, inf] to [ystart, ymax],
    // scaling all RGB components by the same factor
    float new_peak = ymax - d * d / (ymax  + peak - 2.0 * ystart);
    rgb = rgb * new_peak / peak;

    // Interpolate towards the gray axis
    float g = ymax / (desaturate * (peak - new_peak) + ymax);
    rgb = mix(rgb, vec3(new_peak), 1.0 - g);
  }

  return rgb;
}


vec3 gtm_reinhard(vec3 rgb) {
  /**
   * Compresses RGB colors from [0, inf] to [0, 1]. See apply_gtm() for
   * documentation.
   */
  vec3 xyz = p3_to_xyz(rgb);
  float y_new = xyz.y / (1.0 + xyz.y);
  float ratio = y_new / xyz.y;
  rgb = rgb * ratio;
  return rgb;
}


vec3 apply_gtm(vec3 rgb, float ymax) {
  /**
   * Applies the selected global tone mapping function (GTM) on the given HDR
   * color. The input is assumed to be in a nominal [0, 1] scale, such that
   * diffuse reflectances are in that range, whereas direct light sources and
   * specular highlights can exceed 1.0, often by several orders of magnitude.
   * For non-negative inputs, the output is guaranteed to be in [0, ymax].
   *
   * The following GTM functions are available:
   *
   *   0 - none
   *   1 - neutral
   *   2 - reinhard
   */
  switch (tonemap) {
    case 1:
      rgb = gtm_neutral(rgb, ymax);
      break;
    case 2:
      rgb = gtm_reinhard(rgb);
      break;
  }
  return rgb;
}


/**************************************************************************************/
/*
/*    D E B U G G I N G
/*
/**************************************************************************************/


vec3 debug_indicators(vec3 rgb) {
  /**
   * Returns a color-coded debug representation of the given pixel, depending on
   * its value and a user-defined debug mode. The following modes are available:
   *
   *   0 - no-op => keep original pixel color
   *   1 - overexposed => red; out-of-gamut => blue; magenta => both
   *   2 - out-of-gamut => shades of green
   *   3 - normalized color => rgb' = 1.0 - rgb / max(rgb)
   */
  float gdist = max3(gamut_distance(rgb));  // [0, >1]
  float oog_dist = clamp(5.0 * (gdist - 1.0), 0.0, 1.0);  // [1.0, 1.2] => [0, 1]
  bool overflow = any(greaterThanEqual(rgb, ones));  // 1.0 treated as overflow
  bool underflow = all(lessThan(abs(rgb), eps));
  bool oog = gdist >= 1.0;
  switch (debug) {
    case 1:  // red/blue/magenta
      if ((oog && !underflow) || overflow)
        rgb = vec3(overflow, 0.0, oog);
      break;
    case 2:  // shades of green
      if (oog && !underflow)
        rgb = vec3(0.0, oog_dist, 0.0);
      break;
    case 3:  // normalized color
      rgb = 1.0 - gamut_distance(rgb);
      break;
  }
  return rgb;
}


/**************************************************************************************/
/*
/*    M A I N
/*
/**************************************************************************************/


void main() {
  vec2 tc = flip(texcoords, mirror);
  color = sharpen ? conv2d(img, tc) : texture(img, tc);
  color.rgb *= autoexpose ? ae_gain : 1.0;
  color.rgb = apply_gtm(color.rgb, 1.0);
  color.rgb = debug_indicators(color.rgb);
  color.rgb = apply_gamma(color.rgb);
}
