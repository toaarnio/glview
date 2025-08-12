#version 300 es

precision highp float;

// Inputs & outputs

in vec2 texcoords;
out vec4 color;

// Texture sampling & sharpening

const int MAX_KERNEL_WIDTH = 25;

uniform sampler2D img;
uniform int mirror;
uniform bool sharpen;
uniform ivec2 resolution;
uniform float magnification;
uniform float kernel[MAX_KERNEL_WIDTH * MAX_KERNEL_WIDTH];
uniform int kernw;

// Exposure, tone, gamut & contrast

uniform float minval;
uniform float maxval;
uniform float diffuse_white;
uniform float peak_white;
uniform float ev;
uniform bool autoexpose;
uniform float ae_gain;
uniform int tonemap;
uniform float contrast;
uniform struct {
  bool compress;
  vec3 power;
  vec3 thr;
  vec3 scale;
} gamut;

// Standard color spaces

const int sRGB = 0;
const int P3 = 1;
const int Rec2020 = 2;
const int XYZ = 3;

uniform int cs_in;
uniform int cs_out;
uniform int gamma;

// Diagnostics

uniform int debug;


/**************************************************************************************/
/*
/*    U T I L I T I E S
/*
/**************************************************************************************/


const vec3 ones = vec3(1.0);
const vec3 zeros = vec3(0.0);


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


float sum3(vec3 v) {
  /**
   * Returns the sum of the components of the given vector. Annoyingly, there
   * is no built-in function in GLSL for this.
   */
  return v.x + v.y + v.z;
}


mat3 mround(mat3 mtx) {
  /**
   * Rounds the elements of the given matrix to the nearest integer. Annoyingly,
   * the built-in round() function does not work with matrices.
   */
  mtx[0] = round(mtx[0]);
  mtx[1] = round(mtx[1]);
  mtx[2] = round(mtx[2]);
  return mtx;
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


mat3 srgb_to_xyz_mtx() {
  /**
   * Returns the exact sRGB to XYZ conversion matrix defined by the sRGB specification.
   * Note that the matrix is computed from limited-precision primaries and whitepoint,
   * and rounded to four decimals at the end, as per the specification.
   */
  vec2 D65_WP = vec2(0.3127, 0.3290);
  vec2 xy_r = vec2(0.640, 0.330);
  vec2 xy_g = vec2(0.300, 0.600);
  vec2 xy_b = vec2(0.150, 0.060);
  mat3 M = rgb_to_xyz_mtx(xy_r, xy_g, xy_b, D65_WP);
  M = mround(M * 10000.0) / 10000.0;
  return M;
}


mat3 xyz_to_srgb_mtx() {
  /**
   * Returns the exact XYZ to sRGB conversion matrix defined by the sRGB specification.
   * Note that the matrix is computed from limited-precision primaries and whitepoint,
   * and rounded to four decimals at the end, as per the specification.
   */
  mat3 M = inverse(srgb_to_xyz_mtx());
  M = mround(M * 10000.0) / 10000.0;
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


mat3 rec2020_to_xyz_mtx() {
  /**
   * Returns the exact Rec2020 to XYZ conversion matrix defined by the Rec2020
   * specification. Note that the matrix is computed from limited-precision
   * primaries and whitepoint, as per the specification.
   */
  vec2 D65_WP = vec2(0.3127, 0.3290);
  vec2 xy_r = vec2(0.708, 0.292);
  vec2 xy_g = vec2(0.170, 0.797);
  vec2 xy_b = vec2(0.131, 0.046);
  mat3 M = rgb_to_xyz_mtx(xy_r, xy_g, xy_b, D65_WP);
  return M;
}


mat3 xyz_to_rec2020_mtx() {
  /**
   * Returns the exact XYZ to Rec2020 conversion matrix defined by the Rec2020
   * specification. Note that the matrix is computed from limited-precision
   * primaries and whitepoint, as per the specification.
   */
  mat3 M = inverse(rec2020_to_xyz_mtx());
  return M;
}


vec3 srgb_to_xyz(vec3 rgb) {
  /**
   * Transforms the given sRGB color to CIE XYZ.
   */
  vec3 xyz = srgb_to_xyz_mtx() * rgb;
  return xyz;
}


vec3 xyz_to_srgb(vec3 xyz) {
  /**
   * Transforms the given CIE XYZ color to sRGB.
   */
  vec3 rgb = xyz_to_srgb_mtx() * xyz;
  return rgb;
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


vec3 rec2020_to_xyz(vec3 rgb) {
  /**
   * Transforms the given Rec2020 color to CIE XYZ.
   */
  vec3 xyz = rec2020_to_xyz_mtx() * rgb;
  return xyz;
}


vec3 xyz_to_rec2020(vec3 xyz) {
  /**
   * Transforms the given CIE XYZ color to Rec2020.
   */
  vec3 rgb = xyz_to_rec2020_mtx() * xyz;
  return rgb;
}


vec3 csconv(vec3 rgb, int cs_in, int cs_out) {
  /**
   * Transforms the given color from a given input color space to a given output
   * color space. If the input and output color spaces are the same, the color is
   * unchanged. The following color spaces are available as both input & output:
   *
   *   0 - sRGB
   *   1 - DCI-P3
   *   2 - Rec2020
   *   3 - CIEXYZ
   *
   * As a concrete example, to display a P3-JPEG captured by a modern smartphone
   * on a high-quality wide-gamut monitor, you would set the input color space to
   * DCI-P3 and the output to Rec2020, while making sure that the monitor is set
   * to Rec2020 mode.
   */
  if (cs_in != cs_out) {
    vec3 xyz;
    switch (cs_in) {
      case sRGB:
        xyz = srgb_to_xyz(rgb);
        break;
      case P3:
        xyz = p3_to_xyz(rgb);
        break;
      case Rec2020:
        xyz = rec2020_to_xyz(rgb);
        break;
      case XYZ:
        xyz = rgb;
        break;
      default:
        xyz = rgb;
        break;
    }
    switch (cs_out) {
      case sRGB:
        rgb = xyz_to_srgb(xyz);
        break;
      case P3:
        rgb = xyz_to_p3(xyz);
        break;
      case Rec2020:
        rgb = xyz_to_rec2020(xyz);
        break;
      case XYZ:
        rgb = xyz;
        break;
      default:
        rgb = xyz;
        break;
    }
  }
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
  float sharp_gray = 0.0f;
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
      sharp_gray += weight * grayscale;
    }
  }
  vec4 org = texture(img, tc);
  float org_gray = max(org.r + org.g + org.b, 0.0f);
  float boost = clamp(sharp_gray / org_gray, 0.5f, 5.0f);
  org.rgb *= boost;
  return org;
}


/**************************************************************************************/
/*
/*    G A M U T   C O M P R E S S I O N
/*
/**************************************************************************************/


vec3 gamut_distance(vec3 rgb) {
  /**
   * Returns component-wise relative distances to gamut boundary; >1.0 means
   * out of gamut. At least one of the input colors is always non-negative by
   * definition of homogeneous coordinates -- it's not possible for a point to
   * be outside of all three sides of a triangle at the same time.
   */
  vec3 max_rgb = vec3(max3(rgb));  // [0, maxval]
  vec3 abs_dist = max_rgb - rgb;  // [0, maxdiff]
  vec3 rel_dist = abs_dist / max_rgb;  // [0, >1]
  return rel_dist;
}


vec3 compress_distance(vec3 dist) {
  /**
   * Compresses the given component-wise distances from [0, limit] to [0, 1],
   * where limit is a user-defined value greater than 1.0. Extreme out-of-gamut
   * colors with distances above the limit will be >1.0 even after compression.
   */
  vec3 denom = (dist - gamut.thr) / gamut.scale;  // may be negative or large
  denom = max(denom, 0.0);  // >= 0.0, may be large
  denom = 1.0 + pow(denom, gamut.power);  // >= 1.0, may be inf
  denom = pow(denom, 1.0 / gamut.power);  // >= 1.0, may be inf
  vec3 cdist = gamut.thr + (dist - gamut.thr) / denom;  // [0, 1]
  return cdist;
}


vec3 compress_gamut(vec3 rgb) {
  /**
   * Compresses out-of-gamut (negative) RGB colors into the gamut. As a a side
   * effect, in-gamut colors that are close to the gamut boundary are also
   * compressed by some amount, depending on the user-defined compression curve
   * shape.
   */
  vec3 max_rgb = vec3(max3(rgb));  // [0, maxval]
  vec3 rel_dist = gamut_distance(rgb);  // >1.0 means out of gamut
  vec3 cdist = compress_distance(rel_dist);  // [0, >1] => [0, 1]
  vec3 crgb = (1.0 - cdist) * max_rgb;  // [0, 1] * [0, maxval] = [0, maxval]
  return crgb;
}


/**************************************************************************************/
/*
/*    G L O B A L   T O N E   M A P P I N G
/*
/**************************************************************************************/


vec3 gtm_neutral(vec3 rgb, float ystart, float yclip) {
  /**
   * Compresses RGB colors from [ystart, yclip] to [ystart, 1], keeping
   * the [0, ystart] range intact.
   *
   * See https://www.desmos.com/calculator/3bb4905ea6 for an interactive
   * visualization.
   */
  float peak = max3(rgb);
  if (peak > ystart) {

    // solve for ymax (the asymptote of ynew) for the given [ystart, yclip],
    // such that y = yclip yields ynew = 1, and y > yclip yields ynew > 1

    yclip = max(yclip, 1.0 + 1e-6);  // avoid divide by zero
    float ymax = (yclip - 2.0 * ystart + ystart * ystart) / (yclip - 1.0);

    // scale all RGB components by the same factor; note that the factor is
    // necessarily less than 1.0, because ynew is always less than peak

    float d = ymax - ystart;
    float ynew = ymax - d * d / (ymax  + peak - 2.0 * ystart);
    float scale = ynew / peak;
    rgb = rgb * scale;

    // optionally, interpolate gradually towards the gray axis

    float desaturate = 0.0;
    float gray_pct = 1.0 - ymax / (desaturate * (peak - ynew) + ymax);
    rgb = mix(rgb, vec3(ynew), gray_pct);
  }
  return rgb;
}


vec3 gtm_reinhard(vec3 rgb, int cspace) {
  /**
   * Compresses RGB colors from [0, inf] to [0, 1]. See apply_gtm() for
   * documentation.
   */
  vec3 xyz = csconv(rgb, cspace, XYZ);
  float y_new = xyz.y / (1.0 + xyz.y);
  float ratio = y_new / xyz.y;
  rgb = rgb * ratio;
  return rgb;
}


vec3 filmic_curve(vec3 rgb) {
  float A = 0.15;  // shoulder strength (default: 0.15)
  float B = 0.50;  // linear strength (default: 0.50)
  float C = 0.10;  // linear angle (default: 0.10)
  float D = 0.20;  // toe strength (default: 0.20)
  float E = 0.03;  // toe numerator (default: 0.02)
  float F = 0.30;  // toe denominator (default: 0.30)
  vec3 numerator = rgb * (A * rgb + C * B) + D * E;
  vec3 denominator = rgb * (A * rgb + B) + D * F;
  float bias = E / F;
  rgb = numerator / denominator - bias;
  return rgb;
}


vec3 gtm_filmic(vec3 rgb, int cspace, vec3 peak_white) {
  /**
   * A filmic tonemapping operator based on the Uncharted 2 curve, with
   * a straight-line extension for negative (out-of-gamut) inputs. This
   * provides a more cinematic and perceptually pleasing result than
   * simple Reinhard, with better saturation and highlight rolloff.
   * The operator is applied in Rec2020 color space to minimize the
   * adverse impact of negative colors.
   *
   * See http://filmicworlds.com/blog/filmic-tonemapping-operators/ for
   * background and https://www.desmos.com/calculator/jimezytyho for an
   * interactive visualization.
   */
  vec3 wide = csconv(rgb, cspace, Rec2020);
  vec3 pos = filmic_curve(wide) / filmic_curve(peak_white);
  vec3 neg = wide / peak_white;
  wide = mix(pos, neg, lessThan(wide, zeros));
  rgb = csconv(wide, Rec2020, cspace);
  return rgb;
}


vec3 apply_gtm(int tmo, vec3 rgb, int cspace, float diffuse_white, float peak_white) {
  /**
   * Applies the selected tone mapping operator (TMO) on the given HDR color.
   * The input is assumed to be in the given linear RGB color space, and have
   * the given diffuse and peak white levels. Note that peak_white is defined
   * as relative to diffuse_white, and must be >= 1.0.
   *
   * The input color is first normalized such that [0, diffuse_white] maps to
   * [0, 1], then compressed by the chosen tone mapping operator:
   *
   *   0 - none - return the input color unmodified
   *   1 - neutral - compress [0.8, peak_white] to [0.8, 1]
   *   2 - reinhard - compress [0, inf] to [0, 1]
   *   3 - filmic - compress [0, peak_white] to [0, 1]
   */
  switch (tmo) {
    case 1:
      rgb = rgb / diffuse_white;
      rgb = gtm_neutral(rgb, 0.8, peak_white);
      break;
    case 2:
      rgb = rgb / diffuse_white;
      rgb = gtm_reinhard(rgb, cspace);
      break;
    case 3:
      rgb = rgb / diffuse_white;
      rgb = gtm_filmic(rgb, cspace, vec3(peak_white));
      break;
  }
  return rgb;
}


/**************************************************************************************/
/*
/*    G L O B A L   C O N T R A S T   E N H A N C E M E N T
/*
/**************************************************************************************/


vec3 apply_gce(int mode, vec3 rgb, int cspace, float contrast, float whitelevel) {
  /**
   * Applies a polynomial S-curve to luminance for global contrast enhancement.
   * Hue is preserved (approximately, not accounting for perceptual effects) by
   * multiplying each component of the input color by the same factor.
   *
   * The domain and range of the contrast enhancement function is [0, whitelevel];
   * outside of that domain, the function reduces to a constant scalar multiplier.
   * See visualization at https://www.desmos.com/calculator/jimezytyho.
   *
   * :param mode: contrast enhancement mode; 0 = off, >0 = on
   * :param rgb: input color; nominal range = [0, whitelevel]
   * :param cspace: input color space for deriving luminance
   * :param contrast: strength of contrast enhancement; range = [0, 1]
   * :param whitelevel: maximum input luminance; typically 1.0
   * :returns: contrast-enhanced input color; nominal range = [0, whitelevel]
   */
  float luma = csconv(rgb, cspace, XYZ).y;
  if (mode > 0 && luma > 0.0) {
    float normed_luma = luma / whitelevel;  // [0, wl] => [0, 1]
    float contrasted_luma = smoothstep(0.0, whitelevel, luma);  // [0, wl] => [0, 1]
    float luma_gain = mix(normed_luma, contrasted_luma, contrast) / normed_luma;
    rgb = rgb * luma_gain;  // gain = [0, >1]
  }
  return rgb;
}


/**************************************************************************************/
/*
/*    D E B U G G I N G
/*
/**************************************************************************************/


vec3 debug_indicators(vec3 rgb, vec3 orig_rgb, float diffuse_level, float peak_level) {
  /**
   * Returns a color-coded debug representation of the given pixel, depending on
   * its value and a user-defined debug mode. The following modes are available:
   *
   *   0 - no-op => keep original pixel color
   *   1 - red => overexposed; blue => out-of-gamut; magenta => both
   *   2 - shades of green => distance to gamut
   *   3 - normalized color => rgb' = 1.0 - rgb / max(rgb)
   *   4 - red => above diffuse white; magenta => above peak white
   */
  float gdist = max3(gamut_distance(rgb));  // [0, >1]
  float oog_dist = clamp(5.0 * (gdist - 1.0), 0.0, 1.0);  // [1.0, 1.2] => [0, 1]
  float maxx = max3(abs(rgb));
  float overflow_dist = 0.5 + 0.5 * (maxx - 1.0) / (peak_white - 1.0);  // [0.5, 1]
  bool overflow = maxx >= 1.0;  // 1.0 treated as overflow
  overflow_dist = overflow ? overflow_dist : 0.0;
  bool underflow = maxx < (1.0 / 255.0f);
  bool oog = gdist >= 1.0;
  switch (debug) {
    case 1:  // red/blue/magenta
      if (overflow || (oog && !underflow))
        rgb = vec3(overflow_dist, 0.0, oog);
      break;
    case 2:  // shades of green
      if (oog && !underflow)
        rgb = vec3(0.0, oog_dist, 0.0);
      break;
    case 3:  // normalized color
      rgb = 1.0 - gamut_distance(rgb);
      break;
    case 4:  // red => above diffuse white; magenta => above peak white
      if (max3(abs(orig_rgb)) > diffuse_level)
        rgb = vec3(1.0, 0.0, 0.0);
      if (max3(abs(orig_rgb)) > peak_level * diffuse_level)
        rgb.b = 1.0;
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
  vec3 debug_rgb;
  vec2 tc = flip(texcoords, mirror);
  float gain = autoexpose ? ae_gain * exp(ev) : exp(ev);  // exp(x) == 2^x
  color = sharpen ? conv2d(img, tc) : texture(img, tc);
  color.rgb = (color.rgb - minval) / maxval;  // [minval, maxval] => [0, 1]
  debug_rgb = color.rgb * exp(ev);
  color.rgb = color.rgb * gain;
  color.rgb = csconv(color.rgb, cs_in, cs_out);
  color.rgb = apply_gtm(tonemap, color.rgb, cs_out, diffuse_white * gain, peak_white);
  color.rgb = apply_gce(tonemap, color.rgb, cs_out, contrast, 1.0);
  color.rgb = gamut.compress ? compress_gamut(color.rgb) : color.rgb;
  color.rgb = debug_indicators(color.rgb, debug_rgb, diffuse_white, peak_white);
  color.rgb = apply_gamma(color.rgb);
}
