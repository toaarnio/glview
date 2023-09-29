#version 140

precision highp float;

uniform int debug;
uniform int cs_in;
uniform int cs_out;
uniform bool grayscale;
uniform bool degamma;
uniform int gamma;
uniform float ev;
uniform float maxval;
uniform int orientation;
uniform sampler2D texture;
uniform struct {
  bool compress;
  vec3 power;
  vec3 thr;
  vec3 scale;
} gamut;

in vec2 texcoords;
out vec4 color;


/**************************************************************************************/
/*
/*    U T I L I T I E S
/*
/**************************************************************************************/


const vec3 ones = vec3(1.0);
const vec3 zeros = vec3(0.0);
const vec3 eps = vec3(5.0 / 256);


float max3(vec3 v) {
  /**
   * Returns the maximum value of the given vector. Annoyingly, there is no
   * built-in function in GLSL for this.
   */
  return max(max(v.x, v.y), v.z);
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
  float m1 = 2610.0 / 16384;
  float m2 = 2523.0 / 4096 * 128;
  float c1 = 3424.0 / 4096;
  float c2 = 2413.0 / 4096 * 32;
  float c3 = 2392.0 / 4096 * 32;
  vec3 y = rgb / 10000;
  y = pow(y, vec3(m1));
  rgb = (c1 + c2 * y) / (1 + c3 * y);
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


vec3 csconv(vec3 rgb) {
  /**
   * Transforms the given color from a given input color space to a given output
   * color space. If the input and output color spaces are the same, the color is
   * unchanged. The following color spaces are available as both input & output:
   *
   *   0 - sRGB
   *   1 - DCI-P3
   *   2 - Rec2020
   *
   * As a concrete example, to display a P3-JPEG captured by a modern smartphone
   * on a high-quality wide-gamut monitor, you would set the input color space to
   * DCI-P3 and the output to Rec2020, while making sure that the monitor is set
   * to Rec2020 mode.
   */
  if (cs_in != cs_out) {
    vec3 xyz;
    switch (cs_in) {
      case 0:
        xyz = srgb_to_xyz(rgb);
        break;
      case 1:
        xyz = p3_to_xyz(rgb);
        break;
      case 2:
        xyz = rec2020_to_xyz(rgb);
        break;
      default:
        xyz = rgb;
        break;
    }
    switch (cs_out) {
      case 0:
        rgb = xyz_to_srgb(xyz);
        break;
      case 1:
        rgb = xyz_to_p3(xyz);
        break;
      case 2:
        rgb = xyz_to_rec2020(xyz);
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
/*    G A M U T   C O M P R E S S I O N
/*
/**************************************************************************************/


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


vec3 compress_distance(vec3 dist) {
  /**
   * Compresses the given component-wise distances from [0, limit] to [0, 1],
   * where limit is a user-defined value greater than 1.0.  Extreme out-of-gamut
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
/*    I M A G E   R O T A T I O N
/*
/**************************************************************************************/


vec2 rotate(vec2 tc, int degrees) {
  /**
   * Flips the given texture coordinates in the Y direction, then rotates
   * counterclockwise by 0/90/180/270 degrees.
   */
  tc = vec2(tc.x, 1.0 - tc.y);
  switch (degrees) {
    case 90:
      tc = vec2(tc.y, 1.0 - tc.x);
      break;
    case 180:
      tc = vec2(1.0 - tc.x, 1.0 - tc.y);
      break;
    case 270:
      tc = vec2(1.0 - tc.y, tc.x);
      break;
  }
  return tc;
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
  color = texture2D(texture, rotate(texcoords, orientation));
  color.rgb = color.rgb / maxval;
  color.rgb = degamma ? srgb_degamma(color.rgb) : color.rgb;
  color.rgb = csconv(color.rgb);
  color.rgb = grayscale ? color.rrr : color.rgb;
  color.rgb = gamut.compress ? compress_gamut(color.rgb) : color.rgb;
  color.rgb = color.rgb * exp(ev);  // exp(x) == 2^x
  color.rgb = debug_indicators(color.rgb);
  color.rgb = apply_gamma(color.rgb);
}
