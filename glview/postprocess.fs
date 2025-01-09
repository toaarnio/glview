#version 140

precision highp float;

uniform int gamma;
uniform sampler2D texture;

in vec2 texcoords;
out vec4 color;


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
/*    S H A R P E N I N G
/*
/**************************************************************************************/


vec4 sharpen(sampler2D texture, vec2 texcoords) {
  color = texture2D(texture, texcoords);
  return color;
}


/**************************************************************************************/
/*
/*    M A I N
/*
/**************************************************************************************/


void main() {
  color = sharpen(texture, texcoords);
  color.rgb = apply_gamma(color.rgb);
}
