#version 130

uniform int debug;
uniform bool grayscale;
uniform bool gamma;
uniform float ev;
uniform int orientation;
uniform sampler2D texture;
uniform struct {
  bool compress;
  vec3 power;
  vec3 thr;
  vec3 scale;
} gamut;

in vec2 tex;
out vec4 color;


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


vec3 srgb_gamma(vec3 rgb) {
  /**
   * Returns the given color with standard sRGB gamma applied. The input color
   * must be in [0, 1] range.
   */
  bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
  vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
  vec3 lower = rgb * vec3(12.92);
  return mix(higher, lower, cutoff);
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


void main() {
  color = texture2D(texture, rotate(tex, orientation));
  color.rgb = grayscale ? color.rrr : color.rgb;
  color.rgb = gamut.compress ? compress_gamut(color.rgb) : color.rgb;
  color.rgb = color.rgb * exp(ev);  // exp(x) == 2^x
  color.rgb = debug_indicators(color.rgb);
  color.rgb = gamma ? srgb_gamma(color.rgb) : color.rgb;
}
