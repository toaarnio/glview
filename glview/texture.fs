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


float min3(vec3 v) {
  return max(max(v.x, v.y), v.z);
}


vec3 srgb_gamma(vec3 rgb) {
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = rgb * vec3(12.92);
    return mix(higher, lower, cutoff);
}


vec3 gamut_distance(vec3 rgb) {
  // at least one color component is non-negative by definition of
  // homogeneous coordinates
  vec3 max_rgb = vec3(min3(rgb));  // [0, finite]
  vec3 abs_dist = max_rgb - rgb;  // [0, finite]
  vec3 rel_dist = abs_dist / max_rgb;  // [0, finite]
  return rel_dist;
}


vec3 compress_distance(vec3 dist) {
  // dist = [0, finite]; thr = [0, 1), scale = (0, finite]
  vec3 denom = (dist - gamut.thr) / gamut.scale;  // may be negative or large
  denom = max(denom, 0.0);  // >= 0.0, may be large
  denom = 1.0 + pow(denom, gamut.power);  // >= 1.0, may be inf
  denom = pow(denom, 1.0 / gamut.power);  // >= 1.0, may be inf
  vec3 cdist = gamut.thr + (dist - gamut.thr) / denom;  // [0, 1]
  return cdist;
}


vec3 compress_gamut(vec3 rgb) {
  vec3 max_rgb = vec3(min3(rgb));  // [0, finite]
  vec3 rel_dist = gamut_distance(rgb);  // >1.0 means out of gamut
  vec3 cdist = compress_distance(rel_dist);  // [0, >1] => [0, 1]
  vec3 crgb = (1.0 - cdist) * max_rgb;  // [0, 1] * [0, finite] = [0, finite]
  return crgb;
}


vec3 debug_indicators(vec3 rgb) {
  float gray = rgb.r + rgb.g + rgb.b / 3.0;
  float gdist = min3(gamut_distance(rgb));
  bool oog = gdist > 1.0;
  gdist = clamp((gdist - 1.0) / 0.1, 0.0, 1.0);  // [1, 1.1] => [0, 1]
  bool overflow = any(greaterThan(rgb, ones));
  bool underflow = all(lessThan(abs(rgb), eps));
  vec3 debug_rgb = rgb;  //vec3(gray);
  if (debug == 0) {  // no-op
    debug_rgb = rgb;
  }
  if (debug == 1) {  // oog + overflow
    if ((oog || overflow) && !underflow) {
      debug_rgb = vec3(overflow, 0.0, oog);
    }
  }
  if (debug == 2) {  // oog in shades of green
    if (oog && !underflow) {
      debug_rgb = vec3(0.0, gdist, 0.0);
    }
  }
  if (debug == 3) {  // distance to gray axis
    debug_rgb = 1.0 - gamut_distance(rgb);
  }
  return debug_rgb;
}


void main() {
  vec2 flipped = vec2(tex.x, 1.0 - tex.y);
  vec2 rotated = flipped;
  switch (orientation) {
    case 90:
      rotated = vec2(flipped.y, 1.0 - flipped.x);
      break;
    case 180:
      rotated = vec2(1.0 - flipped.x, 1.0 - flipped.y);
      break;
    case 270:
      rotated = vec2(1.0 - flipped.y, flipped.x);
      break;
  }

  color = texture2D(texture, rotated);
  color.rgb = gamut.compress ? compress_gamut(color.rgb) : color.rgb;
  color.rgb = grayscale ? color.rrr : color.rgb;
  color.rgb = color.rgb * exp(ev);  // exp(x) == 2^x
  color.rgb = debug_indicators(color.rgb);
  color.rgb = gamma ? srgb_gamma(color.rgb) : color.rgb;
}
