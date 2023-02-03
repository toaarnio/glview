#version 130

uniform bool debug;
uniform bool grayscale;
uniform bool gamma;
uniform float ev;
uniform int orientation;
uniform sampler2D texture;

in vec2 tex;
out vec4 color;

vec3 srgb_gamma(vec3 rgb) {
    bvec3 cutoff = lessThan(rgb, vec3(0.0031308));
    vec3 higher = vec3(1.055) * pow(rgb, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = rgb * vec3(12.92);
    return mix(higher, lower, cutoff);
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

  color = texture2D(texture, rotated) * exp(ev);  // exp(x) == 2^x
  color.rgb = grayscale ? color.rrr : color.rgb;
  color.rgb = gamma ? srgb_gamma(color.rgb) : color.rgb;
  color.r = debug ? color.r + 0.5 : color.r;
}
