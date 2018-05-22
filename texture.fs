#version 130

uniform bool grayscale;
uniform int orientation;
uniform sampler2D texture;

in vec2 tex;
out vec4 color;

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
  if (grayscale) {
    color = vec4(texture2D(texture, rotated).rrr, 1.0);
  } else {
    color = vec4(texture2D(texture, rotated).rgb, 1.0);
  }
}
