#version 330

uniform bool grayscale;
uniform sampler2D texture;

in vec2 tex;
out vec4 color;

void main() {
  vec2 flipped = vec2(tex.x, 1.0 - tex.y);
  if (grayscale) {
    color = vec4(texture2D(texture, flipped).rrr, 1.0);
  } else {
    color = vec4(texture2D(texture, flipped).rgb, 1.0);
  }
}
