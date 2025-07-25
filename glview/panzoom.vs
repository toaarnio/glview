#version 300 es

uniform vec2 aspect;
uniform float scale;
uniform vec2 mousepos;

in vec2 vert;
out vec2 texcoords;

void main() {
  gl_Position = vec4(vert, 0.0, 1.0);
  gl_Position.xy += mousepos;
  gl_Position.xy *= scale;
  gl_Position.xy *= aspect;
  texcoords = vert / 2.0 + vec2(0.5, 0.5);
}
