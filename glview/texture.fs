#version 300 es

precision highp float;

uniform sampler2D img;
uniform int orientation;
uniform bool degamma;
uniform bool grayscale;

in vec2 texcoords;
out vec4 color;


/**************************************************************************************/
/*
/*    G A M M A
/*
/**************************************************************************************/


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
/*    M A I N
/*
/**************************************************************************************/


void main() {
  color = texture(img, rotate(texcoords, orientation));
  color.rgb = degamma ? srgb_degamma(color.rgb) : color.rgb;
  color.rgb = grayscale ? color.rrr : color.rgb;
}
