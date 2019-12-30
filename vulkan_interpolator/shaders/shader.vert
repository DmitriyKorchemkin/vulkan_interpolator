#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPos;
layout(location = 1) in float inColor;

layout(location = 0) out float outColor;


void main() {
  outColor = inColor;
  gl_Position = vec4(inPos, 0., 1.);
}
