#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec2 inPos;
layout(location = 1) in float inColor;

layout(push_constant) uniform Transform {
  float sx;
  float sy;
  float tx;
  float ty;
} tform;

layout(location = 0) out float outColor;


void main() {
  outColor = inColor;
 gl_Position = vec4(inPos.x*tform.sx + tform.tx, inPos.y*tform.sy + tform.ty, 0., 1.);
}
