#version 330 core

out vec4 FragColor;
in vec2 v_uv;

uniform sampler2D u_camera;
uniform vec2 u_resolution;
uniform float u_time;
uniform vec4 u_hand_box;
uniform float u_has_box;
uniform int u_mask_mode;
uniform float u_intensity;

#include "common.glsl"

float get_luma(vec2 uv) {
    vec3 c = texture(u_camera, uv).rgb;
    return dot(c, vec3(0.299, 0.587, 0.114));
}

void main() {
    vec4 base_color = texture(u_camera, v_uv);
    float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);

    if (mask <= 0.001 || u_intensity <= 0.001) {
        FragColor = base_color;
        return;
    }

    vec2 step_size = 1.5 / u_resolution;

    // Sobel filter
    float tl = get_luma(v_uv + vec2(-step_size.x, -step_size.y));
    float tc = get_luma(v_uv + vec2(0.0, -step_size.y));
    float tr = get_luma(v_uv + vec2(step_size.x, -step_size.y));
    float ml = get_luma(v_uv + vec2(-step_size.x, 0.0));
    float mr = get_luma(v_uv + vec2(step_size.x, 0.0));
    float bl = get_luma(v_uv + vec2(-step_size.x, step_size.y));
    float bc = get_luma(v_uv + vec2(0.0, step_size.y));
    float br = get_luma(v_uv + vec2(step_size.x, step_size.y));

    float gx = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
    float gy = (bl + 2.0 * bc + br) - (tl + 2.0 * tc + tr);
    float edge = length(vec2(gx, gy)) * 3.5 * u_intensity;

    // Color gradient
    vec3 neon_tint = 0.5 + 0.5 * cos(u_time * 2.0 + v_uv.xyx * 3.0 + vec3(0.0, 2.0, 4.0));
    vec3 neon_color = neon_tint * edge;

    // Dim background
    neon_color += base_color.rgb * mix(1.0, 0.15, clamp(u_intensity, 0.0, 1.0));

    FragColor = mix(base_color, vec4(neon_color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
