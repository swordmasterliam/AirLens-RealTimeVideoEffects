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

void main() {
    vec4 base_color = texture(u_camera, v_uv);
    float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);

    if (mask <= 0.001 || u_intensity <= 0.001) {
        FragColor = base_color;
        return;
    }

    vec2 uv = v_uv;

    // Glitch slice offset
    float slice_y = floor(uv.y * 30.0);
    float slice_noise = hash21(vec2(slice_y, floor(u_time * 12.0)));

    float glitch_trigger = step(0.70, slice_noise);
    float shift_x = (slice_noise - 0.5) * 0.06 * glitch_trigger * u_intensity;

    // Scanlines
    float scanline = sin(uv.y * u_resolution.y * 1.5) * 0.12 * u_intensity;

    // RGB split
    vec2 uv_r = vec2(uv.x + shift_x + 0.015 * u_intensity, uv.y);
    vec2 uv_g = vec2(uv.x + shift_x * 0.5, uv.y);
    vec2 uv_b = vec2(uv.x - shift_x - 0.012 * u_intensity, uv.y);

    float r = texture(u_camera, uv_r).r;
    float g = texture(u_camera, uv_g).g;
    float b = texture(u_camera, uv_b).b;

    vec3 glitch_color = vec3(r, g, b) - vec3(scanline);

    // Noise grain
    float grain = (hash21(uv * 500.0 + fract(u_time * 43.0)) - 0.5) * 0.12 * u_intensity;
    glitch_color += grain;

    glitch_color.r = mix(glitch_color.r, glitch_color.r * 1.1, clamp(u_intensity, 0.0, 1.0));
    glitch_color.b = mix(glitch_color.b, glitch_color.b * 1.2, clamp(u_intensity, 0.0, 1.0));

    FragColor = mix(base_color, vec4(glitch_color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
