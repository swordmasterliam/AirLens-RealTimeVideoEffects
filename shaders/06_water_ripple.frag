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

    vec2 p = v_uv * 12.0;
    float t = u_time * 2.5;

    // Wave pattern
    float wave = sin(p.x * 2.0 + t) * cos(p.y * 1.8 + t * 0.8)
               + sin(p.x * 3.5 - t * 1.2 + p.y * 2.0) * 0.5
               + cos(length(p - 6.0) * 4.0 - t * 2.0) * 0.3;

    // Normal estimation
    vec2 normal = vec2(
        cos(p.x * 2.0 + t) * cos(p.y * 1.8 + t * 0.8) * 2.0,
        sin(p.x * 2.0 + t) * -sin(p.y * 1.8 + t * 0.8) * 1.8
    ) * 0.015 * u_intensity;

    float aspect = u_resolution.x / u_resolution.y;
    vec2 uv_offset = vec2(normal.x / aspect, normal.y);
    vec2 refracted_uv = v_uv + uv_offset;

    // Chromatic offset
    float r = texture(u_camera, refracted_uv + uv_offset * 0.2).r;
    float g = texture(u_camera, refracted_uv).g;
    float b = texture(u_camera, refracted_uv - uv_offset * 0.2).b;

    vec3 water_color = vec3(r, g, b);

    // Specular highlight
    float specular = pow(clamp(wave * 0.4 + 0.5, 0.0, 1.0), 6.0) * 0.4 * u_intensity;
    water_color += vec3(0.4, 0.7, 1.0) * specular;

    // Color tint
    water_color = mix(water_color, water_color * vec3(0.85, 0.95, 1.1), 0.3 * clamp(u_intensity, 0.0, 1.0));

    FragColor = mix(base_color, vec4(water_color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
