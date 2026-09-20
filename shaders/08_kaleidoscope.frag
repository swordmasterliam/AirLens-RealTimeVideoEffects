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

    vec2 center;
    if (u_mask_mode == 2 || u_has_box < 0.5) {
        center = vec2(0.5, 0.5);
    } else if (is_valid_point(u_thumb1) && is_valid_point(u_index1) &&
               is_valid_point(u_thumb2) && is_valid_point(u_index2)) {
        center = (u_thumb1 + u_index1 + u_thumb2 + u_index2) * 0.25;
    } else {
        center = (u_hand_box.xy + u_hand_box.zw) * 0.5;
    }
    float aspect = u_resolution.x / u_resolution.y;

    vec2 p = v_uv - center;
    p.x *= aspect;

    // Polar coordinates
    float r = length(p);
    float a = atan(p.y, p.x);

    // Segments
    float segments = 8.0;
    float segment_angle = 3.14159265 * 2.0 / segments;

    // Rotation
    a += u_time * 0.3 * u_intensity;

    // Fold angle
    a = mod(a, segment_angle);
    a = abs(a - segment_angle * 0.5);

    // Cartesian coordinates
    vec2 kaleido_p = vec2(cos(a), sin(a)) * r;
    kaleido_p.x /= aspect;
    vec2 kaleido_uv = fract(center + kaleido_p * (1.0 + 0.5 * sin(u_time * 0.5)));

    // Mirror repeat
    kaleido_uv = 1.0 - abs(fract(kaleido_uv * 0.5) * 2.0 - 1.0);

    vec3 color = texture(u_camera, kaleido_uv).rgb;

    // Boost saturation
    float luma = dot(color, vec3(0.299, 0.587, 0.114));
    color = mix(vec3(luma), color, 1.4);

    FragColor = mix(base_color, vec4(color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
