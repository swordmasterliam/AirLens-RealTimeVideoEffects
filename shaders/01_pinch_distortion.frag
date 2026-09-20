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
    vec2 box_size;
    if (u_mask_mode == 2 || u_has_box < 0.5) {
        center = vec2(0.5, 0.5);
        box_size = vec2(1.0, 1.0);
    } else if (is_valid_point(u_thumb1) && is_valid_point(u_index1) &&
               is_valid_point(u_thumb2) && is_valid_point(u_index2)) {
        center = (u_thumb1 + u_index1 + u_thumb2 + u_index2) * 0.25;
        box_size = max(u_hand_box.zw - u_hand_box.xy, vec2(0.01));
    } else {
        center = (u_hand_box.xy + u_hand_box.zw) * 0.5;
        box_size = max(u_hand_box.zw - u_hand_box.xy, vec2(0.01));
    }

    float aspect = u_resolution.x / u_resolution.y;
    vec2 rel = v_uv - center;
    vec2 rel_aspect = vec2(rel.x * aspect, rel.y);
    float dist = length(rel_aspect);

    // Ripple and pinch distortion
    float wave1 = sin(dist * 32.0 - u_time * 5.0);
    float wave2 = cos(dist * 18.0 + u_time * 3.5 + rel.x * 12.0);
    float pinch = 1.0 - smoothstep(0.0, length(box_size * 0.5), dist);

    vec2 dir = dist > 0.0001 ? normalize(rel_aspect) : vec2(0.0);
    float disp_amount = (wave1 * 0.015 + wave2 * 0.012 + pinch * 0.04) * u_intensity;
    vec2 uv_disp = vec2(dir.x / aspect, dir.y) * disp_amount;
    vec2 uv_distorted = v_uv + uv_disp;

    // Chromatic aberration
    float chrom_offset = 0.012 * u_intensity * (0.5 + 0.5 * pinch);
    vec2 chrom_disp = vec2(dir.x / aspect, dir.y) * chrom_offset;
    float r = texture(u_camera, uv_distorted + chrom_disp).r;
    float g = texture(u_camera, uv_distorted).g;
    float b = texture(u_camera, uv_distorted - chrom_disp).b;
    vec4 effect_color = vec4(r, g, b, 1.0);

    float shimmer = (0.5 + 0.5 * sin(u_time * 8.0 + dist * 20.0)) * 0.15 * pinch * clamp(u_intensity, 0.0, 1.0);
    effect_color.rgb += vec3(0.1, 0.35, 0.6) * shimmer;

    FragColor = mix(base_color, effect_color, mask * clamp(u_intensity, 0.0, 1.0));
}
