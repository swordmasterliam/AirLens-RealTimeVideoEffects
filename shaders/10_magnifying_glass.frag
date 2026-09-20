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

    vec2 rel = v_uv - center;
    vec2 rel_aspect = vec2(rel.x * aspect, rel.y);
    float r = length(rel_aspect);

    float radius = 0.40;
    vec2 zoom_uv = v_uv;

    // Magnification
    if (r < radius) {
        float norm_r = r / radius;
        float z = sqrt(max(0.0, 1.0 - norm_r * norm_r));
        float mag = mix(1.0, 0.55, z * u_intensity);
        zoom_uv = center + rel * mag;
    }

    // Chromatic aberration
    float chrom = 0.015 * (1.0 - smoothstep(0.0, radius, r)) * u_intensity;
    vec2 dir_aspect = r > 0.0001 ? normalize(rel_aspect) : vec2(0.0);
    vec2 chrom_disp = vec2(dir_aspect.x / aspect, dir_aspect.y) * chrom;

    float r_col = texture(u_camera, zoom_uv + chrom_disp).r;
    float g_col = texture(u_camera, zoom_uv).g;
    float b_col = texture(u_camera, zoom_uv - chrom_disp).b;

    vec3 glass_color = vec3(r_col, g_col, b_col);

    // Lens rim highlight
    float rim = smoothstep(radius - 0.02, radius - 0.005, r) * (1.0 - smoothstep(radius - 0.005, radius + 0.005, r));
    glass_color += vec3(0.9, 0.95, 1.0) * rim * 0.8 * clamp(u_intensity, 0.0, 1.0);

    FragColor = mix(base_color, vec4(glass_color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
