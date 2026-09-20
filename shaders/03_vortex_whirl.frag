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

    float radius = 0.45;
    vec2 swirl_uv = v_uv;

    if (r < radius) {
        float percent = (radius - r) / radius;
        float theta = percent * percent * 8.0 * u_intensity;
        float s = sin(theta);
        float c = cos(theta);
        vec2 rotated = vec2(rel_aspect.x * c - rel_aspect.y * s, rel_aspect.x * s + rel_aspect.y * c);
        swirl_uv = center + vec2(rotated.x / aspect, rotated.y);
    }

    // Chromatic aberration
    float chrom = 0.008 * (1.0 - smoothstep(0.0, radius, r)) * u_intensity;
    vec2 dir_aspect = r > 0.0001 ? normalize(rel_aspect) : vec2(0.0);
    vec2 dir = vec2(dir_aspect.x / aspect, dir_aspect.y);

    float cr = texture(u_camera, swirl_uv + dir * chrom).r;
    float cg = texture(u_camera, swirl_uv).g;
    float cb = texture(u_camera, swirl_uv - dir * chrom).b;

    vec3 vortex_color = vec3(cr, cg, cb);

    // Center ring glow
    float accretion = smoothstep(0.05, 0.18, r) * (1.0 - smoothstep(0.18, 0.35, r));
    vortex_color += vec3(0.8, 0.4, 1.0) * accretion * 0.6 * u_intensity;

    FragColor = mix(base_color, vec4(vortex_color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
