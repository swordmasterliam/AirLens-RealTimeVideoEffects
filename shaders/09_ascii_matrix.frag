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

// Procedural glyph generator
float char_pattern(float n, vec2 p) {
    p = clamp(p, 0.0, 1.0);
    vec2 grid = floor(p * 5.0);
    float bit = hash21(grid + vec2(n * 17.13, n * 31.41));
    return step(0.45, bit);
}

void main() {
    vec4 base_color = texture(u_camera, v_uv);
    float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);

    if (mask <= 0.001 || u_intensity <= 0.001) {
        FragColor = base_color;
        return;
    }

    vec2 cell_size = vec2(10.0, 14.0);
    vec2 cells = u_resolution / cell_size;
    vec2 cell_id = floor(v_uv * cells);
    vec2 cell_uv = fract(v_uv * cells);

    // Cell luminance
    vec2 sample_uv = (cell_id + 0.5) / cells;
    vec3 c = texture(u_camera, sample_uv).rgb;
    float lum = dot(c, vec3(0.299, 0.587, 0.114));

    // Rain drop animation
    float col_rand = hash21(vec2(cell_id.x, 123.45));
    float rain_y = fract(u_time * (0.8 + col_rand * 1.5) + col_rand);
    float rain_dist = fract(v_uv.y - rain_y);
    float rain_tail = 1.0 - smoothstep(0.0, 0.4, rain_dist);

    // Cycle characters
    float char_id = floor(hash21(cell_id + floor(u_time * 4.0)) * 16.0);
    float glyph = char_pattern(char_id, cell_uv);

    // Palette
    vec3 matrix_green = vec3(0.1, 0.95, 0.25);
    vec3 matrix_white = vec3(0.8, 1.0, 0.85);

    vec3 matrix_color = mix(matrix_green, matrix_white, step(0.92, 1.0 - rain_dist));
    matrix_color *= glyph * (lum * 1.3 + 0.2) * (0.4 + 0.6 * rain_tail) * u_intensity;

    // Background tone
    matrix_color += vec3(0.0, 0.04, 0.01) * lum * clamp(u_intensity, 0.0, 1.0);

    FragColor = mix(base_color, vec4(matrix_color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
