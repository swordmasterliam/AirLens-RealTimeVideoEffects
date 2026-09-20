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

    // Pixel grid
    float pixel_size = mix(6.0, 36.0, clamp(u_intensity * 0.5, 0.1, 2.0));
    vec2 grid_res = u_resolution / pixel_size;
    vec2 block_uv = (floor(v_uv * grid_res) + 0.5) / grid_res;

    vec3 pixel_color = texture(u_camera, block_uv).rgb;

    // Quantize colors
    float levels = 6.0;
    pixel_color = floor(pixel_color * levels + 0.5) / levels;

    // Phosphor pattern
    vec2 frag_coord = v_uv * u_resolution;
    int subpixel = int(mod(frag_coord.x, 3.0));
    vec3 phosphor = vec3(0.85);
    if (subpixel == 0) phosphor.r = 1.25;
    else if (subpixel == 1) phosphor.g = 1.25;
    else phosphor.b = 1.25;

    pixel_color *= phosphor;

    // Grid lines
    vec2 block_frac = fract(v_uv * grid_res);
    float border = step(0.06, block_frac.x) * step(0.06, block_frac.y);
    pixel_color *= mix(0.7, 1.0, border);

    FragColor = mix(base_color, vec4(pixel_color, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
