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

vec3 thermal_color(float val) {
    // Thermal color ramp
    val = clamp(val, 0.0, 1.0);
    vec3 c0 = vec3(0.05, 0.0, 0.2);
    vec3 c1 = vec3(0.0, 0.2, 0.9);
    vec3 c2 = vec3(0.0, 0.9, 0.6);
    vec3 c3 = vec3(0.9, 0.9, 0.0);
    vec3 c4 = vec3(1.0, 0.2, 0.0);
    vec3 c5 = vec3(1.0, 1.0, 1.0);

    if (val < 0.2) return mix(c0, c1, val / 0.2);
    if (val < 0.4) return mix(c1, c2, (val - 0.2) / 0.2);
    if (val < 0.7) return mix(c2, c3, (val - 0.4) / 0.3);
    if (val < 0.9) return mix(c3, c4, (val - 0.7) / 0.2);
    return mix(c4, c5, (val - 0.9) / 0.1);
}

void main() {
    vec4 base_color = texture(u_camera, v_uv);
    float mask = compute_mask(v_uv, u_hand_box, u_has_box, u_mask_mode, 0.02);

    if (mask <= 0.001 || u_intensity <= 0.001) {
        FragColor = base_color;
        return;
    }

    // Luminance / heat approximation
    float lum = dot(base_color.rgb, vec3(0.299, 0.587, 0.114));
    lum = pow(lum, 1.0 / (0.8 + 0.4 * u_intensity));

    // Contrast boost
    lum = clamp((lum - 0.1) * 1.25, 0.0, 1.0);

    vec3 heat = thermal_color(lum);

    // Sensor noise
    float noise = (hash21(v_uv * 600.0 + fract(u_time * 17.0)) - 0.5) * 0.05 * clamp(u_intensity, 0.0, 1.0);
    heat += noise;

    FragColor = mix(base_color, vec4(heat, 1.0), mask * clamp(u_intensity, 0.0, 1.0));
}
