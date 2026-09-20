// Common shader utilities

#ifndef U_FINGER_UNIFORMS_DEFINED
#define U_FINGER_UNIFORMS_DEFINED
uniform vec2 u_thumb1;
uniform vec2 u_index1;
uniform vec2 u_thumb2;
uniform vec2 u_index2;

#define MAX_FROZEN_SHAPES 8
uniform int u_num_frozen;
uniform vec4 u_frozen_boxes[MAX_FROZEN_SHAPES];
uniform vec2 u_frozen_pts[MAX_FROZEN_SHAPES * 4];
uniform int u_frozen_is_quad[MAX_FROZEN_SHAPES];
#endif

bool is_inside_box(vec2 uv, vec4 box) {
    return (uv.x >= box.x && uv.x <= box.z && uv.y >= box.y && uv.y <= box.w);
}

float get_box_mask(vec2 uv, vec4 box, float feather) {
    if (!is_inside_box(uv, box)) {
        return 0.0;
    }
    if (feather <= 0.0001) {
        return 1.0;
    }
    float dx = min(uv.x - box.x, box.z - uv.x);
    float dy = min(uv.y - box.y, box.w - uv.y);
    float dist = min(dx, dy);
    return smoothstep(0.0, feather, dist);
}

bool is_valid_point(vec2 pt) {
    return (abs(pt.x) > 0.001 || abs(pt.y) > 0.001);
}

float dist_to_segment(vec2 p, vec2 a, vec2 b) {
    vec2 ba = b - a;
    vec2 pa = p - a;
    float d = dot(ba, ba);
    if (d < 0.000001) {
        return length(pa);
    }
    float h = clamp(dot(pa, ba) / d, 0.0, 1.0);
    return length(pa - ba * h);
}

bool point_in_quad(vec2 p, vec2 p0, vec2 p1, vec2 p2, vec2 p3) {
    int count = 0;
    if ((p0.y > p.y) != (p1.y > p.y)) {
        if (p.x < (p1.x - p0.x) * (p.y - p0.y) / (p1.y - p0.y) + p0.x) count++;
    }
    if ((p1.y > p.y) != (p2.y > p.y)) {
        if (p.x < (p2.x - p1.x) * (p.y - p1.y) / (p2.y - p1.y) + p1.x) count++;
    }
    if ((p2.y > p.y) != (p3.y > p.y)) {
        if (p.x < (p3.x - p2.x) * (p.y - p2.y) / (p3.y - p2.y) + p2.x) count++;
    }
    if ((p3.y > p.y) != (p0.y > p.y)) {
        if (p.x < (p0.x - p3.x) * (p.y - p3.y) / (p0.y - p3.y) + p3.x) count++;
    }
    return (count % 2) == 1;
}

void sort_quad(inout vec2 p0, inout vec2 p1, inout vec2 p2, inout vec2 p3) {
    vec2 c = (p0 + p1 + p2 + p3) * 0.25;
    float a0 = atan(p0.y - c.y, p0.x - c.x);
    float a1 = atan(p1.y - c.y, p1.x - c.x);
    float a2 = atan(p2.y - c.y, p2.x - c.x);
    float a3 = atan(p3.y - c.y, p3.x - c.x);

    float ta;
    vec2 tp;
    if (a0 > a1) { ta = a0; a0 = a1; a1 = ta; tp = p0; p0 = p1; p1 = tp; }
    if (a2 > a3) { ta = a2; a2 = a3; a3 = ta; tp = p2; p2 = p3; p3 = tp; }
    if (a0 > a2) { ta = a0; a0 = a2; a2 = ta; tp = p0; p0 = p2; p2 = tp; }
    if (a1 > a3) { ta = a1; a1 = a3; a3 = ta; tp = p1; p1 = p3; p3 = tp; }
    if (a1 > a2) { ta = a1; a1 = a2; a2 = ta; tp = p1; p1 = p2; p2 = tp; }
}

float get_quad_mask_pts(vec2 uv, vec2 p0, vec2 p1, vec2 p2, vec2 p3, float feather) {
    sort_quad(p0, p1, p2, p3);

    bool inside = point_in_quad(uv, p0, p1, p2, p3);
    if (!inside) {
        return 0.0;
    }

    if (feather <= 0.0001) {
        return 1.0;
    }

    float d0 = dist_to_segment(uv, p0, p1);
    float d1 = dist_to_segment(uv, p1, p2);
    float d2 = dist_to_segment(uv, p2, p3);
    float d3 = dist_to_segment(uv, p3, p0);
    float min_dist = min(min(d0, d1), min(d2, d3));

    return smoothstep(0.0, feather, min_dist);
}

float get_quad_mask(vec2 uv, float feather) {
    return get_quad_mask_pts(uv, u_thumb1, u_index1, u_thumb2, u_index2, feather);
}

// mode: 0 = inside bounds (quadrilateral if 4 fingers, box if 2 fingers), 1 = outside bounds (inverted), 2 = fullscreen
float compute_mask(vec2 uv, vec4 box, float has_box, int mode, float feather) {
    if (mode == 2) {
        return 1.0;
    }

    float combined_mask = 0.0;

    // 1. Evaluate frozen shapes (if any)
    for (int i = 0; i < MAX_FROZEN_SHAPES; i++) {
        if (i >= u_num_frozen) break;
        float m = 0.0;
        if (u_frozen_is_quad[i] == 1) {
            vec2 q0 = u_frozen_pts[i * 4 + 0];
            vec2 q1 = u_frozen_pts[i * 4 + 1];
            vec2 q2 = u_frozen_pts[i * 4 + 2];
            vec2 q3 = u_frozen_pts[i * 4 + 3];
            m = get_quad_mask_pts(uv, q0, q1, q2, q3, feather);
        } else {
            m = get_box_mask(uv, u_frozen_boxes[i], feather);
        }
        combined_mask = max(combined_mask, m);
    }

    // 2. Evaluate active live hand shape (if active)
    if (has_box >= 0.001) {
        float live_mask = 0.0;
        if (is_valid_point(u_thumb1) && is_valid_point(u_index1) &&
            is_valid_point(u_thumb2) && is_valid_point(u_index2)) {
            live_mask = get_quad_mask(uv, feather);
        } else {
            live_mask = get_box_mask(uv, box, feather);
        }
        combined_mask = max(combined_mask, live_mask * has_box);
    }

    if (u_num_frozen <= 0 && has_box < 0.001) {
        return 0.0;
    }

    if (mode == 0) {
        return combined_mask;
    } else {
        float effective_has_shape = max(has_box, u_num_frozen > 0 ? 1.0 : 0.0);
        return (1.0 - combined_mask) * effective_has_shape;
    }
}

mat2 rotate2d(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return mat2(c, -s, s, c);
}

float hash21(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}
