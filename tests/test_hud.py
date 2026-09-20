"""Unit tests for HUD overlay and reticle rendering logic."""

import math
import unittest
from types import SimpleNamespace
from tracker import TrackingData
from hud import HudRenderer


class TestHudReticle(unittest.TestCase):
    """Tests for reticle geometry generation: 4 fingers connected vs 2 fingers box."""

    def setUp(self):
        # Create a HudRenderer instance with OpenGL initialization bypassed
        self.hud = object.__new__(HudRenderer)
        self.hud.visible = True
        self.hud.show_box_visual = True

    def test_two_hands_four_fingers_connected_loop(self):
        """When 2 hands (4 fingers) are tracked, lines should connect the 4 fingers in a closed loop."""
        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=(0.2, 0.7),   # bottom-left
            index1=(0.2, 0.3),   # top-left
            thumb2=(0.8, 0.7),   # bottom-right
            index2=(0.8, 0.3),   # top-right
            box=(0.2, 0.3, 0.8, 0.7),
        )

        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        # 4 lines, each with 2 vertices -> 8 vertices total
        self.assertEqual(len(lines), 8)

        # Extract segments
        segments = [
            ((lines[0][0], lines[0][1]), (lines[1][0], lines[1][1])),
            ((lines[2][0], lines[2][1]), (lines[3][0], lines[3][1])),
            ((lines[4][0], lines[4][1]), (lines[5][0], lines[5][1])),
            ((lines[6][0], lines[6][1]), (lines[7][0], lines[7][1])),
        ]

        # Verify segments form a closed loop
        self.assertEqual(segments[0][1], segments[1][0])
        self.assertEqual(segments[1][1], segments[2][0])
        self.assertEqual(segments[2][1], segments[3][0])
        self.assertEqual(segments[3][1], segments[0][0])

        # Verify each of the 4 finger tips is present in the loop
        loop_points = {segments[0][0], segments[1][0], segments[2][0], segments[3][0]}
        expected_points = {state.thumb1, state.index1, state.thumb2, state.index2}
        self.assertEqual(loop_points, expected_points)

    def test_mock_tracker_data_connects_four_fingers(self):
        """Verify tracker mock gesture coordinates produce 4 connected finger lines."""
        t1 = (0.25, 0.65)
        i1 = (0.25, 0.35)
        t2 = (0.75, 0.35)
        i2 = (0.75, 0.65)

        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=t1,
            index1=i1,
            thumb2=t2,
            index2=i2,
            box=(0.25, 0.35, 0.75, 0.65),
        )

        lines = self.hud._compute_reticle_lines(state, alpha=0.8)
        self.assertEqual(len(lines), 8)

        # Check line alpha matches scaled alpha
        expected_alpha = 0.8 * 0.85
        self.assertAlmostEqual(lines[0][5], expected_alpha, places=4)

        # Ensure all 4 points are visited in angular order without self-crossing
        visited = [ (lines[i][0], lines[i][1]) for i in (0, 2, 4, 6) ]
        self.assertEqual(len(set(visited)), 4)
        for pt in [t1, i1, t2, i2]:
            self.assertIn(pt, visited)

    def test_rotated_and_arbitrary_quadrilateral(self):
        """Verify that rotated hand positions form a closed loop connecting all 4 fingers."""
        # Diamond configuration
        top = (0.5, 0.1)
        right = (0.8, 0.5)
        bottom = (0.5, 0.9)
        left = (0.2, 0.5)

        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=left,
            index1=top,
            thumb2=bottom,
            index2=right,
        )

        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        self.assertEqual(len(lines), 8)

        visited = [(lines[0][0], lines[0][1]), (lines[2][0], lines[2][1]),
                   (lines[4][0], lines[4][1]), (lines[6][0], lines[6][1])]
        self.assertEqual(set(visited), {top, right, bottom, left})

    def test_one_hand_two_fingers_draws_box(self):
        """When 1 hand (2 fingers) is tracked, it should draw just a box matching state.box."""
        box = (0.15, 0.20, 0.60, 0.70)
        state = TrackingData(
            has_box=1.0,
            hands_count=1,
            thumb1=(0.25, 0.60),
            index1=(0.30, 0.25),
            thumb2=(0.0, 0.0),
            index2=(0.0, 0.0),
            box=box,
        )

        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        self.assertEqual(len(lines), 8)

        x0, y0, x1, y1 = box
        expected_segments = [
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
        ]

        actual_segments = [
            ((lines[0][0], lines[0][1]), (lines[1][0], lines[1][1])),
            ((lines[2][0], lines[2][1]), (lines[3][0], lines[3][1])),
            ((lines[4][0], lines[4][1]), (lines[5][0], lines[5][1])),
            ((lines[6][0], lines[6][1]), (lines[7][0], lines[7][1])),
        ]

        self.assertEqual(actual_segments, expected_segments)

    def test_fade_out_retains_two_hand_polygon_when_hands_lost(self):
        """When hands_count becomes 0 after tracking 2 hands, 4 fingers are still connected during fade."""
        state = TrackingData(
            has_box=0.6,
            hands_count=0,
            thumb1=(0.2, 0.7),
            index1=(0.2, 0.3),
            thumb2=(0.8, 0.7),
            index2=(0.8, 0.3),
            box=(0.2, 0.3, 0.8, 0.7),
        )

        lines = self.hud._compute_reticle_lines(state, alpha=0.6)
        self.assertEqual(len(lines), 8)

        # Should still be the 4 finger vertices, not the box
        pts = {(lines[0][0], lines[0][1]), (lines[2][0], lines[2][1]),
               (lines[4][0], lines[4][1]), (lines[6][0], lines[6][1])}
        self.assertEqual(pts, {state.thumb1, state.index1, state.thumb2, state.index2})
        self.assertAlmostEqual(lines[0][5], 0.6 * 0.85, places=4)

    def test_fade_out_retains_one_hand_box_when_hands_lost(self):
        """When hands_count becomes 0 after tracking 1 hand, box is still drawn during fade."""
        box = (0.2, 0.3, 0.6, 0.7)
        state = TrackingData(
            has_box=0.4,
            hands_count=0,
            thumb1=(0.3, 0.5),
            index1=(0.4, 0.3),
            thumb2=(0.0, 0.0),
            index2=(0.0, 0.0),
            box=box,
        )

        lines = self.hud._compute_reticle_lines(state, alpha=0.4)
        self.assertEqual(len(lines), 8)
        self.assertEqual(lines[0][0], box[0])  # x0
        self.assertEqual(lines[0][1], box[1])  # y0
        self.assertEqual(lines[1][0], box[2])  # x1
        self.assertEqual(lines[1][1], box[1])  # y0

    def test_default_tracking_data_falls_back_to_box(self):
        """Default TrackingData with 0 fingers falls back to box cleanly."""
        state = TrackingData(has_box=1.0)
        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        self.assertEqual(len(lines), 8)
        # Matches default box (0.25, 0.25, 0.75, 0.75)
        self.assertEqual((lines[0][0], lines[0][1]), (0.25, 0.25))
        self.assertEqual((lines[1][0], lines[1][1]), (0.75, 0.25))

    def test_compute_reticle_points_four_fingers(self):
        """4 fingers produce 4 fingertip highlight points with respective colors."""
        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=(0.2, 0.7),
            index1=(0.2, 0.3),
            thumb2=(0.8, 0.7),
            index2=(0.8, 0.3),
        )
        pts = self.hud._compute_reticle_points(state, alpha=1.0)
        self.assertEqual(len(pts), 4)

        # Hand 1: gold (1.0, 0.85, 0.2)
        self.assertEqual(pts[0][:2], (0.2, 0.7))
        self.assertAlmostEqual(pts[0][2], 1.0)
        self.assertAlmostEqual(pts[0][3], 0.85)

        # Hand 2: cyan (0.0, 0.9, 1.0)
        self.assertEqual(pts[2][:2], (0.8, 0.7))
        self.assertAlmostEqual(pts[2][2], 0.0)
        self.assertAlmostEqual(pts[2][3], 0.9)

    def test_compute_reticle_points_two_fingers(self):
        """2 fingers (1 hand) produce only 2 fingertip highlight points."""
        state = TrackingData(
            has_box=1.0,
            hands_count=1,
            thumb1=(0.2, 0.7),
            index1=(0.2, 0.3),
            thumb2=(0.0, 0.0),
            index2=(0.0, 0.0),
        )
        pts = self.hud._compute_reticle_points(state, alpha=1.0)
        self.assertEqual(len(pts), 2)
        self.assertEqual(pts[0][:2], (0.2, 0.7))
        self.assertEqual(pts[1][:2], (0.2, 0.3))

    def test_two_hands_missing_finger_falls_back_to_box(self):
        """When 2 hands are detected but one fingertip is missing (0, 0), fallback to box."""
        box = (0.2, 0.2, 0.8, 0.8)
        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=(0.2, 0.7),
            index1=(0.2, 0.3),
            thumb2=(0.0, 0.0),  # missing / undetected thumb
            index2=(0.8, 0.3),
            box=box,
        )
        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        self.assertEqual(len(lines), 8)
        # Should be the bounding box
        self.assertEqual((lines[0][0], lines[0][1]), (box[0], box[1]))
        self.assertEqual((lines[1][0], lines[1][1]), (box[2], box[1]))

    def test_negative_coordinates_valid_fingertips(self):
        """Fingertips slightly past the screen boundary (negative coords) are treated as valid."""
        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=(-0.05, 0.7),
            index1=(-0.05, 0.3),
            thumb2=(0.8, 0.7),
            index2=(0.8, 0.3),
        )
        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        self.assertEqual(len(lines), 8)
        pts = {(lines[0][0], lines[0][1]), (lines[2][0], lines[2][1]),
               (lines[4][0], lines[4][1]), (lines[6][0], lines[6][1])}
        self.assertEqual(pts, {state.thumb1, state.index1, state.thumb2, state.index2})

        # Check points computation also includes negative coords
        pt_highlights = self.hud._compute_reticle_points(state, alpha=1.0)
        self.assertEqual(len(pt_highlights), 4)

    def test_collinear_same_angle_points_radial_sorting(self):
        """Points sharing identical polar angles from centroid are deterministically sorted by distance."""
        # Centroid will be at (0.5, 0.5)
        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=(0.6, 0.5),   # Angle 0, dist 0.1
            index1=(0.8, 0.5),   # Angle 0, dist 0.3
            thumb2=(0.3, 0.8),   # Angle > 0
            index2=(0.3, 0.2),   # Angle < 0
        )
        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        self.assertEqual(len(lines), 8)
        # Segments form a closed loop
        for i in range(0, 8, 2):
            next_idx = (i + 2) % 8
            self.assertEqual((lines[i+1][0], lines[i+1][1]), (lines[next_idx][0], lines[next_idx][1]))

    def test_pinch_coincident_fingers(self):
        """When thumb and index tip touch (coincident coordinates), geometry doesn't error."""
        state = TrackingData(
            has_box=1.0,
            hands_count=2,
            thumb1=(0.3, 0.5),
            index1=(0.3, 0.5),   # pinched together with thumb1
            thumb2=(0.8, 0.7),
            index2=(0.8, 0.3),
        )
        lines = self.hud._compute_reticle_lines(state, alpha=1.0)
        self.assertEqual(len(lines), 8)

    def test_malformed_state_safe(self):
        """None or incomplete attributes in tracking state do not raise exceptions."""
        mock_state = SimpleNamespace(
            has_box=1.0,
            hands_count=1,
            thumb1=None,
            index1=(0.3, 0.4),
            thumb2=None,
            index2=None,
            box=None
        )
        lines = self.hud._compute_reticle_lines(mock_state, alpha=1.0)
        self.assertEqual(len(lines), 8)
        # Falls back to default box
        self.assertEqual((lines[0][0], lines[0][1]), (0.25, 0.25))

    def test_render_gpu_reticle_null_safety(self):
        """_render_gpu_reticle safely returns if OpenGL resources are missing."""
        self.hud._reticle_prog = 0
        self.hud._reticle_vao = 0
        self.hud._reticle_vbo = 0
        # Should return without raising any exception
        self.hud._render_gpu_reticle(TrackingData(has_box=1.0))

    def test_hud_high_dpi_scale_button_rect(self):
        """Verify HudRenderer set_scale scales _btn_rect for High-DPI screens (BUG-03)."""
        self.hud._base_btn_rect = (200, 10, 435, 38)
        self.hud._btn_rect = (200, 10, 435, 38)
        self.hud._scale_x = 1.0
        self.hud._scale_y = 1.0

        self.hud.set_scale(1.5, 1.5)
        self.assertEqual(self.hud._btn_rect, (300.0, 15.0, 652.5, 57.0))
        self.assertTrue(self.hud.is_gesture_button_clicked(400.0, 30.0))
        self.assertFalse(self.hud.is_gesture_button_clicked(250.0, 30.0))


if __name__ == "__main__":
    unittest.main()
