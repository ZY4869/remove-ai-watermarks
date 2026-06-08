"""Tests for the GFPGAN face-restoration post-pass.

The pure feather-composite helper is unit-tested without the model; the
model-running paths are gated behind ``is_available()`` (a multi-hundred-MB
download), matching the discipline used for the other ML-adjacent modules.
"""

from __future__ import annotations

import numpy as np
import pytest

from remove_ai_watermarks import face_restore


class TestIsAvailable:
    def test_returns_bool(self):
        assert isinstance(face_restore.is_available(), bool)

    def test_reflects_dependencies(self):
        import importlib.util

        expected = all(importlib.util.find_spec(m) is not None for m in ("gfpgan", "facexlib"))
        assert face_restore.is_available() is expected


class TestCompositeFaces:
    """Unit tests for the pure ``_composite_faces`` helper (cv2/numpy only)."""

    def _base_and_restored(self, h: int = 100, w: int = 120):
        base = np.zeros((h, w, 3), dtype=np.uint8)  # black
        restored = np.full((h, w, 3), 255, dtype=np.uint8)  # white
        return base, restored

    def test_output_shape_and_dtype(self):
        base, restored = self._base_and_restored()
        out = face_restore._composite_faces(base, restored, [(40.0, 30.0, 80.0, 70.0)])
        assert out.shape == base.shape
        assert out.dtype == np.uint8

    def test_box_region_pulls_toward_restored(self):
        base, restored = self._base_and_restored()
        out = face_restore._composite_faces(base, restored, [(40.0, 30.0, 80.0, 70.0)])
        # Center of the box should be near the restored (white) value.
        cy, cx = 50, 60
        assert out[cy, cx].mean() > 200

    def test_far_from_box_stays_base(self):
        base, restored = self._base_and_restored()
        out = face_restore._composite_faces(base, restored, [(40.0, 30.0, 80.0, 70.0)], pad=2)
        # Top-left corner is far from the box and feather, so it stays black.
        assert out[0, 0].mean() < 5

    def test_empty_boxes_returns_base_unchanged(self):
        base, restored = self._base_and_restored()
        out = face_restore._composite_faces(base, restored, [])
        assert np.array_equal(out, base)

    def test_box_fully_outside_is_skipped(self):
        base, restored = self._base_and_restored(h=100, w=120)
        # Box entirely beyond the right/bottom edge -> clipped to empty -> no-op.
        out = face_restore._composite_faces(base, restored, [(200.0, 200.0, 260.0, 260.0)], pad=0)
        assert np.array_equal(out, base)

    def test_near_edge_box_clips_without_error(self):
        base, restored = self._base_and_restored(h=100, w=120)
        # Box reaching past the bottom-right corner must clip, not raise.
        out = face_restore._composite_faces(base, restored, [(100.0, 80.0, 130.0, 110.0)], pad=10)
        assert out.shape == base.shape
        # The clipped in-bounds region still pulls toward white.
        assert out[95, 115].mean() > 100


@pytest.mark.skipif(not face_restore.is_available(), reason="requires the 'restore' extra (gfpgan/facexlib)")
class TestRestoreFacesModel:
    """Model-running smoke test, gated behind the optional extra."""

    def test_no_faces_returns_cleaned_unchanged(self):
        # A flat gray image has no faces; restore_faces must return the cleaned
        # input unchanged (the no-op path).
        cleaned = np.full((128, 128, 3), 127, dtype=np.uint8)
        original = np.full((128, 128, 3), 127, dtype=np.uint8)
        out = face_restore.restore_faces(original, cleaned)
        assert out.shape == cleaned.shape
        assert np.array_equal(out, cleaned)
