"""Tests for the Gemini visible-watermark engine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.gemini_engine import (
    DetectionResult,
    GeminiEngine,
    WatermarkPosition,
    WatermarkSize,
    _calculate_alpha_map,
    detect_sparkle_confidence,
    get_watermark_config,
    get_watermark_size,
)

# ── WatermarkSize / config helpers ──────────────────────────────────


class TestWatermarkConfig:
    """Tests for watermark size detection and position calculation."""

    def test_small_image_gets_small_watermark(self):
        assert get_watermark_size(800, 600) == WatermarkSize.SMALL

    def test_large_image_gets_large_watermark(self):
        assert get_watermark_size(1920, 1080) == WatermarkSize.LARGE

    def test_boundary_image_stays_small(self):
        """Exactly 1024x1024 should be SMALL (rule: > 1024 for LARGE)."""
        assert get_watermark_size(1024, 1024) == WatermarkSize.SMALL

    def test_one_dimension_small(self):
        """Only one dimension > 1024 → still SMALL."""
        assert get_watermark_size(2000, 500) == WatermarkSize.SMALL

    def test_config_small_returns_correct_values(self):
        config = get_watermark_config(800, 600)
        assert config.margin_right == 32
        assert config.margin_bottom == 32
        assert config.logo_size == 48

    def test_config_large_returns_correct_values(self):
        config = get_watermark_config(1920, 1080)
        assert config.margin_right == 64
        assert config.margin_bottom == 64
        assert config.logo_size == 96

    def test_position_calculation(self):
        pos = WatermarkPosition(margin_right=32, margin_bottom=32, logo_size=48)
        x, y = pos.get_position(800, 600)
        assert x == 800 - 32 - 48  # 720
        assert y == 600 - 32 - 48  # 520


# ── Alpha map ───────────────────────────────────────────────────────


class TestAlphaMap:
    """Tests for alpha map calculation."""

    def test_pure_black_gives_zero_alpha(self):
        black = np.zeros((10, 10, 3), dtype=np.uint8)
        alpha = _calculate_alpha_map(black)
        assert alpha.shape == (10, 10)
        np.testing.assert_array_equal(alpha, 0.0)

    def test_pure_white_gives_one_alpha(self):
        white = np.full((10, 10, 3), 255, dtype=np.uint8)
        alpha = _calculate_alpha_map(white)
        np.testing.assert_allclose(alpha, 1.0)

    def test_grayscale_input(self):
        gray = np.full((10, 10), 128, dtype=np.uint8)
        alpha = _calculate_alpha_map(gray)
        np.testing.assert_allclose(alpha, 128 / 255.0)

    def test_max_channel_used(self):
        """Alpha should use max(R, G, B)."""
        img = np.zeros((1, 1, 3), dtype=np.uint8)
        img[0, 0] = [50, 200, 100]  # BGR
        alpha = _calculate_alpha_map(img)
        assert pytest.approx(alpha[0, 0], rel=1e-3) == 200 / 255.0


# ── GeminiEngine ────────────────────────────────────────────────────


class TestGeminiEngine:
    """Tests for the GeminiEngine class."""

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def test_engine_loads_alpha_maps(self):
        small = self.engine.get_alpha_map(WatermarkSize.SMALL)
        large = self.engine.get_alpha_map(WatermarkSize.LARGE)
        assert small.shape == (48, 48)
        assert large.shape == (96, 96)

    def test_remove_watermark_returns_same_shape(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark(image)
        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_remove_watermark_does_not_modify_input(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        original = image.copy()
        self.engine.remove_watermark(image)
        np.testing.assert_array_equal(image, original)

    def test_remove_watermark_large_image(self, tmp_large_image_path):
        image = cv2.imread(str(tmp_large_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark(image)
        assert result.shape == image.shape

    def test_remove_watermark_custom_region(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark_custom(image, (10, 10, 48, 48))
        assert result.shape == image.shape

    def test_detect_on_grayscale_does_not_crash(self):
        # A 2D grayscale array reaching detect_watermark (registry adapter / library
        # API) must not crash the FP-gate's axis=2 reduction; it is normalized to BGR.
        gray = np.full((300, 300), 100, dtype=np.uint8)
        result = self.engine.detect_watermark(gray)
        assert result is not None

    def test_remove_on_bgra_returns_3_channel(self):
        bgra = np.zeros((300, 300, 4), dtype=np.uint8)
        bgra[..., 3] = 255
        result = self.engine.remove_watermark(bgra)
        assert result.shape == (300, 300, 3)

    def test_remove_watermark_custom_large_region(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark_custom(image, (10, 10, 96, 96))
        assert result.shape == image.shape

    def test_remove_watermark_custom_arbitrary_region(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark_custom(image, (5, 5, 60, 60))
        assert result.shape == image.shape

    def test_force_size(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.remove_watermark(image, force_size=WatermarkSize.LARGE)
        assert result.shape == image.shape


# ── Detection ───────────────────────────────────────────────────────


class TestDetection:
    """Tests for watermark detection."""

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def test_detect_returns_result_object(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.detect_watermark(image)
        assert isinstance(result, DetectionResult)
        assert 0.0 <= result.confidence <= 1.0

    def test_detect_empty_image_returns_no_detection(self):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = self.engine.detect_watermark(empty)
        assert not result.detected
        assert result.confidence == 0.0

    def test_detect_none_image_returns_no_detection(self):
        result = self.engine.detect_watermark(None)
        assert not result.detected

    def test_detect_random_image_low_confidence(self, tmp_image_path):
        """Random noise should not look like a watermark."""
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.detect_watermark(image)
        # Random image may or may not be detected; confidence should be meaningful
        assert isinstance(result.spatial_score, float)
        assert isinstance(result.gradient_score, float)


class TestDetectSparkleConfidence:
    """File-level entry point used by identify.py."""

    def test_returns_float_in_range_for_real_image(self, tmp_image_path):
        conf = detect_sparkle_confidence(tmp_image_path)
        assert conf is not None
        assert 0.0 <= conf <= 1.0

    def test_returns_none_for_unreadable_file(self, tmp_path):
        # cv2.imread returns None for a non-image; the helper must not raise.
        bogus = tmp_path / "not_an_image.png"
        bogus.write_bytes(b"this is not a PNG")
        assert detect_sparkle_confidence(bogus) is None


# ── Inpainting ──────────────────────────────────────────────────────


class TestInpainting:
    """Tests for residual inpainting."""

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def test_inpaint_ns(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), method="ns")
        assert result.shape == image.shape

    def test_inpaint_telea(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), method="telea")
        assert result.shape == image.shape

    def test_inpaint_gaussian(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), method="gaussian")
        assert result.shape == image.shape

    def test_inpaint_zero_strength(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (150, 150, 48, 48), strength=0.0)
        np.testing.assert_array_equal(result, image)

    def test_inpaint_tiny_region_returns_unchanged(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        result = self.engine.inpaint_residual(image, (10, 10, 2, 2))
        np.testing.assert_array_equal(result, image)

    def test_inpaint_does_not_modify_input(self, tmp_image_path):
        image = cv2.imread(str(tmp_image_path), cv2.IMREAD_COLOR)
        original = image.copy()
        self.engine.inpaint_residual(image, (150, 150, 48, 48))
        np.testing.assert_array_equal(image, original)


class TestOverSubtractionGuard:
    """Issue #30: reverse-alpha must not turn the sparkle into a black pit.

    On a dark background the captured alpha over-estimates the real sparkle opacity,
    so the fixed-alpha reverse blend over-subtracts and drives the footprint to black.
    The engine detects this and inpaints the footprint instead.
    """

    # Composite the mark at ~60% of the captured opacity: the engine's alpha maxes at
    # ~0.51, real dark-background sparkles sit nearer ~0.31, so 0.6x reproduces the
    # capture-over-estimates-reality mismatch that triggers the bug.
    _REALISTIC_ALPHA_SCALE = 0.6

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def _composite_sparkle(self, bg_value: int, size: int = 1400, alpha_scale: float = _REALISTIC_ALPHA_SCALE):
        """Build a flat BGR image of ``bg_value`` with the sparkle composited in.

        The mark is composited at a LOWER effective opacity than the engine's captured
        alpha map (``alpha_scale`` < 1), reproducing the real-world mismatch behind
        issue #30: the captured alpha (~0.51) over-estimates a real sparkle whose
        effective opacity is lower, so the fixed-alpha reverse blend over-subtracts.
        Placed at the configured large-image position so the detector locates it.
        """
        img = np.full((size, size, 3), bg_value, dtype=np.float32)
        config = get_watermark_config(size, size)
        x, y = config.get_position(size, size)
        alpha = self.engine.get_alpha_map(WatermarkSize.LARGE)
        ah, aw = alpha.shape[:2]
        a = (alpha * alpha_scale)[:, :, None]
        roi = img[y : y + ah, x : x + aw]
        img[y : y + ah, x : x + aw] = a * 255.0 + (1.0 - a) * roi
        return np.clip(img, 0, 255).astype(np.uint8), (x, y, aw, ah)

    def test_dark_background_does_not_leave_black_pit(self):
        image, (x, y, w, h) = self._composite_sparkle(bg_value=60)
        out = self.engine.remove_watermark(image)
        footprint = out[y : y + h, x : x + w]
        # The recovered footprint must read like the dark background, not a black hole.
        assert footprint.min() > 25, f"black pit: min={footprint.min()}"
        assert abs(float(footprint.mean()) - 60.0) < 25.0

    def test_bright_background_keeps_reverse_alpha(self):
        """A bright background does not over-subtract, so reverse-alpha is used."""
        bright, pos = self._composite_sparkle(bg_value=230)
        alpha = self.engine.get_interpolated_alpha(pos[2])
        assert self.engine._reverse_alpha_oversubtracts(bright, alpha, (pos[0], pos[1])) is False
        dark, dpos = self._composite_sparkle(bg_value=60)
        dalpha = self.engine.get_interpolated_alpha(dpos[2])
        assert self.engine._reverse_alpha_oversubtracts(dark, dalpha, (dpos[0], dpos[1])) is True


class TestUnderSubtractionGain:
    """Under-subtraction fix: a sparkle MORE opaque than the captured alpha must not
    survive removal. The captured alpha (~0.51) under-represents such marks, so the
    fixed-alpha reverse blend leaves a bright residual; the per-image gain scales the
    alpha up to match this image's opacity. Mirror of TestOverSubtractionGuard.
    """

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def _composite_sparkle(self, bg_value: int, alpha_scale: float, size: int = 1400):
        """Flat ``bg_value`` image with the sparkle composited at ``alpha_scale`` opacity.

        ``alpha_scale`` > 1 makes the mark MORE opaque than the engine's captured alpha,
        reproducing the under-subtraction case (real under-removed marks estimate ~1.47).
        """
        img = np.full((size, size, 3), bg_value, dtype=np.float32)
        config = get_watermark_config(size, size)
        x, y = config.get_position(size, size)
        alpha = self.engine.get_alpha_map(WatermarkSize.LARGE)
        ah, aw = alpha.shape[:2]
        a = np.clip(alpha * alpha_scale, 0.0, 1.0)[:, :, None]
        roi = img[y : y + ah, x : x + aw]
        img[y : y + ah, x : x + aw] = a * 255.0 + (1.0 - a) * roi
        return np.clip(img, 0, 255).astype(np.uint8), (x, y, aw, ah)

    def test_more_opaque_sparkle_estimates_gain_above_deadband(self):
        image, pos = self._composite_sparkle(bg_value=80, alpha_scale=1.3)
        alpha = self.engine.get_interpolated_alpha(pos[2])
        gain = self.engine._estimate_alpha_gain(image, alpha, (pos[0], pos[1]))
        assert gain > self.engine._ALPHA_GAIN_DEADBAND, f"gain {gain} did not exceed deadband"

    def test_matching_sparkle_estimates_unit_gain(self):
        """A sparkle that matches the captured opacity gets ~1.0 (no scaling)."""
        image, pos = self._composite_sparkle(bg_value=80, alpha_scale=1.0)
        alpha = self.engine.get_interpolated_alpha(pos[2])
        gain = self.engine._estimate_alpha_gain(image, alpha, (pos[0], pos[1]))
        assert gain <= self.engine._ALPHA_GAIN_DEADBAND, f"matching sparkle scaled by {gain}"

    def test_more_opaque_sparkle_is_removed(self):
        """The gain-scaled removal clears a more-opaque sparkle without a black pit.

        Asserted on the footprint PIXELS, not the detector: the detector's NCC is
        degenerate on a perfectly flat synthetic background (zero-variance regions
        spuriously match), so a re-detect conf is meaningless here -- on real textured
        images the same removal drops the detector from ~0.80 to ~0.27 (spaces corpus).
        """
        image, (x, y, w, h) = self._composite_sparkle(bg_value=80, alpha_scale=1.3)
        assert self.engine.detect_watermark(image).detected
        before_max = int(image[y : y + h, x : x + w].max())  # bright sparkle present
        assert before_max > 150
        out = self.engine.remove_watermark(image)
        footprint = out[y : y + h, x : x + w]
        # Sparkle gone: no bright residual, no black pit, footprint reads like the bg.
        assert int(footprint.max()) < 80 + 30, f"bright residual: max={footprint.max()}"
        assert int(footprint.min()) > 25, f"black pit: min={footprint.min()}"
        assert abs(float(footprint.mean()) - 80.0) < 20.0


class TestVerifyAndRepair:
    """Self-verify fallback: a sparkle that survives reverse-alpha is inpaint-repaired,
    but only when that lowers the re-detect confidence (so it can never regress).

    The detector NCC is degenerate on flat synthetic backgrounds, so the keep-best
    control flow is driven through a stubbed ``detect_watermark`` rather than a real
    re-detect (mirroring the reasoning in TestUnderSubtractionGain).
    """

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()
        self.alpha = self.engine.get_interpolated_alpha(96)
        self.pos = (200, 200)

    def _stub_detect(self, confidences):
        """detect_watermark stub yielding the given confidences in order."""
        seq = iter(confidences)

        def fake(image, force_size=None):
            return DetectionResult(detected=True, confidence=next(seq))

        return fake

    def _repair(self, result):
        return self.engine._verify_and_repair(result, self.alpha, self.pos, WatermarkSize.LARGE)

    def test_clean_removal_returned_untouched(self, monkeypatch):
        """Below the fallback threshold, the input is returned byte-identical."""
        img = np.full((600, 600, 3), 90, dtype=np.uint8)
        monkeypatch.setattr(self.engine, "detect_watermark", self._stub_detect([0.2]))
        out = self._repair(img)
        assert out is img  # no copy, no inpaint

    def test_keeps_footprint_inpaint_when_it_helps(self, monkeypatch):
        """A surviving sparkle is footprint-inpainted when that re-detects lower; the
        footprint pixels change."""
        img = np.full((600, 600, 3), 90, dtype=np.uint8)
        # Bright residual block over the footprint so the inpaint visibly changes it.
        img[self.pos[1] : self.pos[1] + 96, self.pos[0] : self.pos[0] + 96] = 240
        # residual 0.7 (>= 0.5 triggers), candidate re-detects 0.2 (< residual -> keep).
        monkeypatch.setattr(self.engine, "detect_watermark", self._stub_detect([0.7, 0.2]))
        out = self._repair(img)
        assert out is not img
        assert not np.array_equal(out, img)  # footprint was inpainted from surroundings

    def test_repair_rejected_when_inpaint_does_not_help(self, monkeypatch):
        """When the inpaint does not lower the re-detect confidence, keep the original."""
        img = np.full((600, 600, 3), 90, dtype=np.uint8)
        # residual 0.7, candidate re-detects 0.75 (>= residual -> reject the inpaint).
        monkeypatch.setattr(self.engine, "detect_watermark", self._stub_detect([0.7, 0.75]))
        out = self._repair(img)
        assert out is img


class TestSparkleFalsePositiveGate:
    """False-positive gate: a low-confidence shape match whose core is NOT brighter
    than its surroundings (ornate/flat content, not a white sparkle overlay) is
    demoted below the detection threshold. Real sparkles escape via high confidence
    or a bright core-ring margin.
    """

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def _composite_sparkle(self, bg_value: int, alpha_scale: float, size: int = 1400):
        img = np.full((size, size, 3), bg_value, dtype=np.float32)
        config = get_watermark_config(size, size)
        x, y = config.get_position(size, size)
        alpha = self.engine.get_alpha_map(WatermarkSize.LARGE)
        ah, aw = alpha.shape[:2]
        a = np.clip(alpha * alpha_scale, 0.0, 1.0)[:, :, None]
        roi = img[y : y + ah, x : x + aw]
        img[y : y + ah, x : x + aw] = a * 255.0 + (1.0 - a) * roi
        return np.clip(img, 0, 255).astype(np.uint8), (x, y, aw, ah)

    def test_bright_core_has_high_margin(self):
        image, (x, y, w, _h) = self._composite_sparkle(bg_value=60, alpha_scale=1.0)
        margin = self.engine._core_ring_margin(image, self.engine.get_interpolated_alpha(w), (x, y))
        assert margin is not None
        assert margin > self.engine._SPARKLE_FP_MARGIN

    def test_flat_region_has_low_margin(self):
        """A uniform region (no white sparkle) has ~zero core-ring margin."""
        flat = np.full((1400, 1400, 3), 128, dtype=np.uint8)
        config = get_watermark_config(1400, 1400)
        pos = config.get_position(1400, 1400)
        alpha = self.engine.get_interpolated_alpha(96)
        margin = self.engine._core_ring_margin(flat, alpha, pos)
        assert margin is not None
        assert abs(margin) < self.engine._SPARKLE_FP_MARGIN

    def test_strong_sparkle_not_demoted(self):
        image, _ = self._composite_sparkle(bg_value=60, alpha_scale=1.0)
        det = self.engine.detect_watermark(image)
        assert det.detected
        assert det.confidence >= 0.5

    def test_strong_sparkle_on_white_kept(self):
        """A real sparkle on a near-white background has a LOW core-ring margin (the
        white overlay barely lifts white) but a HIGH NCC confidence, so the gate must
        NOT demote it -- high confidence is the escape hatch."""
        image, (x, y, w, _h) = self._composite_sparkle(bg_value=251, alpha_scale=1.0)
        margin = self.engine._core_ring_margin(image, self.engine.get_interpolated_alpha(w), (x, y))
        assert margin is not None
        assert margin < self.engine._SPARKLE_FP_MARGIN  # low margin
        det = self.engine.detect_watermark(image)
        assert det.detected
        assert det.confidence >= 0.65  # but kept via high confidence

    def test_low_margin_blurred_blob_is_demoted(self):
        """A heavily-blurred faint near-white blob NCC-matches the sparkle shape (the
        stage scores fuse above the 0.5 promote bar) but has no bright core (low
        margin), so the gate demotes the returned confidence below it -- the content
        false-positive case."""
        size = 1400
        config = get_watermark_config(size, size)
        x, y = config.get_position(size, size)
        alpha = self.engine.get_alpha_map(WatermarkSize.LARGE)
        ah, aw = alpha.shape[:2]
        img = np.full((size, size, 3), 247, dtype=np.float32)
        a = np.clip(alpha * 0.5, 0.0, 1.0)[:, :, None]
        img[y : y + ah, x : x + aw] = a * 255.0 + (1.0 - a) * img[y : y + ah, x : x + aw]
        img = cv2.GaussianBlur(np.clip(img, 0, 255).astype(np.uint8), (31, 31), 0)
        det = self.engine.detect_watermark(img)
        # The raw stage scores fuse above 0.5 (would be promoted)...
        pre = det.spatial_score * 0.5 + det.gradient_score * 0.3 + det.variance_score * 0.2
        assert pre > 0.5
        # ...but the no-bright-core gate caps the returned confidence below the bar.
        assert det.confidence < 0.5
        assert not det.detected


class TestCornerPromotion:
    """Issue #36: a small sparkle in the corner must not be lost to a larger decoy.

    The size weight that suppresses tiny-patch false positives also lets a larger,
    mediocre match elsewhere outrank a small, near-perfect sparkle in the corner --
    so a faint sparkle on a busy background (e.g. a portrait whose bright collar
    out-scores it) reads as clean. The corner-promotion override rescues it.
    """

    _W, _H = 400, 520
    _CORNER = (_W - 40 - 20, _H - 40 - 20, 20)  # bottom-right small sparkle (x, y, scale)
    _DECOY = (15, 210, 92)  # large decoy: inside the search window, left of the corner

    @pytest.fixture(autouse=True)
    def _setup_engine(self):
        self.engine = GeminiEngine()

    def _paste(self, img: np.ndarray, scale: int, x: int, y: int, alpha_scale: float) -> None:
        tmpl = cv2.resize(self.engine._alpha_large, (scale, scale), interpolation=cv2.INTER_AREA)
        a = (tmpl * alpha_scale)[:, :, None]
        roi = img[y : y + scale, x : x + scale]
        img[y : y + scale, x : x + scale] = a * 255.0 + (1.0 - a) * roi

    def _scene(self, bg_value: int = 40) -> np.ndarray:
        """Dark scene with a large decoy on the left and a small sparkle in the corner.

        Without the corner-promotion fix the global, size-weighted search locks onto
        the larger decoy; with it the small corner sparkle wins.
        """
        img = np.full((self._H, self._W, 3), bg_value, dtype=np.float32)
        self._paste(img, self._DECOY[2], self._DECOY[0], self._DECOY[1], 0.55)
        self._paste(img, self._CORNER[2], self._CORNER[0], self._CORNER[1], 0.55)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _in_bottom_right(self, region: tuple[int, int, int, int]) -> bool:
        x, y = region[0], region[1]
        return x >= self._W * 0.6 and y >= self._H * 0.6

    def test_small_corner_sparkle_is_detected_and_localized(self):
        det = self.engine.detect_watermark(self._scene())
        assert det.detected
        # Must localize to the planted corner sparkle, not the larger left-side decoy.
        assert self._in_bottom_right(det.region), f"localized to decoy, not corner: {det.region}"
        assert abs(det.region[0] - self._CORNER[0]) < 16
        assert abs(det.region[1] - self._CORNER[1]) < 16

    def test_size_weighted_search_alone_traps_on_the_decoy(self):
        """Guard the mechanism: the size-weighted argmax ALONE mislocalizes to the decoy.

        Proves the scene genuinely needs the dual-candidate fusion selection (so the
        localization test above is not a fluke): replicating the old size-weighted-only
        pick lands on the larger left-side decoy, while ``detect_watermark`` -- which
        scores all top-K candidates by full fusion -- lands on the corner sparkle.
        """
        scene = self._scene()
        h, w = scene.shape[:2]
        ss = min(min(w, h), 512)
        sx1, sy1 = max(0, w - ss), max(0, h - ss)
        gray = cv2.cvtColor(scene[sy1:h, sx1:w], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        best = (-1.0, (0, 0))
        for scale, max_val, max_loc in self.engine._scan_scales(gray):
            adj = max_val * min(1.0, (scale / 96.0) ** 0.5)
            if adj > best[0]:
                best = (adj, max_loc)
        weighted_region = (sx1 + best[1][0], sy1 + best[1][1])
        assert not self._in_bottom_right((*weighted_region, 0, 0)), "size-weighted argmax expected to hit the decoy"
        # The full detector, scoring both candidates by fusion, rescues the corner.
        assert self._in_bottom_right(self.engine.detect_watermark(scene).region)

    def test_no_promotion_on_clean_flat_image(self):
        """A flat image with no sparkle yields no corner match to promote."""
        flat = np.full((self._H, self._W, 3), 40, dtype=np.uint8)
        assert self.engine._corner_promote(flat, -1.0) is None

    def test_low_gradient_decoy_loses_to_high_gradient_corner_sparkle(self, monkeypatch):
        """Regression (reporter 2026-06-12, v0.7.2 detected / v0.8-0.11 missed): the
        256->512 search widening let a large, low-gradient size-weighted match outrank
        a smaller, high-gradient corner sparkle whose raw NCC (~0.77) sits BELOW the
        0.85 corner-promote gate -- so a real Gemini sparkle on light bedding read as
        clean. The detector now scores all top-K candidates by full fusion and keeps
        the highest; the gradient term (~0.04 on flat content vs ~1.0 on a true
        sparkle) rescues the corner. Mirrors the real signature:
        decoy spatial 0.63 / grad 0.04 vs corner spatial 0.78 / grad 0.96.
        """
        eng = self.engine
        size = 500
        img = np.full((size, size, 3), 225, dtype=np.float32)
        # Low-gradient smooth bright blob = a content false match the size weight favors.
        blob = np.zeros((size, size), np.float32)
        cv2.circle(blob, (89, 89), 60, 1.0, -1)
        blob = cv2.GaussianBlur(blob, (0, 0), 30)
        img += (blob * 25)[:, :, None]
        # A real, high-gradient sparkle in the bottom-right corner.
        cx, cy, cs = 430, 430, 48
        tmpl = cv2.resize(eng._alpha_large, (cs, cs), interpolation=cv2.INTER_AREA)
        a = (tmpl * 0.6)[:, :, None]
        img[cy : cy + cs, cx : cx + cs] = a * 255.0 + (1.0 - a) * img[cy : cy + cs, cx : cx + cs]
        img = np.clip(img, 0, 255).astype(np.uint8)
        dx, dy, ds = 40, 40, 98

        # The two locations carry the real-signature gradient split.
        assert eng._grad_var_scores(img, ds, dx, dy)[0] < 0.1  # decoy: shape-only, ~0.04
        assert eng._grad_var_scores(img, cs, cx, cy)[0] > 0.8  # corner sparkle: ~1.0

        def fake_scan(_self, _gray):
            yield ds, 0.66, (dx, dy)  # large, low-fidelity: size-weighted adj 0.66 -> winner
            yield cs, 0.80, (cx, cy)  # small sparkle: raw 0.80 but adj 0.57 -> size-weight loses

        monkeypatch.setattr(GeminiEngine, "_scan_scales", fake_scan)
        det = eng.detect_watermark(img)
        # Full-fusion selection keeps the high-gradient corner sparkle, not the decoy.
        assert det.detected
        assert det.confidence >= 0.5
        assert abs(det.region[0] - cx) < 4, f"localized to decoy: {det.region}"
        assert abs(det.region[1] - cy) < 4, f"localized to decoy: {det.region}"
