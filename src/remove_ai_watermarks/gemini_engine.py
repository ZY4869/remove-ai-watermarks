"""Gemini visible watermark removal engine.

Port of the GeminiWatermarkTool reverse-alpha-blending algorithm from C++ to Python.
Original author: Allen Kuo (allenk) — https://github.com/allenk/GeminiWatermarkTool

The Gemini AI watermark is applied using alpha blending:
    watermarked = a * logo + (1 - a) * original

We reverse this to recover the original:
    original = (watermarked - a * logo) / (1 - a)

The alpha maps are derived from background captures of the Gemini watermark
on pure-black backgrounds (48x48 for small images, 96x96 for large images).
"""

# cv2/numpy boundary: cv2 and numpy ship no usable type info for the array ops
# below, so strict pyright cannot know their element types. Relax the unknown-type
# rules for this file only; the public signatures are still annotated with NDArray[Any].
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class WatermarkSize(Enum):
    """Watermark size mode based on image dimensions."""

    SMALL = "small"  # 48x48, for images <= 1024x1024
    LARGE = "large"  # 96x96, for images > 1024x1024


@dataclass
class DetectionResult:
    """Result of watermark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    size: WatermarkSize = WatermarkSize.SMALL

    # stage scores
    spatial_score: float = 0.0
    gradient_score: float = 0.0
    variance_score: float = 0.0


@dataclass
class WatermarkPosition:
    """Watermark position configuration."""

    margin_right: int
    margin_bottom: int
    logo_size: int

    def get_position(self, image_width: int, image_height: int) -> tuple[int, int]:
        """Get top-left position for a given image size."""
        x = image_width - self.margin_right - self.logo_size
        y = image_height - self.margin_bottom - self.logo_size
        return (x, y)


def get_watermark_config(width: int, height: int) -> WatermarkPosition:
    """Get the appropriate watermark configuration based on image size.

    Rules discovered from Gemini:
      - W > 1024 AND H > 1024: 96x96 logo at (W-64-96, H-64-96)
      - Otherwise:              48x48 logo at (W-32-48, H-32-48)
    """
    if width > 1024 and height > 1024:
        return WatermarkPosition(margin_right=64, margin_bottom=64, logo_size=96)
    return WatermarkPosition(margin_right=32, margin_bottom=32, logo_size=48)


def get_watermark_size(width: int, height: int) -> WatermarkSize:
    """Determine watermark size mode from image dimensions."""
    if width > 1024 and height > 1024:
        return WatermarkSize.LARGE
    return WatermarkSize.SMALL


def _calculate_alpha_map(bg_capture: NDArray[Any]) -> NDArray[Any]:
    """Calculate alpha map from a background capture.

    The alpha map represents how much the watermark affects each pixel.
    alpha = max(R, G, B) / 255.0
    """
    if len(bg_capture.shape) == 2:
        gray = bg_capture.astype(np.float32)
    elif bg_capture.shape[2] >= 3:
        # Use max of channels for brightness
        gray = np.max(bg_capture[:, :, :3], axis=2).astype(np.float32)
    else:
        gray = bg_capture[:, :, 0].astype(np.float32)

    return gray / 255.0


def _load_embedded_asset(name: str) -> NDArray[Any]:
    """Load an embedded PNG asset and decode it with OpenCV."""
    asset_path = Path(__file__).parent / "assets" / name
    if not asset_path.exists():
        raise FileNotFoundError(f"Embedded asset not found: {asset_path}")

    data = asset_path.read_bytes()
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode embedded asset: {name}")
    return img


class GeminiEngine:
    """Engine for removing visible Gemini watermarks via reverse alpha blending.

    This is a Python port of the GeminiWatermarkTool C++ engine.
    """

    # Footprint pixels with alpha at/above this are the sparkle body; below it the
    # mark barely affects the pixel, so those are excluded from both the
    # over-subtraction test and the inpaint mask.
    _FOOTPRINT_ALPHA = 0.1
    # If more than this fraction of footprint pixels over-subtract (numerator < 0),
    # the fixed alpha does not match this image's sparkle and reverse-alpha would
    # punch a dark pit -- inpaint instead. demo_banana measures 0.0 (reverse-alpha
    # kept), the issue #30 dark-grass image measures ~0.61 (inpaint), so the 0.05
    # gate separates them with a wide margin.
    _OVERSUB_FOOTPRINT_FRAC = 0.05

    # Per-image alpha gain (under-subtraction fix). The captured alpha peaks ~0.51
    # (a ~51%-opaque sparkle). Some real Gemini sparkles are rendered MORE opaque,
    # so the fixed alpha under-subtracts and reverse-alpha leaves a bright residual
    # the detector still fires on (~11% of marks on the spaces corpus). Estimate
    # this image's effective sparkle opacity from the bright core vs the local
    # background and scale the alpha to match, capped so alpha stays < 0.99. The
    # gain is clamped to >= 1.0 so it only ever STRENGTHENS removal: ~1.0 when the
    # sparkle matches the capture (working cases unchanged), >1 when more opaque.
    # On the spaces corpus the gain cleanly separates -- under-removed marks ~1.47,
    # cleanly-removed ~1.00. 1.94 is the cap that reaches alpha 0.99 from 0.51.
    _ALPHA_GAIN_MAX = 1.94
    _ALPHA_GAIN_CORE_FRAC = 0.8  # body pixels at >= this * peak alpha define the core
    # Deadband: apply the gain only above this, so a sparkle that already matches the
    # capture (estimated gain ~1.0-1.04 from background noise) stays byte-identical to
    # the pre-fix output. Under-removed marks estimate >= 1.26, well clear of the band.
    _ALPHA_GAIN_DEADBAND = 1.05

    # Corner promotion (issue #36): the size weight that suppresses tiny-patch
    # false positives also buries a small, near-perfect sparkle when a larger,
    # mediocre match sits elsewhere (e.g. a bright collar in a portrait). A small
    # faint sparkle on a busy background therefore loses the global argmax and the
    # image reads as clean -- the regression osachub reported when the search
    # window widened 256px -> 512px (v0.7.2's tighter window still found it).
    # Remedy: if the bottom-right corner holds a very-high-fidelity raw-NCC match,
    # trust it regardless of size, without reverting the wider window (which is
    # needed for variant margins). The threshold sits midway between the worst
    # real-photo corner match (~0.78 across native + downscaled real photos) and a
    # genuine faint sparkle (~0.93), so it adds true detections without adding
    # false ones; it only ever overrides a lower-fidelity global pick, so it cannot
    # weaken an existing detection.
    _CORNER_PROMOTE_NCC = 0.85
    # Bottom-right corner side for the promotion search, as a fraction of the
    # image's short side, clamped to an absolute pixel band. Relative so the corner
    # stays a true corner at every scale: a fixed 256 px is a genuine corner on a
    # large image but covers ~70% of a small portrait, where a busy real photo can
    # then raw-match the star template at ~0.81 (only 0.04 below the promote gate).
    # Scaling the side down on small images drops that worst case to ~0.69, while
    # the upper clamp stops it ballooning on huge images (more corner area = more
    # random texture to false-match -- a real photo reached ~0.83 at 512 px). The
    # Gemini sparkle sits ~60-160 px from the corner (fixed margins, not
    # proportional), and the [96, 384] band covers that at every measured size.
    _CORNER_PROMOTE_FRAC = 0.20
    _CORNER_PROMOTE_MIN = 96
    _CORNER_PROMOTE_MAX = 384

    def __init__(self, logo_value: float = 255.0) -> None:
        """Initialize the engine with embedded alpha maps.

        Args:
            logo_value: The logo brightness value (default 255.0 = white).
        """
        self.logo_value = logo_value

        # Load embedded background captures
        bg_small = _load_embedded_asset("gemini_bg_48.png")
        bg_large = _load_embedded_asset("gemini_bg_96.png")

        # Ensure correct sizes
        if bg_small.shape[:2] != (48, 48):
            bg_small = cv2.resize(bg_small, (48, 48), interpolation=cv2.INTER_AREA)
        if bg_large.shape[:2] != (96, 96):
            bg_large = cv2.resize(bg_large, (96, 96), interpolation=cv2.INTER_AREA)

        # Calculate alpha maps
        self._alpha_small = _calculate_alpha_map(bg_small)
        self._alpha_large = _calculate_alpha_map(bg_large)

        logger.debug(
            "Alpha maps loaded: small=%s, large=%s",
            self._alpha_small.shape,
            self._alpha_large.shape,
        )

    def get_alpha_map(self, size: WatermarkSize) -> NDArray[Any]:
        """Get the base alpha map for a specific standard size."""
        if size == WatermarkSize.SMALL:
            return self._alpha_small
        return self._alpha_large

    def get_interpolated_alpha(self, size_px: int) -> NDArray[Any]:
        """Create an interpolated alpha map dynamically scaled from the high-res 96x96 base."""
        source = self._alpha_large
        if size_px == source.shape[1]:
            return source.copy()

        interp = cv2.INTER_LINEAR if size_px > source.shape[1] else cv2.INTER_AREA
        return cv2.resize(source, (size_px, size_px), interpolation=interp)

    # ── Detection ────────────────────────────────────────────────────

    def _scan_scales(self, gray: NDArray[Any]) -> Iterator[tuple[int, float, tuple[int, int]]]:
        """Yield ``(scale, max_ncc, max_loc)`` for the alpha template matched at each scale.

        Shared multi-scale ``TM_CCOEFF_NORMED`` primitive over a normalized [0, 1]
        grayscale region, used by both the size-weighted global search in
        ``detect_watermark`` and the raw-NCC corner pass in ``_corner_promote`` --
        each applies its own scoring/argmax to the yielded values. The 96x96
        ``_alpha_large`` is the high-quality source downscaled per scale; the range
        covers aggressively downscaled to slightly upscaled logos.
        """
        for scale in range(16, 120, 2):
            if scale > gray.shape[0] or scale > gray.shape[1]:
                continue
            tmpl = cv2.resize(self._alpha_large, (scale, scale), interpolation=cv2.INTER_AREA)
            match_res = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match_res)
            yield scale, float(max_val), max_loc

    def detect_watermark(
        self,
        image: NDArray[Any],
        force_size: WatermarkSize | None = None,
    ) -> DetectionResult:
        """Detect Gemini watermark using multi-scale Snap Engine logic (ported from C++ vendor algorithm)."""
        result = DetectionResult()

        if image is None or image.size == 0:
            return result

        h, w = image.shape[:2]
        base_size = force_size or get_watermark_size(w, h)
        result.size = base_size

        # Dynamically search bottom-right corner. 512 covers up to 512px from the
        # corner -- enough for known Gemini margin variations (standard: 64+96=160px;
        # observed variants up to ~300px). 256 was too tight and caused misses.
        search_size = int(min(min(w, h), 512))
        sx1 = max(0, w - search_size)
        sy1 = max(0, h - search_size)

        search_region = image[sy1:h, sx1:w]
        if len(search_region.shape) == 3 and search_region.shape[2] >= 3:
            gray_sr = cv2.cvtColor(search_region, cv2.COLOR_BGR2GRAY)
        else:
            gray_sr = search_region.copy()

        gray_sr_f = gray_sr.astype(np.float32) / 255.0

        # Phase 1 & 2: multi-scale spatial NCC search, size-weighted argmax.
        best_scale = 0
        best_score = -1.0
        best_raw_ncc = -1.0
        best_loc = (0, 0)
        for scale, max_val, max_loc in self._scan_scales(gray_sr_f):
            # Size-adjusted score to overcome NCC bias toward tiny patches (mimics C++ weight)
            weight = min(1.0, (scale / 96.0) ** 0.5)
            adj_val = max_val * weight
            if adj_val > best_score:
                best_score = adj_val
                best_scale = scale
                best_loc = max_loc
                best_raw_ncc = max_val

        # Exact dynamic location & size
        pos_x = sx1 + best_loc[0]
        pos_y = sy1 + best_loc[1]

        # Corner promotion: a near-perfect but small sparkle in the bottom-right
        # corner is otherwise outranked by a larger, mediocre size-weighted match
        # (see _CORNER_PROMOTE_NCC). Override the global pick with it when present.
        promoted = self._corner_promote(image, best_raw_ncc)
        if promoted is not None:
            best_scale, pos_x, pos_y, best_raw_ncc = promoted

        result.region = (pos_x, pos_y, best_scale, best_scale)
        result.spatial_score = float(best_raw_ncc)

        # Generate exact alpha map for matched size
        alpha_region = self.get_interpolated_alpha(best_scale)

        # Extract exactly the matched region for Gradient & Variance analysis
        x1 = pos_x
        y1 = pos_y
        x2 = min(w, x1 + best_scale)
        y2 = min(h, y1 + best_scale)

        region = image[y1:y2, x1:x2]
        if len(region.shape) == 3 and region.shape[2] >= 3:
            gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        else:
            gray_region = region.copy()

        gray_f = gray_region.astype(np.float32) / 255.0

        # Adjust alpha_region if clipped by image boundary (rare, but possible)
        ay1, ax1 = 0, 0
        alpha_region = alpha_region[ay1 : ay1 + (y2 - y1), ax1 : ax1 + (x2 - x1)]

        if result.spatial_score < 0.25:
            result.confidence = float(max(0.0, result.spatial_score * 0.5))
            return result

        # ── Stage 2: Gradient NCC ────────────────────────────────────
        img_gx = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        img_gy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        img_gmag = cv2.magnitude(img_gx, img_gy)

        alpha_gx = cv2.Sobel(alpha_region, cv2.CV_32F, 1, 0, ksize=3)
        alpha_gy = cv2.Sobel(alpha_region, cv2.CV_32F, 0, 1, ksize=3)
        alpha_gmag = cv2.magnitude(alpha_gx, alpha_gy)

        grad_match = cv2.matchTemplate(img_gmag, alpha_gmag, cv2.TM_CCOEFF_NORMED)
        _, grad_score, _, _ = cv2.minMaxLoc(grad_match)
        result.gradient_score = float(grad_score)

        # ── Stage 3: Variance Analysis ───────────────────────────────
        var_score = 0.0
        ref_h = min(y1, best_scale)

        if ref_h > 8:
            ref_region = image[y1 - ref_h : y1, x1:x2]
            gray_ref = cv2.cvtColor(ref_region, cv2.COLOR_BGR2GRAY) if len(ref_region.shape) == 3 else ref_region

            _, s_wm = cv2.meanStdDev(gray_region)
            _, s_ref = cv2.meanStdDev(gray_ref)

            if s_ref[0][0] > 5.0:
                var_score = max(0.0, min(1.0, 1.0 - (s_wm[0][0] / s_ref[0][0])))

        result.variance_score = float(var_score)

        # ── Fusion ───────────────────────────────────────────────────
        confidence = result.spatial_score * 0.50 + result.gradient_score * 0.30 + var_score * 0.20
        result.confidence = float(max(0.0, min(1.0, confidence)))
        result.detected = result.confidence >= 0.35

        logger.debug(
            "Detection: spatial=%.3f, grad=%.3f, var=%.3f → conf=%.3f (%s)",
            result.spatial_score,
            result.gradient_score,
            var_score,
            result.confidence,
            "DETECTED" if result.detected else "not detected",
        )

        return result

    def _corner_promote(
        self,
        image: NDArray[Any],
        current_raw_ncc: float,
    ) -> tuple[int, int, int, float] | None:
        """Search the bottom-right corner for a very-high-fidelity sparkle match.

        Returns ``(scale, x, y, raw_ncc)`` when the corner holds a match with raw
        NCC >= ``_CORNER_PROMOTE_NCC`` that beats the global pick's ``current_raw_ncc``,
        else None. Used to rescue a small sparkle that the size weight buried under
        a larger, lower-fidelity match elsewhere. See ``_CORNER_PROMOTE_NCC`` and
        ``_CORNER_PROMOTE_FRAC`` for the corner sizing.
        """
        h, w = image.shape[:2]
        side = max(
            self._CORNER_PROMOTE_MIN, min(self._CORNER_PROMOTE_MAX, round(min(w, h) * self._CORNER_PROMOTE_FRAC))
        )
        cs = int(min(min(w, h), side))
        cx1, cy1 = max(0, w - cs), max(0, h - cs)
        corner = image[cy1:h, cx1:w]
        gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY) if corner.ndim == 3 and corner.shape[2] >= 3 else corner
        gray = gray.astype(np.float32) / 255.0

        best_raw = -1.0
        best_scale = 0
        best_loc = (0, 0)
        for scale, max_val, max_loc in self._scan_scales(gray):
            if max_val > best_raw:
                best_raw = max_val
                best_scale = scale
                best_loc = max_loc

        if best_raw >= self._CORNER_PROMOTE_NCC and best_raw > current_raw_ncc:
            return best_scale, cx1 + best_loc[0], cy1 + best_loc[1], float(best_raw)
        return None

    # ── Removal ──────────────────────────────────────────────────────

    def remove_watermark(
        self,
        image: NDArray[Any],
        force_size: WatermarkSize | None = None,
    ) -> NDArray[Any]:
        """Remove Gemini visible watermark from an image using reverse alpha blending.

        No-op when the detector does not find a watermark: returns an unmodified
        copy. Reverse alpha blending applied where no sparkle exists creates a
        visible inverse artifact, so we refuse to touch pixels without a positive
        detection. To bypass detection (e.g. you know the exact region), use
        ``remove_watermark_custom``.

        Args:
            image: BGR image as numpy array (will NOT be modified in-place).
            force_size: Force a specific watermark size (auto-detect if None).

        Returns:
            Cleaned BGR image as numpy array, or an unmodified copy when no
            watermark is detected.
        """
        result = image.copy()

        # Normalize to 3-channel BGR up front: 2D grayscale (no channel axis) and
        # 4-channel BGRA both reach this public entry point and would otherwise
        # crash on the channel-count checks / downstream 3-channel math.
        if result.ndim == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        elif result.shape[2] == 4:
            result = cv2.cvtColor(result, cv2.COLOR_BGRA2BGR)
        elif result.shape[2] == 1:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

        size = force_size or get_watermark_size(result.shape[1], result.shape[0])

        # Detect dynamic position & size (on the normalized 3-channel image so a
        # grayscale/BGRA input does not crash the detector).
        detection = self.detect_watermark(result, force_size=size)

        if not detection.detected:
            logger.debug(
                "No watermark detected (conf=%.3f); returning image unchanged.",
                detection.confidence,
            )
            return result

        pos = (detection.region[0], detection.region[1])
        alpha_map = self.get_interpolated_alpha(detection.region[2])
        # Match the captured alpha to this image's sparkle opacity (under-subtraction
        # fix): a more-opaque-than-captured sparkle would otherwise leave a bright
        # residual. gain == 1.0 leaves the working cases byte-identical.
        gain = self._estimate_alpha_gain(result, alpha_map, pos)
        if gain > self._ALPHA_GAIN_DEADBAND:
            alpha_map = np.clip(alpha_map * gain, 0.0, 0.99)
        logger.debug(
            "Removing watermark at (%d, %d) size %dx%d [conf=%.3f]",
            pos[0],
            pos[1],
            detection.region[2],
            detection.region[3],
            detection.confidence,
        )

        # The captured alpha map (max ~0.51 = a ~50%-opaque white sparkle) is exact
        # only when the real mark's effective opacity matches it. On a dark/textured
        # background the sparkle's effective alpha is lower than the capture, so the
        # fixed-alpha reverse blend OVER-subtracts and drives the footprint to black --
        # the "white sparkle turns into a black pit" bug (issue #30). The signature is
        # a large fraction of footprint pixels whose numerator (watermarked - a*logo)
        # goes negative, which is physically impossible under a brightening overlay.
        # In that case inpaint the small footprint from the surrounding pixels instead;
        # on a bright background no pixel over-subtracts, so reverse-alpha is used and
        # the result is byte-identical to before (verified on demo_banana: 0% vs 61%).
        if self._reverse_alpha_oversubtracts(result, alpha_map, pos):
            logger.debug("Reverse-alpha over-subtracts on this background; inpainting sparkle footprint.")
            self._inpaint_footprint(result, alpha_map, pos)
        else:
            self._reverse_alpha_blend(result, alpha_map, pos)
        return result

    def remove_watermark_custom(
        self,
        image: NDArray[Any],
        region: tuple[int, int, int, int],
    ) -> NDArray[Any]:
        """Remove watermark from a custom region with interpolated alpha map.

        Args:
            image: BGR image (will NOT be modified in-place).
            region: (x, y, width, height) of the watermark region.

        Returns:
            Cleaned BGR image.
        """
        result = image.copy()
        x, y, rw, rh = region

        # Check standard sizes
        if rw == 48 and rh == 48:
            self._reverse_alpha_blend(result, self._alpha_small, (x, y))
            return result
        if rw == 96 and rh == 96:
            self._reverse_alpha_blend(result, self._alpha_large, (x, y))
            return result

        # Interpolate alpha map for custom size
        interp = cv2.INTER_LINEAR if rw > 96 else cv2.INTER_AREA
        alpha = cv2.resize(self._alpha_large, (rw, rh), interpolation=interp)
        self._reverse_alpha_blend(result, alpha, (x, y))
        return result

    def _footprint_indices(
        self,
        alpha_map: NDArray[Any],
        position: tuple[int, int],
        image_shape: tuple[int, ...],
    ) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Return (alpha_roi, (y1, y2, x1, x2)) for the placed footprint, or None.

        Shared by the over-subtraction test and the inpaint mask so both operate on
        exactly the same clipped, in-bounds region.
        """
        x, y = position
        ah, aw = alpha_map.shape[:2]
        ih, iw = image_shape[:2]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(iw, x + aw), min(ih, y + ah)
        if x1 >= x2 or y1 >= y2:
            return None
        ax1, ay1 = x1 - x, y1 - y
        alpha_roi = alpha_map[ay1 : ay1 + (y2 - y1), ax1 : ax1 + (x2 - x1)]
        return alpha_roi, (y1, y2, x1, x2)

    def _estimate_alpha_gain(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> float:
        """Scale factor matching the captured alpha to this image's sparkle opacity.

        The captured alpha (peak ~0.51) under-represents sparkles rendered more
        opaque; reverse-alpha then leaves a bright residual. Estimate the effective
        opacity at the sparkle core (observed brightness vs the local background
        ring) and return ``a_eff / a_capture``, clamped to ``[1.0, _ALPHA_GAIN_MAX]``
        so it only ever STRENGTHENS removal (1.0 = no change on a matching sparkle).
        Returns 1.0 when the background cannot be estimated reliably.
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return 1.0
        alpha_roi, (y1, y2, x1, x2) = placed
        a_cap = float(alpha_roi.max())
        if a_cap < 0.2:
            return 1.0
        gray = image.astype(np.float32).mean(axis=2)
        core = alpha_roi >= a_cap * self._ALPHA_GAIN_CORE_FRAC
        if not bool(core.any()):
            return 1.0
        core_obs = float(np.percentile(gray[y1:y2, x1:x2][core], 75))
        # Local background = a ring just outside the footprint box.
        ih, iw = image.shape[:2]
        pad = int((x2 - x1) * 0.7)
        ry1, ry2 = max(0, y1 - pad), min(ih, y2 + pad)
        rx1, rx2 = max(0, x1 - pad), min(iw, x2 + pad)
        ring = gray[ry1:ry2, rx1:rx2]
        ring_mask = np.ones(ring.shape, dtype=bool)
        ring_mask[y1 - ry1 : y2 - ry1, x1 - rx1 : x2 - rx1] = False
        if int(ring_mask.sum()) < 10:
            return 1.0
        bg = float(np.median(ring[ring_mask]))
        if 255.0 - bg < 5.0:
            return 1.0
        a_eff = float(np.clip((core_obs - bg) / (255.0 - bg), 0.0, 0.99))
        return float(np.clip(a_eff / a_cap, 1.0, self._ALPHA_GAIN_MAX))

    def _reverse_alpha_oversubtracts(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> bool:
        """True when reverse-alpha would drive the footprint dark (issue #30).

        Tests the numerator ``watermarked - alpha*logo`` over the sparkle body: a
        brightening overlay can never make it negative, so a large negative fraction
        means the fixed alpha over-estimates this image's opacity.
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return False
        alpha_roi, (y1, y2, x1, x2) = placed
        body = alpha_roi >= self._FOOTPRINT_ALPHA
        if not bool(body.any()):
            return False
        roi = image[y1:y2, x1:x2].astype(np.float32)
        numerator = roi.mean(axis=2) - np.clip(alpha_roi, 0.0, 0.99) * self.logo_value
        frac = float((numerator[body] < 0).sum()) / float(body.sum())
        return frac > self._OVERSUB_FOOTPRINT_FRAC

    def _inpaint_footprint(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> None:
        """Inpaint the sparkle body from surrounding pixels, in-place.

        Fallback for backgrounds where reverse-alpha over-subtracts: a small mask of
        the footprint (alpha >= threshold, dilated) is reconstructed by cv2 NS inpaint
        from the continuous surroundings, so the sparkle is replaced by plausible
        background instead of a black pit.
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return
        alpha_roi, (y1, y2, x1, x2) = placed
        # Inpaint only a padded crop around the footprint, not the whole image: the
        # mask is zero outside a ~96x96 corner, so inpainting the full (multi-MP)
        # image would be ~hundreds of times more work for an identical result. The
        # padding gives cv2 enough surrounding context to reconstruct the sparkle.
        ih, iw = image.shape[:2]
        pad = 24
        cy1, cy2 = max(0, y1 - pad), min(ih, y2 + pad)
        cx1, cx2 = max(0, x1 - pad), min(iw, x2 + pad)
        crop = image[cy1:cy2, cx1:cx2]
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        mask[y1 - cy1 : y2 - cy1, x1 - cx1 : x2 - cx1] = (alpha_roi >= self._FOOTPRINT_ALPHA).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel, iterations=2)
        image[cy1:cy2, cx1:cx2] = cv2.inpaint(crop, mask, 6, cv2.INPAINT_NS)

    def _reverse_alpha_blend(
        self,
        image: NDArray[Any],
        alpha_map: NDArray[Any],
        position: tuple[int, int],
    ) -> None:
        """Apply reverse alpha blending in-place.

        Formula: original = (watermarked - a * logo) / (1 - a)
        """
        placed = self._footprint_indices(alpha_map, position, image.shape)
        if placed is None:
            return
        alpha_roi, (y1, y2, x1, x2) = placed
        image_roi = image[y1:y2, x1:x2].astype(np.float32)

        alpha_threshold = 0.002
        max_alpha = 0.99

        # Vectorized reverse alpha blending
        alpha = alpha_roi.copy()
        mask = alpha >= alpha_threshold
        alpha = np.clip(alpha, 0.0, max_alpha)
        one_minus_alpha = 1.0 - alpha

        # Expand alpha for 3-channel broadcast
        alpha_3d = alpha[:, :, np.newaxis]
        one_minus_3d = one_minus_alpha[:, :, np.newaxis]
        mask_3d = mask[:, :, np.newaxis]

        # original = (watermarked - alpha * logo) / (1 - alpha)
        restored = (image_roi - alpha_3d * self.logo_value) / one_minus_3d
        restored = np.clip(restored, 0.0, 255.0)

        # Apply only where alpha is significant
        image_roi = np.where(mask_3d, restored, image_roi)
        image[y1:y2, x1:x2] = image_roi.astype(np.uint8)

    # ── Inpainting cleanup ───────────────────────────────────────────

    def inpaint_residual(
        self,
        image: NDArray[Any],
        region: tuple[int, int, int, int],
        strength: float = 0.85,
        method: Literal["gaussian", "telea", "ns"] = "ns",
        inpaint_radius: int = 10,
        padding: int = 32,
    ) -> NDArray[Any]:
        """Apply inpaint cleanup on residual artifacts after reverse alpha blend.

        Uses a sparse mask derived from alpha map gradient to repair only
        the sparkle-edge pixels where interpolation broke the math.

        Args:
            image: BGR image (will NOT be modified in-place).
            region: (x, y, w, h) of the watermark region.
            strength: Blend strength (0.0 = keep original, 1.0 = fully inpainted).
            method: Inpaint method ("gaussian", "telea", or "ns").
            inpaint_radius: Radius for cv2.inpaint.
            padding: Context padding around region in pixels.

        Returns:
            Cleaned BGR image.
        """
        result = image.copy()
        x, y, rw, rh = region

        if rw < 4 or rh < 4:
            return result

        strength = max(0.0, min(1.0, strength))
        if strength < 0.001:
            return result

        # Padded region
        px1 = max(0, x - padding)
        py1 = max(0, y - padding)
        px2 = min(image.shape[1], x + rw + padding)
        py2 = min(image.shape[0], y + rh + padding)

        if (px2 - px1) < 8 or (py2 - py1) < 8:
            return result

        # Inner rect relative to padded
        ix1 = x - px1
        iy1 = y - py1

        # Get alpha map (interpolated if needed)
        source_alpha = self._alpha_large
        interp = cv2.INTER_LINEAR if rw > source_alpha.shape[1] else cv2.INTER_AREA
        alpha_resized = cv2.resize(source_alpha, (rw, rh), interpolation=interp)

        # Compute gradient mask from alpha
        grad_x = cv2.Sobel(alpha_resized, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(alpha_resized, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(grad_x, grad_y)

        grad_min, grad_max = grad_mag.min(), grad_mag.max()
        if grad_max <= grad_min:
            return result

        # Normalize and apply gamma correction
        grad_norm = (grad_mag - grad_min) / (grad_max - grad_min)
        grad_weight = np.sqrt(grad_norm)

        # Dilate the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        grad_weight = cv2.dilate(grad_weight, kernel)

        if method == "gaussian":
            # Soft blend with Gaussian blur
            padded_roi = result[py1:py2, px1:px2].copy()
            blurred = cv2.GaussianBlur(padded_roi, (0, 0), sigmaX=2.0)

            # Create weight mask on padded area (only inner region has weights)
            weight_full = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
            weight_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = grad_weight * strength

            weight_3d = weight_full[:, :, np.newaxis]
            blended = padded_roi.astype(np.float32) * (1 - weight_3d) + blurred.astype(np.float32) * weight_3d
            result[py1:py2, px1:px2] = blended.astype(np.uint8)
        else:
            # OpenCV inpainting (TELEA or NS)
            inpaint_flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS

            # Create binary mask from gradient weight
            binary_mask = (grad_weight * 255).astype(np.uint8)
            _, binary_mask = cv2.threshold(binary_mask, 30, 255, cv2.THRESH_BINARY)

            # Expand mask to padded region
            mask_full = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            mask_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = binary_mask

            padded_roi = result[py1:py2, px1:px2].copy()
            inpainted = cv2.inpaint(padded_roi, mask_full, inpaint_radius, inpaint_flag)

            # Blend with strength
            weight_full = np.zeros((py2 - py1, px2 - px1), dtype=np.float32)
            weight_full[iy1 : iy1 + rh, ix1 : ix1 + rw] = grad_weight * strength
            weight_3d = weight_full[:, :, np.newaxis]

            blended = padded_roi.astype(np.float32) * (1 - weight_3d) + inpainted.astype(np.float32) * weight_3d
            result[py1:py2, px1:px2] = blended.astype(np.uint8)

        return result


def detect_sparkle_confidence(image_path: Path) -> float | None:
    """Visible-sparkle detection confidence for a file, for provenance use.

    Loads the image with cv2 and runs :meth:`GeminiEngine.detect_watermark`.
    Returns the NCC confidence in [0, 1], or None if the image cannot be read
    (cv2 returns None for unsupported containers such as HEIC). Kept here so the
    cv2 dependency stays in this module; callers apply their own threshold.
    """
    from remove_ai_watermarks import image_io

    img = image_io.imread(image_path)
    if img is None:
        return None
    return float(GeminiEngine().detect_watermark(img).confidence)
