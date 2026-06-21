"""Tests for cross-platform and cross-device compatibility.

Verifies that device detection, MPS fallback, and platform-specific
code paths work correctly on CPU, MPS (macOS), and CUDA (Linux/Windows).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from remove_ai_watermarks.noai.progress import is_mps_error
from remove_ai_watermarks.noai.utils import get_image_format, is_supported_format
from remove_ai_watermarks.noai.watermark_profiles import (
    DEFAULT_STRENGTH,
    GEMINI_STRENGTH,
    OPENAI_STRENGTH,
    UNKNOWN_STRENGTH,
    normalize_profile,
    resolve_strength,
    strength_default_help,
)
from remove_ai_watermarks.noai.watermark_remover import get_device, is_watermark_removal_available

# ── Device detection ────────────────────────────────────────────────


class TestDeviceDetection:
    """Tests for get_device() across platforms."""

    def test_returns_valid_device(self):
        device = get_device()
        assert device in ("cpu", "mps", "cuda", "xpu")

    def test_cpu_fallback_when_no_gpu(self):
        """On CI / machines without GPU, should fall back to cpu or mps."""
        device = get_device()
        # Just verify it doesn't crash and returns a valid string
        assert isinstance(device, str)

    @patch("remove_ai_watermarks.noai.watermark_remover._HAS_TORCH", False)
    def test_no_torch_returns_cpu(self):
        assert get_device() == "cpu"

    def test_xpu_selected_when_available(self):
        """An XPU-enabled torch (no CUDA) routes to the Intel GPU backend.

        The whole torch module is mocked so the smoke-test ops succeed without
        any real device; cuda must read False so the cuda branch is skipped.
        """
        fake_torch = MagicMock()
        fake_torch.cuda.is_available.return_value = False
        fake_torch.xpu.is_available.return_value = True
        with patch("remove_ai_watermarks.noai.watermark_remover.torch", fake_torch):
            assert get_device() == "xpu"
        fake_torch.tensor.assert_called_with([1.0], device="xpu")

    def test_init_accepts_xpu_and_selects_fp16(self):
        """WatermarkRemover accepts device='xpu' and picks fp16 (not fp32)."""
        if not is_watermark_removal_available():
            pytest.skip("torch/diffusers not installed")
        import torch

        from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover

        remover = WatermarkRemover(device="xpu")
        assert remover.device == "xpu"
        assert remover.torch_dtype == torch.float16

    def test_seed_generator_falls_back_to_cpu_when_device_rng_unsupported(self):
        """A device with no RNG backend (e.g. some torch-xpu builds) falls back
        to a CPU generator instead of raising when --seed is used."""
        from remove_ai_watermarks.noai import watermark_remover as wr

        def fake_generator(device="cpu"):
            if device == "xpu":
                raise RuntimeError("Device type xpu is not supported for torch.Generator()")
            gen = MagicMock()
            gen.manual_seed.return_value = f"gen:{device}"
            return gen

        fake_torch = MagicMock()
        fake_torch.Generator.side_effect = fake_generator
        with patch.object(wr, "torch", fake_torch):
            assert wr._make_seed_generator("xpu", 123) == "gen:cpu"
            assert wr._make_seed_generator("cuda", 123) == "gen:cuda"


class TestMpsErrorDetection:
    """Tests for MPS error detection helper."""

    def test_detects_mps_error(self):
        err = RuntimeError("MPS backend out of memory")
        assert is_mps_error(err) is True

    def test_non_mps_error(self):
        err = RuntimeError("CUDA out of memory")
        assert is_mps_error(err) is False

    def test_generic_error(self):
        err = RuntimeError("something went wrong")
        assert is_mps_error(err) is False


# ── Model profiles ──────────────────────────────────────────────────


class TestModelProfiles:
    """Tests for watermark_profiles.py profile-name normalization."""

    def test_canonical_profiles_unchanged(self):
        assert normalize_profile("sdxl") == "sdxl"
        assert normalize_profile("controlnet") == "controlnet"
        assert normalize_profile("qwen") == "qwen"

    def test_default_alias_resolves_to_sdxl(self):
        # "default" is the legacy alias for "sdxl" (back-compat for existing scripts).
        assert normalize_profile("default") == "sdxl"

    def test_normalize_is_case_and_whitespace_insensitive(self):
        assert normalize_profile("  Default ") == "sdxl"
        assert normalize_profile("CONTROLNET") == "controlnet"


class _StubImage:
    """Minimal PIL.Image stand-in: just the ``width``/``height`` the pure helper reads."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height


class TestQwenKwargs:
    """_build_qwen_kwargs is pure (no torch); guards the Qwen-Image call shape.

    watermark_remover imports torch under a try/except, so the module (and this pure
    helper) imports fine in the core+dev CI env where torch is absent.
    """

    def test_uses_true_cfg_not_guidance_scale(self):
        from remove_ai_watermarks.noai.watermark_remover import _build_qwen_kwargs

        gen = object()
        img = _StubImage(2816, 1536)
        kwargs = _build_qwen_kwargs(img, strength=0.3, num_inference_steps=40, true_cfg_scale=4.0, generator=gen)
        # Qwen uses true_cfg_scale, NOT SDXL's guidance_scale.
        assert kwargs["true_cfg_scale"] == 4.0
        assert "guidance_scale" not in kwargs
        # The scrub still comes from strength; image + generator pass through.
        assert kwargs["strength"] == 0.3
        assert kwargs["image"] is img
        assert kwargs["generator"] is gen
        # Faithful-regeneration prompt + an explicit negative prompt.
        assert kwargs["prompt"]
        assert kwargs["negative_prompt"]

    def test_passes_explicit_aspect_preserving_size(self):
        # Without height/width the pipeline defaults to 1024x1024 and squishes non-square
        # input (the abba mixed-seam regression). Both already multiples of 16 -> unchanged.
        from remove_ai_watermarks.noai.watermark_remover import _build_qwen_kwargs

        kwargs = _build_qwen_kwargs(
            _StubImage(2816, 1536), strength=0.25, num_inference_steps=40, true_cfg_scale=4.0, generator=None
        )
        assert kwargs["width"] == 2816
        assert kwargs["height"] == 1536

    def test_qwen_target_size_floors_to_multiple_of_16(self):
        from remove_ai_watermarks.noai.watermark_remover import _qwen_target_size

        assert _qwen_target_size(2816, 1536) == (2816, 1536)  # already /16
        assert _qwen_target_size(1122, 1402) == (1120, 1392)  # floored
        assert _qwen_target_size(10, 10) == (16, 16)  # min clamp, never 0

    def test_qwen_model_id_is_qwen_image(self):
        from remove_ai_watermarks.noai.watermark_profiles import QWEN_MODEL_ID

        assert QWEN_MODEL_ID == "Qwen/Qwen-Image"


class TestResolveStrength:
    """resolve_strength applies the vendor default only when strength is unset."""

    def test_none_is_vendor_adaptive(self):
        # No vendor -> unknown default; OpenAI lower, Google == unknown. The sdxl/controlnet
        # pipelines share this ladder (the certified controlnet floors); qwen has its own
        # (see test_qwen_pipeline_uses_its_own_higher_ladder).
        assert resolve_strength(None) == UNKNOWN_STRENGTH
        assert resolve_strength(None, "openai") == OPENAI_STRENGTH
        assert resolve_strength(None, "google") == GEMINI_STRENGTH
        assert resolve_strength(None, None) == UNKNOWN_STRENGTH
        # An unrecognized vendor string falls through to the unknown default.
        assert resolve_strength(None, "adobe") == UNKNOWN_STRENGTH
        # sdxl/controlnet pipelines (and the "default" alias) use the same shared ladder.
        assert resolve_strength(None, "google", "controlnet") == GEMINI_STRENGTH
        assert resolve_strength(None, "google", "sdxl") == GEMINI_STRENGTH

    def test_qwen_pipeline_uses_its_own_higher_ladder(self):
        # Qwen's certified Gemini floor (0.25) is HIGHER than controlnet's (0.15); OpenAI
        # matches (0.10). Unknown vendor on qwen tracks the higher Gemini value. This retires
        # the old manual "pass --strength 0.25 for Gemini on qwen" workaround.
        from remove_ai_watermarks.noai.watermark_profiles import QWEN_GEMINI_STRENGTH, QWEN_OPENAI_STRENGTH

        assert QWEN_GEMINI_STRENGTH == 0.25
        assert QWEN_OPENAI_STRENGTH == 0.10
        assert resolve_strength(None, "google", "qwen") == QWEN_GEMINI_STRENGTH
        assert resolve_strength(None, "openai", "qwen") == QWEN_OPENAI_STRENGTH
        assert resolve_strength(None, None, "qwen") == QWEN_GEMINI_STRENGTH  # unknown -> higher floor
        assert resolve_strength(None, "google", "qwen") > resolve_strength(None, "google", "controlnet")
        # An explicit strength still wins on qwen.
        assert resolve_strength(0.12, "google", "qwen") == 0.12

    def test_ladder_is_the_certified_controlnet_floors(self):
        # The unified ladder == the oracle-certified controlnet floors. Lowered on the
        # 2026-06-14 Modal re-test (OpenAI 0.10, Google/unknown 0.15); Google is the
        # more-robust watermark, so it is higher.
        assert OPENAI_STRENGTH == 0.10
        assert GEMINI_STRENGTH == 0.15
        assert UNKNOWN_STRENGTH == 0.15
        assert OPENAI_STRENGTH < GEMINI_STRENGTH

    def test_default_strength_alias_is_unknown_vendor_value(self):
        assert DEFAULT_STRENGTH == UNKNOWN_STRENGTH
        assert OPENAI_STRENGTH < UNKNOWN_STRENGTH

    def test_strength_default_help_derives_from_constants(self):
        # The CLI --strength help is built from this, so it can never drift from the ladder.
        h = strength_default_help()
        assert str(OPENAI_STRENGTH) in h
        assert str(GEMINI_STRENGTH) in h
        assert str(UNKNOWN_STRENGTH) in h

    def test_explicit_value_overrides_vendor(self):
        assert resolve_strength(0.3) == 0.3
        assert resolve_strength(0.3, "openai") == 0.3

    def test_explicit_zero_is_respected_not_treated_as_unset(self):
        # 0.0 is falsy but explicit -- must not fall through to the vendor default
        # (the old `strength or DEFAULT` bug would have). Range validation lives in
        # remove_watermark, not here.
        assert resolve_strength(0.0) == 0.0
        assert resolve_strength(0.0, "google") == 0.0


class TestVendorForStrength:
    """vendor_for_strength normalizes the C2PA SynthID proxy to openai/google/None."""

    @staticmethod
    def _patch(value):
        return patch("remove_ai_watermarks.metadata.synthid_source", return_value=value)

    def test_openai(self):
        from remove_ai_watermarks.noai.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI"):
            assert vendor_for_strength(Path("x.png")) == "openai"

    def test_google(self):
        from remove_ai_watermarks.noai.watermark_profiles import vendor_for_strength

        with self._patch("Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    def test_both_issuers_google_wins(self):
        # The more-robust watermark wins -> safer (higher) strength.
        from remove_ai_watermarks.noai.watermark_profiles import vendor_for_strength

        with self._patch("OpenAI, Google"):
            assert vendor_for_strength(Path("x.png")) == "google"

    def test_none_when_no_synthid_source(self):
        from remove_ai_watermarks.noai.watermark_profiles import vendor_for_strength

        with self._patch(None):
            assert vendor_for_strength(Path("x.png")) is None

    def test_unreadable_metadata_is_none(self):
        from remove_ai_watermarks.noai.watermark_profiles import vendor_for_strength

        with patch("remove_ai_watermarks.metadata.synthid_source", side_effect=OSError):
            assert vendor_for_strength(Path("x.png")) is None


# ── Format utilities ────────────────────────────────────────────────


class TestFormatUtils:
    """Tests for utils.py format helpers."""

    def test_supported_png(self, tmp_path):
        assert is_supported_format(tmp_path / "test.png")

    def test_supported_jpg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpg")

    def test_supported_jpeg(self, tmp_path):
        assert is_supported_format(tmp_path / "test.jpeg")

    def test_supported_webp(self, tmp_path):
        assert is_supported_format(tmp_path / "test.webp")

    def test_unsupported_bmp(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.bmp")

    def test_unsupported_gif(self, tmp_path):
        assert not is_supported_format(tmp_path / "test.gif")

    def test_get_format_png(self, tmp_path):
        assert get_image_format(tmp_path / "x.png") == "PNG"

    def test_get_format_jpg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpg") == "JPEG"

    def test_get_format_jpeg(self, tmp_path):
        assert get_image_format(tmp_path / "x.jpeg") == "JPEG"

    def test_get_format_webp_defaults_png(self, tmp_path):
        # .webp falls through to PNG in current implementation
        assert get_image_format(tmp_path / "x.webp") == "PNG"


# ── Availability checks ────────────────────────────────────────────


class TestAvailability:
    """Tests for dependency availability checks."""

    def test_watermark_removal_available(self):
        # Reflects the actual environment: True iff torch + diffusers (the gpu
        # extra) are importable. The core+dev CI env has no diffusers, so this
        # must not assume the full stack is present.
        import importlib.util

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_watermark_removal_available() is expected

    def test_invisible_is_available(self):
        import importlib.util

        from remove_ai_watermarks.invisible_engine import is_available

        expected = all(importlib.util.find_spec(m) is not None for m in ("torch", "diffusers"))
        assert is_available() is expected


# ── Platform-specific path handling ─────────────────────────────────


class TestPlatformPaths:
    """Verify path handling works on current platform."""

    def test_pathlib_works_for_assets(self):
        from pathlib import Path

        asset_dir = Path(__file__).parent.parent / "src" / "remove_ai_watermarks" / "assets"
        assert (asset_dir / "gemini_bg_48.png").exists()
        assert (asset_dir / "gemini_bg_96.png").exists()

    def test_asset_loading_works(self):
        """Verify embedded assets load correctly (critical for packaging)."""
        from remove_ai_watermarks.gemini_engine import GeminiEngine

        engine = GeminiEngine()
        # If we get here without error, asset loading works
        assert engine._alpha_small.shape == (48, 48)
        assert engine._alpha_large.shape == (96, 96)


class TestFp16VaeFix:
    """The plain SDXL img2img pipeline must swap in the fp16-fixed VAE on fp16
    GPUs to avoid the NaN/all-black decode (issue #29). Pure decision logic, no
    torch or model download needed."""

    DEFAULT = "stabilityai/stable-diffusion-xl-base-1.0"

    def test_default_sdxl_on_fp16_needs_fix(self):
        from remove_ai_watermarks.noai.watermark_remover import _needs_fp16_vae_fix

        assert _needs_fp16_vae_fix(self.DEFAULT, self.DEFAULT, is_fp16=True) is True

    def test_fp32_does_not_need_fix(self):
        """cpu/mps run fp32, where the stock SDXL VAE is fine."""
        from remove_ai_watermarks.noai.watermark_remover import _needs_fp16_vae_fix

        assert _needs_fp16_vae_fix(self.DEFAULT, self.DEFAULT, is_fp16=False) is False

    def test_non_default_model_keeps_own_vae(self):
        """A custom (non-SDXL) checkpoint must not get the SDXL-specific VAE."""
        from remove_ai_watermarks.noai.watermark_remover import _needs_fp16_vae_fix

        assert _needs_fp16_vae_fix("runwayml/stable-diffusion-v1-5", self.DEFAULT, is_fp16=True) is False


class TestDegenerateOutputGuard:
    """The fp16 black-output safety net (#29/#41): detect an all-black/NaN frame so
    ``remove_watermark`` can retry in fp32. Pure image statistics, no model needed."""

    def test_all_black_is_degenerate(self):
        from remove_ai_watermarks.noai.watermark_remover import _is_degenerate_image

        black = Image.fromarray(np.zeros((64, 64, 3), np.uint8))
        assert _is_degenerate_image(black) is True

    def test_normal_image_is_not_degenerate(self):
        from remove_ai_watermarks.noai.watermark_remover import _is_degenerate_image

        rng = np.random.default_rng(0)
        normal = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        assert _is_degenerate_image(normal) is False

    def test_dark_but_textured_image_is_not_degenerate(self):
        """A legitimately dark photo with real detail must NOT be flagged (variance guard)."""
        from remove_ai_watermarks.noai.watermark_remover import _is_degenerate_image

        rng = np.random.default_rng(1)
        dark = Image.fromarray(rng.integers(0, 40, (64, 64, 3), dtype=np.uint8))
        assert _is_degenerate_image(dark) is False
