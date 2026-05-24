"""Tests for the provenance identifier (identify.py).

Pure attribution logic is unit-tested directly; end-to-end verdicts assert
against the real committed C2PA / IPTC fixtures in data/samples/.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import pytest

from remove_ai_watermarks.identify import (
    ProvenanceReport,
    _ai_tools_in,
    _attribute_platform,
    _issuers_in,
    identify,
)

# Where the lazy import inside identify._visible_sparkle resolves the detector.
_SPARKLE_TARGET = "remove_ai_watermarks.gemini_engine.detect_sparkle_confidence"

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


# ── Pure attribution logic (no file IO) ─────────────────────────────


class TestAttributePlatform:
    def test_openai(self):
        assert "OpenAI" in (_attribute_platform(["OpenAI"]) or "")

    def test_designer_wins_over_openai_backend(self):
        # Microsoft Designer signs as "OpenAI, Microsoft"; name the product.
        platform = _attribute_platform(["OpenAI", "Microsoft"])
        assert platform
        assert "Designer" in platform

    def test_adobe(self):
        assert _attribute_platform(["Adobe"]) == "Adobe Firefly"

    def test_google(self):
        assert "Google" in (_attribute_platform(["Google LLC"]) or "")

    def test_truepic_is_signer_not_generator(self):
        platform = _attribute_platform(["Truepic"])
        assert platform
        assert "signer" in platform.lower()

    def test_empty_is_none(self):
        assert _attribute_platform([]) is None


class TestIssuersIn:
    def test_finds_openai(self):
        assert _issuers_in(b"...OpenAI...trainedAlgorithmicMedia") == ["OpenAI"]

    def test_finds_multiple_sorted(self):
        assert _issuers_in(b"Microsoft and OpenAI") == ["Microsoft", "OpenAI"]

    def test_none_present(self):
        assert _issuers_in(b"just some bytes") == []


class TestAiToolsIn:
    def test_finds_generator(self):
        assert _ai_tools_in(b"...claim_generator Imagen 3...") == ["Imagen"]

    def test_none_present(self):
        assert _ai_tools_in(b"a regular photo, no tools") == []


class TestIdentifyNonPng:
    """Non-PNG containers (JPEG/WebP/AVIF) carry C2PA where the caBX parser can't
    reach; identify recovers issuer + generator via the binary scan. Synthetic
    byte blobs mirror tests/test_metadata.py::TestSynthIDSourceNonPng.
    """

    def _c2pa_jpeg(self, tmp_path: Path, blob: bytes) -> Path:
        path = tmp_path / "img.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe1jumbc2pa" + blob + b"\xff\xd9")
        return path

    def test_google_imagen_jpeg(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"Google Imagen ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "Google" in r.platform
        # Generator recovered from the non-PNG blob shows up in the c2pa signal.
        c2pa_signal = next(s for s in r.signals if s.name == "c2pa")
        assert "Imagen" in c2pa_signal.detail

    def test_openai_jpeg_has_synthid(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI DALL-E ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert any("SynthID" in w for w in r.watermarks)

    def test_c2pa_without_ai_marker_is_unknown(self, tmp_path: Path):
        # Adobe signs C2PA on plain Photoshop edits too. Without an AI digital-
        # source marker, the honest verdict is unknown -- the C2PA watermark is
        # still listed, but is_ai_generated is not asserted True.
        path = self._c2pa_jpeg(tmp_path, b"Adobe ... no ai marker here")
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is None
        assert any("C2PA" in w for w in r.watermarks)
        assert not any("SynthID" in w for w in r.watermarks)


# ── End-to-end verdicts on real fixtures ────────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestIdentifyRealSamples:
    def test_openai_chatgpt(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform
        assert "OpenAI" in r.platform
        assert any("C2PA" in w for w in r.watermarks)
        assert any("SynthID" in w for w in r.watermarks)

    def test_adobe_firefly_has_no_synthid(self):
        r = identify(SAMPLES_DIR / "firefly-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Adobe Firefly"
        assert not any("SynthID" in w for w in r.watermarks)

    def test_iptc_made_with_ai(self):
        # mj-1.png carries the IPTC digitalSourceType "Made with AI" marker.
        r = identify(SAMPLES_DIR / "mj-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert any("IPTC" in w for w in r.watermarks)

    def test_clean_photo_is_unknown_not_clean(self):
        r = identify(SAMPLES_DIR / "not-ai-1.jpeg", check_visible=False)
        assert r.is_ai_generated is None  # never asserted False
        assert r.platform is None
        assert r.confidence == "none"
        assert r.watermarks == []

    def test_strip_caveat_always_present(self):
        r = identify(SAMPLES_DIR / "not-ai-1.jpeg", check_visible=False)
        assert any("not proof" in c for c in r.caveats)

    def test_returns_report_dataclass(self):
        assert isinstance(identify(SAMPLES_DIR / "firefly-1.png", check_visible=False), ProvenanceReport)


# ── Local diffusion parameters (Stable Diffusion / ComfyUI) ─────────


class TestIdentifyLocalParams:
    """A PNG carrying SD-style generation params is attributed to a local pipeline."""

    def test_sd_params_attributed_to_local_pipeline(self, tmp_png_with_ai_metadata: Path):
        r = identify(tmp_png_with_ai_metadata, check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform is not None
        assert "Stable Diffusion" in r.platform
        assert any("generation parameters" in w for w in r.watermarks)

    def test_gen_params_signal_lists_keys(self, tmp_png_with_ai_metadata: Path):
        r = identify(tmp_png_with_ai_metadata, check_visible=False)
        signal = next(s for s in r.signals if s.name == "gen_params")
        assert "parameters" in signal.detail
        assert signal.confidence == "high"

    def test_clean_png_is_unknown(self, tmp_clean_png: Path):
        r = identify(tmp_clean_png, check_visible=False)
        assert r.is_ai_generated is None
        assert r.platform is None
        assert r.confidence == "none"
        assert r.signals == []


# ── Visible-sparkle fallback (mocked detector) ──────────────────────


class TestIdentifyVisibleSparkle:
    """The visible-sparkle signal gates on the corpus-tuned threshold (0.5)."""

    def test_above_threshold_promotes_to_medium(self, tmp_clean_png: Path):
        with patch(_SPARKLE_TARGET, return_value=0.7):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is True
        assert r.confidence == "medium"
        assert r.platform is not None
        assert "Gemini" in r.platform
        signal = next(s for s in r.signals if s.name == "visible_sparkle")
        assert signal.confidence == "medium"

    def test_below_threshold_not_promoted(self, tmp_clean_png: Path):
        with patch(_SPARKLE_TARGET, return_value=0.4):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is None
        assert not any(s.name == "visible_sparkle" for s in r.signals)

    def test_detector_unavailable_does_not_crash(self, tmp_clean_png: Path):
        with patch(_SPARKLE_TARGET, return_value=None):
            r = identify(tmp_clean_png, check_visible=True)
        assert r.is_ai_generated is None
        assert not any(s.name == "visible_sparkle" for s in r.signals)

    def test_check_visible_false_skips_detector(self, tmp_clean_png: Path):
        # Even a strong detection is ignored when the caller opts out.
        with patch(_SPARKLE_TARGET, return_value=0.99) as mock_detect:
            r = identify(tmp_clean_png, check_visible=False)
        mock_detect.assert_not_called()
        assert not any(s.name == "visible_sparkle" for s in r.signals)

    def test_metadata_keeps_high_even_with_sparkle(self, tmp_png_with_ai_metadata: Path):
        # Metadata verdict (high) is not downgraded by an additional sparkle hit.
        with patch(_SPARKLE_TARGET, return_value=0.7):
            r = identify(tmp_png_with_ai_metadata, check_visible=True)
        assert r.confidence == "high"


# ── Caveats and serialization ───────────────────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestIdentifyCaveats:
    def test_openai_hedge_caveat_present(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert any("before the rollout" in c for c in r.caveats)

    def test_synthid_proxy_caveat_present(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert any("not locally" in c for c in r.caveats)

    def test_caveats_are_deduplicated(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert len(r.caveats) == len(set(r.caveats))


class TestReportSerializable:
    def test_report_is_json_serializable(self, tmp_png_with_ai_metadata: Path):
        # The CLI --json path relies on asdict + json.dumps(default=str).
        report = identify(tmp_png_with_ai_metadata, check_visible=False)
        dumped = json.dumps(asdict(report), default=str)
        assert "is_ai_generated" in dumped
