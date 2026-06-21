"""Watermark removal model profiles and the default strength.

Pure configuration and lookup functions with no ML dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Qwen-Image (20B MMDiT, Apache-2.0 code AND weights) base for the ``qwen`` pipeline:
# an img2img alternative to SDXL with native text rendering (incl. CJK). Loaded only
# when ``--pipeline qwen`` is selected; CUDA/cloud-class (does not fit MPS). CERTIFIED
# oracle floors (2026-06-20): OpenAI **0.10** (seed-robust -- clean on seeds 0-4) and
# Google/Gemini **0.25** (seed 0 verified on 2 images; pin a seed in prod, the Gemini
# oracle rate-limits volume seed-repeat). The Gemini floor (0.25) is HIGHER than the
# certified controlnet Gemini floor (0.15); ``resolve_strength(..., pipeline="qwen")``
# now carries this via ``_QWEN_VENDOR_STRENGTH`` (below), so ``--pipeline qwen`` gets the
# right floor automatically -- the old manual "pass --strength 0.25 for Gemini on qwen"
# workaround is retired.
# (Dispatch uses the bare "qwen" literal, matching the sdxl/controlnet sites, so there
# is no QWEN_PROFILE constant -- only the model id is referenced from code.)
QWEN_MODEL_ID = "Qwen/Qwen-Image"

# Canonical pipeline-profile names + the back-compat alias. The plain SDXL img2img
# profile is ``sdxl``; ``default`` is kept as an accepted alias (it was the profile's
# name before ``controlnet`` became the default-selected pipeline, 2026-06-09).
SDXL_PROFILE = "sdxl"
_PROFILE_ALIASES = {"default": SDXL_PROFILE}


def normalize_profile(profile: str) -> str:
    """Canonicalize a pipeline-profile name, resolving the ``default`` -> ``sdxl`` alias."""
    normalized = profile.strip().lower()
    return _PROFILE_ALIASES.get(normalized, normalized)


# The SDXL-native canny ControlNet used by the ``controlnet`` pipeline. The
# ControlNet is an add-on to the SDXL base checkpoint (DEFAULT_MODEL_ID), not a
# separate base model, so both the ``sdxl`` and ``controlnet`` profiles load the
# same base weights and share the same vendor-adaptive strength ladder (see below).
CONTROLNET_CANNY_MODEL = "xinsir/controlnet-canny-sdxl-1.0"

# Vendor-adaptive default denoising strength for the SDXL img2img scrub, overridable
# from the CLI (`--strength`). The right strength depends on which vendor's SynthID is
# present (detected from the C2PA issuer, metadata.synthid_source). The SAME ladder
# applies to BOTH pipelines (`sdxl` plain img2img and `controlnet`) -- see "why one
# ladder" below.
#
# Data basis (see docs/synthid.md sections 2.2 / 5.5): ORACLE-CERTIFIED controlnet floors.
# A 2026-06-14 re-test on the deployed Modal worker (the production controlnet pipeline)
# LOWERED the ladder back to OpenAI 0.10 / Google 0.15: each output verified on its own
# oracle (openai.com/verify for OpenAI, the Google Gemini app for Google), all clean ->
#   - OpenAI 0.10: 2 photoreal images (1402 / 1448 px), SynthID not found on either.
#   - Google 0.15: 2 NATIVE-resolution images (both 2816x1536), SynthID not found on
#     either -- this directly retires the earlier "native ~2816 likely needs ~0.35+"
#     guess, which was speculative and never oracle-checked at that resolution.
# This supersedes the 2026-06-04 cert (OpenAI 0.20 / Google 0.30), whose higher floor a
# pixel-fidelity sweep showed was ~2x the removal floor and over-regenerated for no
# efficacy gain (Google MAE -20% at 0.15 vs 0.30, no SynthID returning). Unknown vendor
# tracks the Google (more robust watermark) value -> 0.15, still safe-by-default and the
# floor that real (no-vendor) photos hit, so it also minimizes damage when there is in
# fact nothing to remove. CAVEAT: the re-test is n=2 per vendor on photoreal / landscape
# content; FLAT-GRAPHIC hard cases (the historical `sdxl` weak spot) were NOT in the
# sample, so if an oracle still reads SynthID on a flat output, raise `--strength`.
#
# Why ONE ladder for both pipelines (2026-06-09): the certification was run on
# controlnet, and it does NOT transfer to `sdxl` by symmetry -- the two pipelines have
# OPPOSITE hard cases (controlnet leaves SynthID on photoreal, `sdxl` leaves it on flat
# graphics; the content-x-pipeline table in docs/synthid.md §5.1). BUT on its OWN hard
# case (flat fills) `sdxl` is the WEAKER remover -- plain img2img at low strength barely
# perturbs a flat region -- so it needs AT LEAST as much strength as controlnet, not
# less. Hence the certified controlnet floor is the right floor for `sdxl` too. The
# higher strength costs little quality where it matters: `controlnet` is now the default
# pipeline, so `sdxl` is reached only for structure-less inputs (via `--auto`) or an
# explicit `--pipeline sdxl`, where over-regeneration has no faces/text to damage. NOTE:
# this is a MARGIN argument for `sdxl`, not a fresh certification -- there is no local
# SynthID detector, so if an oracle still reads SynthID on a flat `sdxl` output, raise
# `--strength`.
OPENAI_STRENGTH = 0.10
GEMINI_STRENGTH = 0.15
UNKNOWN_STRENGTH = 0.15
# Backwards-compatible alias: the vendor-unknown value (what a caller gets without a
# detected vendor). Kept as DEFAULT_STRENGTH for existing references.
DEFAULT_STRENGTH = UNKNOWN_STRENGTH

# Detected-vendor -> default strength. Vendor strings come from `vendor_for_strength`.
_VENDOR_STRENGTH = {"openai": OPENAI_STRENGTH, "google": GEMINI_STRENGTH}

# Qwen has its OWN certified floors (Modal A100-80GB, 2026-06-20), DIFFERENT from the
# SDXL ladder above: OpenAI 0.10 (seed-robust), Gemini 0.25 (HIGHER than controlnet's
# 0.15 -- the 20B MMDiT perturbs less per denoising step, so it needs more strength to
# clear Gemini SynthID). Unknown vendor tracks the higher (Gemini) value, safe-by-default.
# `resolve_strength(..., pipeline="qwen")` uses this table so `--pipeline qwen` carries the
# right floor automatically -- retiring the old manual "pass --strength 0.25 for Gemini on
# qwen" workaround.
QWEN_OPENAI_STRENGTH = 0.10
QWEN_GEMINI_STRENGTH = 0.25
QWEN_UNKNOWN_STRENGTH = 0.25
_QWEN_VENDOR_STRENGTH = {"openai": QWEN_OPENAI_STRENGTH, "google": QWEN_GEMINI_STRENGTH}


def strength_default_help() -> str:
    """One-line description of the vendor-adaptive default, derived from the constants.

    Single source of truth for the CLI ``--strength`` help so the numbers can never
    drift from the actual ladder (they did once when the per-pipeline split was unified).
    """
    return (
        f"vendor-adaptive (OpenAI {OPENAI_STRENGTH} / Google {GEMINI_STRENGTH} / "
        f"unknown {UNKNOWN_STRENGTH}, from the C2PA issuer; same ladder for both pipelines)"
    )


def resolve_strength(strength: float | None, vendor: str | None = None, pipeline: str | None = None) -> float:
    """Resolve the denoising strength, applying the vendor default when unset.

    ``None`` means "the user did not pass ``--strength``", which resolves
    **vendor-adaptively**: ``vendor`` (``"openai"`` / ``"google"`` / None, from
    ``vendor_for_strength``) selects the per-vendor floor. The ``sdxl`` and ``controlnet``
    pipelines share ONE ladder (``OPENAI_STRENGTH`` / ``GEMINI_STRENGTH`` /
    ``UNKNOWN_STRENGTH`` -- see the module comment for why); ``qwen`` has its OWN higher
    ladder (``_QWEN_VENDOR_STRENGTH``, Gemini 0.25 vs controlnet 0.15), selected when
    ``pipeline`` normalizes to ``"qwen"``. An explicit value always wins (including
    ``0.0`` -- the check is ``is None``, not falsiness). Shared by the CLI (for display)
    and the engine (for execution) so the two never disagree -- both must pass the SAME
    ``vendor`` and ``pipeline``.
    """
    if strength is not None:
        return strength
    if pipeline is not None and normalize_profile(pipeline) == "qwen":
        return _QWEN_VENDOR_STRENGTH.get(vendor or "", QWEN_UNKNOWN_STRENGTH)
    return _VENDOR_STRENGTH.get(vendor or "", UNKNOWN_STRENGTH)


def vendor_for_strength(image_path: Path) -> Literal["openai", "google"] | None:
    """Detect the SynthID vendor for strength selection: ``"openai"`` / ``"google"`` / None.

    Reads the C2PA SynthID proxy (``metadata.synthid_source``) on the ORIGINAL input,
    so it must run before any pass that strips metadata. When both issuers appear (a
    rare multi-sign anomaly) Google wins -- the more-robust watermark -> safer (higher)
    strength. Returns None when metadata is stripped or the issuer is neither vendor,
    which maps to ``UNKNOWN_STRENGTH``. Lazy-imports ``metadata`` to keep this module
    dependency-light.
    """
    try:
        from remove_ai_watermarks.metadata import synthid_source

        src = (synthid_source(image_path) or "").lower()
    except Exception:  # metadata unreadable -> treat as unknown vendor
        return None
    if "google" in src:
        return "google"
    if "openai" in src:
        return "openai"
    return None
