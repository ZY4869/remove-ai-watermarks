"""Optional GFPGAN face-polish post-pass for the invisible removal pipeline.

The diffusion removal pass scrubs the watermark everywhere but lets faces drift in
likeness (canny holds face *structure*, not *identity*). This module sharpens and
re-synthesizes each face from GFPGAN's StyleGAN2 prior, running on the
DIFFUSION-CLEANED image -- not on the original.

**Why "cleaned, not original":** an earlier version of this module ran GFPGAN on the
ORIGINAL (watermarked) image and was oracle-confirmed (2026-06-04) to re-introduce
SynthID into the face regions, because GFPGAN at fidelity weight 0.5 blends ~half
the input pixels with the prior, and SynthID is robust to that partial blend. The
fix is to feed GFPGAN the already-clean image -- whatever pixels it preserves are
already SynthID-free, so the composited face stays clean. Identity is recovered from
the StyleGAN2 prior conditioned on the already-drifted cleaned face (not on the
original face), so identity fidelity is somewhat lower than the would-have-been
identity-as-embedding stack (PhotoMaker-V1), but the upstream PhotoMaker package has
significant compatibility issues with the diffusers version we ship, so this is the
shipping path.

Both GFPGAN (Apache-2.0) and its RetinaFace detector (MIT) are commercial-safe.
The GFPGANv1.4 weights and the RetinaFace detector download on first use and are
never bundled. Requires the optional ``restore`` extra (gfpgan/facexlib/basicsr).
"""

# cv2/torch/gfpgan boundary: gfpgan/basicsr/facexlib ship no usable type stubs and
# this module wraps cv2 (feather composite) and torch; relax the unknown-type rules
# for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
import sys
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# GFPGANv1.4 weights (Apache-2.0). Downloaded on first use, never bundled.
_GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
_GFPGAN_ARCH = "clean"
_GFPGAN_CHANNEL_MULTIPLIER = 2

_restorer: Any | None = None
_restorer_lock = threading.Lock()


def is_available() -> bool:
    """True when the optional GFPGAN face-restoration deps are importable."""
    import importlib.util

    return importlib.util.find_spec("gfpgan") is not None and importlib.util.find_spec("facexlib") is not None


def _apply_basicsr_shim() -> None:
    """Install the ``torchvision.transforms.functional_tensor`` compatibility shim.

    basicsr (a GFPGAN dependency) imports ``rgb_to_grayscale`` from the
    ``torchvision.transforms.functional_tensor`` module, which newer torchvision
    removed. Recreate that module pointing at the public functional API. Idempotent:
    only installed when the real module is missing.
    """
    import importlib.util

    if importlib.util.find_spec("torchvision.transforms.functional_tensor") is not None:
        return
    if "torchvision.transforms.functional_tensor" in sys.modules:
        return

    import types

    import torchvision.transforms.functional as tv_functional

    shim = types.ModuleType("torchvision.transforms.functional_tensor")
    shim.rgb_to_grayscale = tv_functional.rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = shim


def _select_device() -> str:
    """Pick the GFPGAN device: CUDA when present, else CPU.

    The pip GFPGANer has an MPS device-mismatch bug, and this is a cheap post-pass
    on a few face crops, so MPS is deliberately avoided -- CPU is the safe default
    on Apple silicon.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception as e:
        logger.debug("face_restore: CUDA probe failed (%s); using CPU", e)
    return "cpu"


def _get_restorer() -> Any:
    """Return the lazily-built GFPGANer singleton (downloads weights on first use)."""
    global _restorer
    if _restorer is not None:
        return _restorer
    with _restorer_lock:
        if _restorer is None:
            _apply_basicsr_shim()
            from gfpgan import GFPGANer

            _restorer = GFPGANer(
                model_path=_GFPGAN_MODEL_URL,
                upscale=1,
                arch=_GFPGAN_ARCH,
                channel_multiplier=_GFPGAN_CHANNEL_MULTIPLIER,
                device=_select_device(),
            )
    return _restorer


def _composite_faces(
    base_bgr: NDArray[Any],
    restored_bgr: NDArray[Any],
    boxes: list[tuple[float, float, float, float]],
    pad: int = 14,
    feather_div: int = 6,
) -> NDArray[Any]:
    """Feather-composite restored face regions from ``restored_bgr`` into ``base_bgr``.

    Pure cv2/numpy helper (no gfpgan), so it is unit-testable without the model.
    For each ``(x1, y1, x2, y2)`` box: pad and clip to the image, build a Gaussian-
    feathered rectangular alpha, and blend ``restored * a + base * (1 - a)``. Boxes
    that fall fully outside the image (or an empty list) leave ``base_bgr`` unchanged.
    """
    import cv2
    import numpy as np

    out = base_bgr.astype(np.float32)
    h, w = base_bgr.shape[:2]

    for box in boxes:
        x1 = int(box[0]) - pad
        y1 = int(box[1]) - pad
        x2 = int(box[2]) + pad
        y2 = int(box[3]) + pad
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            continue

        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[y1:y2, x1:x2] = 1.0
        k = max(3, (min(bw, bh) // feather_div) | 1)  # odd kernel >= 3
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = alpha[:, :, None]
        out = restored_bgr.astype(np.float32) * alpha + out * (1.0 - alpha)

    return np.clip(out, 0, 255).astype(np.uint8)


def restore_faces(
    original_bgr: NDArray[Any],  # legacy positional kept for API stability; unused
    cleaned_bgr: NDArray[Any],
    weight: float = 0.5,
    pad: int = 14,
    feather_div: int = 6,
) -> NDArray[Any]:
    """Restore face identity in ``cleaned_bgr`` by running GFPGAN on the CLEANED image.

    GFPGAN is a fidelity-restoration net: it sharpens and re-synthesizes face details
    from its StyleGAN2 prior conditioned on the INPUT face. **Running it on the
    diffusion-cleaned image (not the original)** is what makes this pass SynthID-safe:
    the input pixels GFPGAN derives from are already SynthID-free, so the partial
    pixel-blend at the default weight 0.5 cannot re-introduce the watermark.

    The earlier version of this module ran GFPGAN on the ORIGINAL (watermarked) image
    and was oracle-confirmed (2026-06-04) to re-introduce SynthID into the face
    regions. The fix is the single-line source swap below.

    The ``original_bgr`` argument is kept for positional API stability with the
    earlier signature but is no longer used; pass it for legacy callers, ignore it
    in new code.

    Args:
        original_bgr: UNUSED (legacy; kept for positional API stability).
        cleaned_bgr: The diffusion-cleaned image as cv2 BGR (faces drifted from the
            removal pass). GFPGAN runs on THIS, polishing each face without changing
            the watermark state of the source pixels.
        weight: GFPGAN fidelity weight (0-1); lower = more StyleGAN2 regeneration of
            the face from the prior.
        pad: Pixels to grow each face box before compositing.
        feather_div: Larger = sharper composite edge (box-min // feather_div kernel).
    """
    restorer = _get_restorer()
    _, _, restored_img = restorer.enhance(
        cleaned_bgr,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=weight,
    )

    det_faces = getattr(restorer.face_helper, "det_faces", None) or []
    boxes = [(float(b[0]), float(b[1]), float(b[2]), float(b[3])) for b in det_faces]
    if not boxes:
        logger.debug("face_restore: no faces detected; returning cleaned image unchanged")
        return cleaned_bgr

    return _composite_faces(cleaned_bgr, restored_img, boxes, pad=pad, feather_div=feather_div)
