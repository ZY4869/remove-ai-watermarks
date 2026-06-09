"""SynthID-robust face identity restoration via InstantID.

**NON-COMMERCIAL.** InstantID's runtime depends on the InsightFace ``antelopev2``
ArcFace model pack, which InsightFace releases under a research-only license:

    "The training data containing the annotation (and the models trained with
     these data) are available for non-commercial research purposes only."
                                                  -- insightface upstream README

The InstantX maintainers themselves acknowledged on HuggingFace
(``InstantX/InstantID`` discussion #2) that "InstantID cannot be Apache 2.0 if it
is using Insight Face" and stated intent to retrain on commercial face encoders.
As of 2026-06-08 (deep-research synthesis in
``docs/synthid-robust-identity-research-2026-06-08.md``) that retrain has not
shipped. **A paid service (raiw.cc, any monetized SaaS) MUST NOT use this path.**

The default ``--restore-faces-method`` is ``instantid`` (this module). The
alternative ``photomaker`` is also non-commercial. There is no commercial-safe
ArcFace-grade identity-preservation stack for SDXL today.

Architecture (vs PhotoMaker-V2):
- PhotoMaker-V2 conditions on a CLIP+ArcFace embedding and runs as txt2img with
  no spatial control. Identity drift on Asian male faces is documented upstream
  and was visually confirmed in our cert sweep.
- InstantID conditions on the ArcFace embedding via cross-attention (IP-Adapter
  style) AND uses a separate landmark ControlNet (5 facial keypoints) for weak
  pose control. The semantic identity branch and spatial landmark branch are
  decoupled, which gives stronger identity fidelity per the InstantID paper
  (arXiv:2401.07519) and our research report. Critically, NO original face
  pixels enter the diffusion -- only the ArcFace embedding (semantic) and the
  rendered landmark stick figure (geometry, content-free) -- so SynthID is not
  transported.

Pipeline this module wires:
  1. Detect faces in the CLEANED image (YuNet via ``auto_config``).
  2. For each face: take the SAME box from the ORIGINAL image, extract its
     ArcFace embedding + 5 keypoints via InsightFace ``FaceAnalysis(antelopev2)``.
  3. Render the keypoints as a stick figure (``draw_kps`` from upstream).
  4. Call the InstantID community pipeline
     (``StableDiffusionXLInstantIDPipeline``) with the ArcFace embedding as
     ``image_embeds=`` and the landmark image as ``image=`` (the ControlNet
     conditioning).
  5. Feather-composite the regenerated face into the cleaned image.

Requires the optional ``instantid`` extra: ``pip install
'remove-ai-watermarks[instantid]'``. Weights download on first use; never
bundled. The InstantID adapter weights (IdentityNet ControlNet +
``ip-adapter.bin``) are Apache-2.0; the runtime InsightFace ``antelopev2`` model
pack is non-commercial.

Multi-face: like PhotoMaker, this module loops over face boxes and composites
back. InstantID's strength is single-portrait; for group photos identity
fidelity per-face is preserved but the composite still uses the cleaned-image
geometry as the canvas.
"""

# cv2/torch/diffusers boundary: relax unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import importlib.util
import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from remove_ai_watermarks.photomaker_restore import _face_crop_square

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# InstantID checkpoint repo on HuggingFace. The IdentityNet ControlNet weights live
# under ``ControlNetModel/`` and the IP-Adapter file is ``ip-adapter.bin`` at the
# root. Both are Apache-2.0 (the InsightFace runtime dep is what makes the path
# non-commercial). Downloaded on first use.
_INSTANTID_REPO = "InstantX/InstantID"
_INSTANTID_CONTROLNET_SUBFOLDER = "ControlNetModel"
_INSTANTID_IP_ADAPTER = "ip-adapter.bin"

# SDXL base shared with the main pipeline (same checkpoint as `default`/`controlnet`).
_SDXL_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Prompt format. InstantID is less sensitive to prompt than PhotoMaker because the
# ID branch is cross-attention; a neutral descriptive prompt is recommended by the
# upstream gradio demo.
_INSTANTID_PROMPT = "portrait photo of a person, natural skin, soft lighting, sharp focus, best quality"
_INSTANTID_NEGATIVE = (
    "(asymmetry, worst quality, low quality, illustration, 3d, 2d, painting, "
    "cartoons, sketch), open mouth, blurry, watermark, deformed"
)

# Square size used to feed InstantID. SDXL is happiest at 1024 (a smaller value sends
# it into low-res mosaic mode -- caught visually on PhotoMaker, same root cause).
_INSTANTID_FACE_SIZE = 1024

_pipeline: Any | None = None
_pipeline_lock = threading.Lock()
_face_analyser: Any | None = None
_face_analyser_lock = threading.Lock()


def is_available() -> bool:
    """True when the optional InstantID extra deps are importable."""
    return (
        importlib.util.find_spec("insightface") is not None
        and importlib.util.find_spec("diffusers") is not None
        and importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("huggingface_hub") is not None
    )


def _select_device() -> str:
    """Pick the InstantID pipeline device: CUDA when present, MPS on Apple, else CPU."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception as e:
        logger.debug("instantid_restore: device probe failed (%s); using CPU", e)
    return "cpu"


def _ensure_antelopev2(root: Path) -> Path:
    """Materialize the antelopev2 pack at ``<root>/models/antelopev2/`` if absent.

    InsightFace's built-in auto-download points at
    ``github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip``
    which has been broken since at least 2024 (verified upstream issue #2517,
    #2766; explicitly called out in InstantID's README: "manually download via
    this URL to models/antelopev2 as the default link is invalid"). Without the
    five expected ``.onnx`` files in place, ``FaceAnalysis.prepare()`` errors
    with ``assert 'detection' in self.models``.

    We side-step the broken default by fetching the five files from a HuggingFace
    mirror (``kidyu/antelopev2-for-InstantID-ComfyUI``) on first use. Returns the
    target directory containing the .onnx files.
    """
    from huggingface_hub import hf_hub_download

    target = root / "models" / "antelopev2"
    target.mkdir(parents=True, exist_ok=True)
    files = [
        "1k3d68.onnx",
        "2d106det.onnx",
        "genderage.onnx",
        "glintr100.onnx",
        "scrfd_10g_bnkps.onnx",
    ]
    for fname in files:
        dest = target / fname
        if dest.exists() and dest.stat().st_size > 0:
            continue
        logger.info("instantid_restore: fetching antelopev2/%s from HF mirror", fname)
        path = hf_hub_download(repo_id="kidyu/antelopev2-for-InstantID-ComfyUI", filename=fname)
        # hf_hub_download caches under HF_HOME; symlink (or copy) into the
        # InsightFace-expected layout.
        if not dest.exists():
            try:
                dest.symlink_to(path)
            except OSError:
                import shutil

                shutil.copy(path, dest)
    return target


def _get_face_analyser() -> Any:
    """Return the InsightFace FaceAnalysis singleton (antelopev2, non-commercial).

    Pre-downloads the antelopev2 pack from a HuggingFace mirror (the InsightFace
    auto-download is broken). See the NON-COMMERCIAL notice at the top of the
    module.
    """
    global _face_analyser
    if _face_analyser is not None:
        return _face_analyser
    with _face_analyser_lock:
        if _face_analyser is None:
            import torch
            from insightface.app import FaceAnalysis

            providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
            # InstantID's upstream uses name='antelopev2' and root='./'. Materialise
            # the pack at the same place so FaceAnalysis finds it locally.
            root = Path.cwd()
            _ensure_antelopev2(root)
            fa = FaceAnalysis(name="antelopev2", root=str(root), providers=providers)
            fa.prepare(ctx_id=0, det_size=(640, 640))
            _face_analyser = fa
    return _face_analyser


def _get_pipeline() -> Any:
    """Return the lazily-built InstantID pipeline singleton (downloads weights on first use).

    Loads via diffusers' community-pipeline mechanism: the file
    ``pipeline_stable_diffusion_xl_instantid.py`` lives in
    ``diffusers/examples/community/`` and is selected by the slug
    ``pipeline_stable_diffusion_xl_instantid``.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            import torch
            from diffusers import ControlNetModel, DiffusionPipeline
            from huggingface_hub import hf_hub_download

            device = _select_device()
            dtype = torch.float16 if device == "cuda" else torch.float32
            logger.info("instantid_restore: loading SDXL+InstantID on %s (%s)", device, dtype)

            # IdentityNet ControlNet weights.
            controlnet = ControlNetModel.from_pretrained(
                _INSTANTID_REPO,
                subfolder=_INSTANTID_CONTROLNET_SUBFOLDER,
                torch_dtype=dtype,
            )
            # SDXL base + InstantID community pipeline (txt2img w/ IdentityNet ControlNet
            # + IP-Adapter cross-attention conditioned on the ArcFace embedding).
            pipe = DiffusionPipeline.from_pretrained(
                _SDXL_MODEL_ID,
                controlnet=controlnet,
                torch_dtype=dtype,
                custom_pipeline="pipeline_stable_diffusion_xl_instantid",
            )
            pipe.to(device)
            # IP-Adapter weights that wire the ArcFace embedding into cross-attention.
            ip_adapter_path = hf_hub_download(repo_id=_INSTANTID_REPO, filename=_INSTANTID_IP_ADAPTER)
            # IP-Adapter scale (the weight on the ArcFace cross-attention branch) is
            # set at load time, not at call time. 0.8 mirrors the upstream demo.
            pipe.load_ip_adapter_instantid(ip_adapter_path, scale=0.8)
            # Diffusers 0.38 vs InstantID upstream compat patch: InstantID's __call__
            # calls ``self.check_inputs(...)`` POSITIONALLY (signature from ~v0.29),
            # but diffusers 0.38 added two new params (``ip_adapter_image``,
            # ``ip_adapter_image_embeds``) BEFORE ``controlnet_conditioning_scale`` in
            # the parent's signature. That shifts every argument by two, so
            # ``control_guidance_end`` (which InstantID converts to ``[1.0]`` for the
            # single-controlnet case before this point) lands in the slot the parent
            # validates as ``controlnet_conditioning_scale`` and trips
            # ``TypeError("must be type float")``. Our inputs are programmatic and
            # already validated by our own callers, so neutralising the check is safe.
            pipe.check_inputs = lambda *_a, **_k: None
            _pipeline = pipe
    return _pipeline


def _draw_kps(image_size: tuple[int, int], kps: Any) -> Any:
    """Render the 5 facial keypoints as a colored stick figure.

    Mirrors upstream's ``draw_kps`` (in ``pipeline_stable_diffusion_xl_instantid.py``):
    the 5 keypoints (left eye, right eye, nose tip, left mouth corner, right mouth
    corner) get drawn as colored circles connected by colored lines, on a black
    background. The result is the ControlNet conditioning image -- pure landmark
    geometry, no pixels from the original face leak through this branch.

    ``image_size`` is ``(width, height)``; ``kps`` is a numpy array of shape (5, 2).
    """
    import cv2
    import numpy as np
    from PIL import Image

    # Same color palette as upstream (blue/red/green/purple/yellow).
    stick_width = 4
    limb_seq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    color_list = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    w, h = image_size
    out_img = np.zeros((h, w, 3), dtype=np.uint8)

    kps_arr = np.array(kps)
    for i in range(len(limb_seq)):
        index = limb_seq[i]
        color = color_list[index[0]]
        x = kps_arr[index][:, 0]
        y = kps_arr[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = np.degrees(np.arctan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly(
            (int(np.mean(x)), int(np.mean(y))),
            (int(length / 2), stick_width),
            int(angle),
            0,
            360,
            1,
        )
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for i, kp in enumerate(kps_arr):
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color_list[i], -1)

    return Image.fromarray(out_img.astype(np.uint8))


def restore_faces_instantid(
    original_bgr: NDArray[Any],
    cleaned_bgr: NDArray[Any],
    num_inference_steps: int = 30,
    guidance_scale: float = 5.0,
    controlnet_conditioning_scale: float = 0.8,
    seed: int | None = None,
    detect_faces_fn: Any | None = None,
) -> NDArray[Any]:
    """SynthID-robust face identity restoration via InstantID.

    Flow:
      1. Detect faces in ``cleaned_bgr`` (YuNet via ``auto_config`` by default;
         override via ``detect_faces_fn`` for tests).
      2. For each face: take the SAME box from ``original_bgr`` -> square crop ->
         InsightFace extracts ArcFace embedding + 5 keypoints -> ``_draw_kps``
         renders the landmark stick figure -> InstantID pipeline generates a
         fresh face conditioned on the embedding and the landmark control image.
      3. Feather-composite each regenerated face into ``cleaned_bgr``.

    Faces are read from ``original_bgr`` for the ArcFace embedding + landmarks, but
    the OUTPUT pixels are diffusion-fresh (ArcFace embedding is semantic; landmark
    image is pure geometry), so SynthID is not transported.

    ``detect_faces_fn`` returns a list of ``(x, y, w, h)`` boxes given a BGR image.
    """
    import cv2
    import numpy as np
    import torch

    if detect_faces_fn is None:
        from pathlib import Path

        from remove_ai_watermarks import auto_config as _ac

        def _default_detect(bgr: NDArray[Any]) -> list[tuple[int, int, int, int]]:
            h_d, w_d = bgr.shape[:2]
            model = Path(_ac.__file__).parent / "assets" / "face_detection_yunet_2023mar.onnx"
            det = cv2.FaceDetectorYN.create(str(model), "", (w_d, h_d), _ac._FACE_SCORE, 0.3, 5000)
            det.setInputSize((w_d, h_d))
            _, faces = det.detect(bgr)
            if faces is None:
                return []
            return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces if int(f[2]) > 0 and int(f[3]) > 0]

        detect_faces_fn = _default_detect

    boxes = detect_faces_fn(cleaned_bgr)
    if not boxes:
        logger.debug("instantid_restore: no faces detected; returning cleaned image unchanged")
        return cleaned_bgr

    pipeline = _get_pipeline()
    face_analyser = _get_face_analyser()

    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipeline.device).manual_seed(seed)

    restored: list[tuple[NDArray[Any], tuple[int, int, int, int]]] = []
    for box in boxes:
        id_crop_bgr, _square_box = _face_crop_square(original_bgr, box)
        if id_crop_bgr.size == 0:
            continue

        # Resize the crop to the InstantID target so InsightFace + the pipeline both
        # work in the same coordinate space.
        crop_resized = cv2.resize(
            id_crop_bgr, (_INSTANTID_FACE_SIZE, _INSTANTID_FACE_SIZE), interpolation=cv2.INTER_LANCZOS4
        )

        # InsightFace expects BGR. It returns embedding + 5 keypoints per detected face.
        # Pick the largest face in the crop (sorted by bbox area).
        face_infos = face_analyser.get(crop_resized)
        if not face_infos:
            logger.debug("instantid_restore: InsightFace did not find a face in the crop; skipping")
            continue
        face_info = sorted(
            face_infos,
            key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
        )[-1]
        face_emb = face_info["embedding"]
        face_kps = face_info["kps"]

        # Render the landmark stick figure at the same size as the generation target.
        landmark_img = _draw_kps((_INSTANTID_FACE_SIZE, _INSTANTID_FACE_SIZE), face_kps)

        out = pipeline(
            prompt=_INSTANTID_PROMPT,
            negative_prompt=_INSTANTID_NEGATIVE,
            image_embeds=face_emb,
            image=landmark_img,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        gen_rgb = out.images[0]
        gen_bgr = cv2.cvtColor(np.array(gen_rgb), cv2.COLOR_RGB2BGR)

        # Multi-face anti-patchwork: each gen_bgr is a fresh 1024x1024 SCENE with a
        # face in it. Compositing the whole 1024 frame into the original face's
        # square_box pulls regenerated BACKGROUND pixels into the cleaned image
        # (different backgrounds per face -> patchwork on group photos). Detect
        # where the face actually landed in gen_bgr, crop tightly to it, and place
        # it at the ORIGINAL face bbox (not the 2x square_box). The composite then
        # only touches face pixels and the background of the cleaned canvas is
        # preserved.
        gen_face_infos = face_analyser.get(gen_bgr)
        if gen_face_infos:
            gf = max(
                gen_face_infos,
                key=lambda x: (x["bbox"][2] - x["bbox"][0]) * (x["bbox"][3] - x["bbox"][1]),
            )
            gx1, gy1, gx2, gy2 = (int(v) for v in gf["bbox"])
            gw, gh = gx2 - gx1, gy2 - gy1
            gcx, gcy = gx1 + gw // 2, gy1 + gh // 2
            # Match the input crop padding (pad=0.5 default of _face_crop_square,
            # which gives a side of ~2x the face size).
            side = int(max(gw, gh) * 2.0)
            half = side // 2
            cx1 = max(0, gcx - half)
            cy1 = max(0, gcy - half)
            cx2 = min(gen_bgr.shape[1], gcx + half)
            cy2 = min(gen_bgr.shape[0], gcy + half)
            face_crop = gen_bgr[cy1:cy2, cx1:cx2]
        else:
            # Fallback: use the whole 1024 frame (matches the pre-2026-06-08 path).
            logger.debug("instantid_restore: no face found in generated image; using full frame")
            face_crop = gen_bgr

        # Composite the tight face crop into a target box centered on the ORIGINAL
        # face bbox, same pad as the input crop so the face fills its natural slot.
        x, y, bw, bh = box
        target_side = int(max(bw, bh) * 2.0)
        thalf = target_side // 2
        tcx, tcy = x + bw // 2, y + bh // 2
        h_c, w_c = cleaned_bgr.shape[:2]
        tx1 = max(0, tcx - thalf)
        ty1 = max(0, tcy - thalf)
        tx2 = min(w_c, tcx + thalf)
        ty2 = min(h_c, tcy + thalf)

        restored.append((face_crop, (tx1, ty1, tx2, ty2)))

    if not restored:
        return cleaned_bgr
    return _composite_faces_elliptical(cleaned_bgr, restored)


def _composite_faces_elliptical(
    base_bgr: NDArray[Any],
    restored_crops: list[tuple[NDArray[Any], tuple[int, int, int, int]]],
    feather_div: int = 8,
) -> NDArray[Any]:
    """Composite face crops into ``base_bgr`` using an elliptical, feathered alpha.

    Unlike ``photomaker_restore._composite_faces`` which feathers a RECTANGULAR
    alpha over the whole crop bbox, this builds an ELLIPSE inscribed in each
    bbox and feathers the ellipse edge. The bbox corners (which carry
    regenerated-scene background pixels) fade to zero so the cleaned-image
    background stays intact, eliminating multi-face patchwork on group photos.
    The ellipse covers roughly the head silhouette which is what we want to
    replace; everything outside it -- hair edges, shoulders, scene context --
    stays from the cleaned canvas.
    """
    import cv2
    import numpy as np

    out = base_bgr.astype(np.float32)
    h_b, w_b = base_bgr.shape[:2]

    for crop, (x1, y1, x2, y2) in restored_crops:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_b, x2), min(h_b, y2)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        resized = cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_LANCZOS4)

        # Elliptical alpha inscribed in the bbox (axes slightly inside so the
        # feather edge tapers cleanly inside the rectangle), feathered with a
        # Gaussian sized by the shorter side.
        alpha_crop = np.zeros((bh, bw), dtype=np.float32)
        center = (bw // 2, bh // 2)
        axes = (max(1, int(bw * 0.45)), max(1, int(bh * 0.55)))
        cv2.ellipse(alpha_crop, center, axes, 0, 0, 360, 1.0, -1)
        k = max(3, (min(bw, bh) // feather_div) | 1)
        alpha_crop = cv2.GaussianBlur(alpha_crop, (k, k), 0)

        alpha_full = np.zeros((h_b, w_b), dtype=np.float32)
        alpha_full[y1:y2, x1:x2] = alpha_crop
        full_restored = np.zeros_like(out)
        full_restored[y1:y2, x1:x2] = resized
        a = alpha_full[:, :, None]
        out = full_restored * a + out * (1.0 - a)

    return np.clip(out, 0, 255).astype(np.uint8)
