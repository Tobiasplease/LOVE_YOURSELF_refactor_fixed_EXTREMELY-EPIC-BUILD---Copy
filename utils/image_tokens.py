"""Image size ↔ vision tokens (Sep 12 2026).

The token floor moves from the server to the client. `--image-min-tokens 1024`
(Aug 17: "Qwen-VL needs >=1024 image tokens for grounding; our frames encoded
at ~600") forced EVERY image to at least 1024 tokens, whatever its purpose. Now
each call sizes its own picture: the drawing review, the adjudicator, the
paper check and the close looks keep the 1024 they had (upscaled here instead
of on the server); the caption picture and the clip follow room attention
(captioner/attention.py) between ATTENTION_IMAGE_TOKENS_MAX and _MIN.

For this encoder (mmproj: patch 16, spatial merge 2) one token is a 32×32
pixel cell: tokens ≈ (w/32)·(h/32). Sizes are rounded to multiples of 32.
"""

from __future__ import annotations

import math
import os
from typing import Optional, Union

TOKEN_PX = 32


def tokens_of(w: int, h: int) -> int:
    return max(1, (int(w) // TOKEN_PX) * (int(h) // TOKEN_PX))


def dims_for_tokens(w: int, h: int, tokens: int) -> tuple:
    """The (w, h) — multiples of 32, aspect kept — that encode to about `tokens`."""
    s = math.sqrt(max(1, tokens) * TOKEN_PX * TOKEN_PX / float(max(1, w * h)))
    nw = max(TOKEN_PX, int(round(w * s / TOKEN_PX)) * TOKEN_PX)
    nh = max(TOKEN_PX, int(round(h * s / TOKEN_PX)) * TOKEN_PX)
    return nw, nh


def tokens_for_attention(attention: float) -> int:
    from config import config as _c

    lo = int(getattr(_c, "ATTENTION_IMAGE_TOKENS_MIN", 256))
    hi = int(getattr(_c, "ATTENTION_IMAGE_TOKENS_MAX", 1024))
    a = min(1.0, max(0.0, float(attention)))
    return int(round(lo + (hi - lo) * a))


def sized(img, tokens: int):
    """Resize a BGR array to encode at about `tokens` (up or down)."""
    import cv2

    h, w = img.shape[:2]
    nw, nh = dims_for_tokens(w, h, tokens)
    if (nw, nh) == (w, h):
        return img
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if nw < w else cv2.INTER_CUBIC)


def sized_jpeg(image: Union[bytes, bytearray, "np.ndarray", None], tokens: int, quality: int = 90) -> Optional[bytes]:
    """JPEG bytes (or a BGR array) → JPEG bytes at about `tokens`."""
    if image is None:
        return None
    import cv2
    import numpy as np

    if isinstance(image, (bytes, bytearray)):
        arr = cv2.imdecode(np.frombuffer(bytes(image), dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            return bytes(image)
    else:
        arr = image
    ok, buf = cv2.imencode(".jpg", sized(arr, tokens), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else (bytes(image) if isinstance(image, (bytes, bytearray)) else None)


def sized_copy(path: Optional[str], tokens: int) -> Optional[str]:
    """A copy of the image file at about `tokens`, written beside it as
    <stem>_t<tokens>.jpg (reused if present) so the llm log keeps a real path
    and the read can tell the size from the name. Falls back to the original."""
    if not path or not os.path.exists(path):
        return path
    try:
        import cv2

        stem, _ = os.path.splitext(path)
        out = f"{stem}_t{int(tokens)}.jpg"
        if os.path.exists(out):
            return out
        img = cv2.imread(path)
        if img is None:
            return path
        if not cv2.imwrite(out, sized(img, tokens), [cv2.IMWRITE_JPEG_QUALITY, 90]):
            return path
        return out
    except Exception:
        return path


def sized_any(image, tokens: int):
    """Path → sized copy path; bytes/array → sized JPEG bytes; None → None."""
    if image is None:
        return None
    if isinstance(image, str):
        return sized_copy(image, tokens)
    return sized_jpeg(image, tokens)
