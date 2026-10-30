from io import BytesIO
import os

import numpy as np
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from prediction_service import DEVICE, predict_signal, predict_image


MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
MIN_IMAGE_DIMENSION = 50
ALLOWED_IMAGE_FORMATS = {"PNG", "JPEG", "WEBP"}

app = FastAPI(
    title="CardioFusion AI API",
    description=(
        "Research prototype for ECG signal and ECG image classification. "
        "Not intended for clinical diagnosis."
    ),
    version="1.1.0",
)

# Keep production CORS restricted to the configured frontend origin.
_frontend_origin = os.getenv("FRONTEND_ORIGIN", "").strip()
_allowed_origins = [_frontend_origin] if _frontend_origin else []

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


class SignalRequest(BaseModel):
    # Exactly 1000 samples; each sample must contain 12 lead values.
    signal: list[list[float]] = Field(
        ...,
        min_length=1000,
        max_length=1000,
        description="ECG samples shaped (1000, 12), in training lead order.",
    )


@app.get("/")
def root():
    return {
        "message": "CardioFusion AI API is ready.",
        "documentation": "/docs",
    }




def looks_like_ecg(image: Image.Image) -> bool:
    """Experimental code-only ECG image screening heuristic."""
    import numpy as np

    gray = image.convert("L")
    gray.thumbnail((1024, 512))
    pixels = np.asarray(gray, dtype=np.float32)

    if pixels.shape[0] < 2 or pixels.shape[1] < 2:
        return False

    # Reject nearly uniform or blank images.
    if float(np.std(pixels)) < 12:
        return False

    # Compare visual detail across eight horizontal sections.
    boundaries = np.linspace(
        0, pixels.shape[0], 9
    ).astype(int)

    details = []

    for i in range(8):
        section = pixels[boundaries[i]:boundaries[i + 1], :]

        if section.shape[0] < 2 or section.shape[1] < 2:
            return False

        horizontal_edges = np.abs(np.diff(section, axis=1))
        vertical_edges = np.abs(np.diff(section, axis=0))

        detail = (
            np.mean(horizontal_edges) +
            np.mean(vertical_edges)
        )

        details.append(float(detail))

    # Experimental threshold; validate on more images before production use.
    return min(details) >= 15


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            "signal_cnn": "loaded",
            "convnext_tiny": "loaded",
        },
    }


@app.post("/predict/signal")
def predict_ecg_signal(request: SignalRequest):
    try:
        signal = np.asarray(request.signal, dtype=np.float32)

        if signal.shape != (1000, 12):
            raise HTTPException(
                status_code=422,
                detail=(
                    "Signal must contain exactly 1000 rows "
                    "and 12 leads per row."
                ),
            )

        if not np.isfinite(signal).all():
            raise HTTPException(
                status_code=422,
                detail="Signal contains non-finite values.",
            )

        # Preserve the existing trained model and preprocessing.
        return predict_signal(signal)

    except HTTPException:
        raise
    except (ValueError, TypeError, OverflowError) as exc:
        raise HTTPException(
            status_code=422,
            detail="Signal must contain valid finite numeric values.",
        ) from exc


@app.post("/predict/image")
async def predict_ecg_image(file: UploadFile = File(...)):
    contents = await file.read(MAX_IMAGE_BYTES + 1)

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    try:
        # Inspect the actual image format; do not trust MIME type alone.
        with Image.open(BytesIO(contents)) as inspected:
            actual_format = inspected.format

            if actual_format not in ALLOWED_IMAGE_FORMATS:
                raise HTTPException(
                    status_code=415,
                    detail="Upload a valid PNG, JPEG, or WEBP image.",
                )

            width, height = inspected.size

            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                raise HTTPException(
                    status_code=422,
                    detail="Image dimensions must be at least 50 × 50 pixels.",
                )

            if width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large to process safely.",
                )

            inspected.verify()

        # Reopen after verify(), then fully decode the image.
        with Image.open(BytesIO(contents)) as decoded:
            decoded.load()
            image = decoded.convert("RGB")

        # Conservative blank-image check. This is not an ECG authenticity test.
        grayscale = np.asarray(image.convert("L"), dtype=np.uint8)
        if float(grayscale.std()) < 1.0:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The image appears almost completely uniform or blank. "
                    "Upload a clearer ECG image."
                ),
            )

        # Preliminary code-based ECG image screening.
        if not looks_like_ecg(image):
            raise HTTPException(
                status_code=422,
                detail=(
                    "This image may not be a valid ECG. "
                    "Please upload a clear ECG image."
                ),
            )

        # Preserve the existing ConvNeXt prediction service.
        return predict_image(image)

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid, readable image.",
        ) from exc
    finally:
        await file.close()
