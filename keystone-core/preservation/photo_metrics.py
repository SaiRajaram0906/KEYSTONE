from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
__all__ = ["sharpness", "exposure_clip", "image_metrics", "update_photo_meta"]

def _read_gray(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"could not read image: {path}")
    return image

def _read_color(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ValueError(f"could not read image: {path}")
    return image

def sharpness(path: str | Path) -> float:
    """Variance of the Laplacian; larger values mean sharper detail."""
    gray = _read_gray(path)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def exposure_clip(path: str | Path) -> float:
    """Fraction of pixels clipped at either 0 or 255."""
    image = _read_color(path)
    clipped = np.logical_or(image <= 0, image >= 255)
    return float(clipped.any(axis=2).mean())

def image_metrics(path: str | Path) -> dict[str, float | int]:
    image = _read_color(path)
    return {"width": int(image.shape[1]), "height": int(image.shape[0]),
            "sharpness": sharpness(path), "exposure_clip": exposure_clip(path)}

def update_photo_meta(photo, path: str | Path):
    metrics = image_metrics(path)
    return type(photo)(photo_id=photo.photo_id, width=metrics["width"], height=metrics["height"],
                       sharpness=float(metrics["sharpness"]), exposure_clip=float(metrics["exposure_clip"]),
                       footprint_mm=list(photo.footprint_mm), markers_visible=list(photo.markers_visible),
                       iso=photo.iso, path=str(path))
