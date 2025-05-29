# src/preprocessing.py

import cv2
import numpy as np
from PIL import Image

def remove_watermark(img: np.ndarray, threshold: int = 245) -> np.ndarray:
    """
    Remove bright overlays (watermarks) by thresholding + inpainting.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

def suppress_specular(img: np.ndarray, spec_thresh: int = 250) -> np.ndarray:
    """
    Suppress only the brightest specular highlights via HSV threshold + inpainting.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv[:,:,2], spec_thresh, 255)
    return cv2.inpaint(img, mask, inpaintRadius=2, flags=cv2.INPAINT_TELEA)

def denoise_nlmeans(img: np.ndarray) -> np.ndarray:
    """
    Light denoising with NL Means to preserve fine details.
    """
    return cv2.fastNlMeansDenoisingColored(img, None, h=3, hColor=3,
                                           templateWindowSize=7, searchWindowSize=21)

def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Moderate CLAHE on the LAB L-channel for smooth contrast enhancement.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=0.8, tileGridSize=(32,32))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

def preprocess_pil(img_pil: Image.Image) -> Image.Image:
    """
    Full preprocessing pipeline on a PIL image:
      1. Convert to BGR numpy
      2. Remove watermarks
      3. Suppress only strongest specular spots
      4. Denoise lightly
      5. Apply moderate CLAHE
      6. Convert back to PIL RGB
    """
    # PIL → BGR numpy
    img = np.array(img_pil)[:,:,::-1].copy()

    # 1. Watermark removal
    img = remove_watermark(img)

    # 2. Specular highlight suppression
    img = suppress_specular(img)

    # 3. Denoising
    img = denoise_nlmeans(img)

    # 4. Contrast enhancement
    img = apply_clahe(img)

    # BGR → RGB PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
