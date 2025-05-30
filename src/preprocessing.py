import cv2
import numpy as np
from PIL import Image


def remove_overlays(img: np.ndarray,
                    bright_thresh: int = 245,
                    sat_thresh: int = 60,
                    val_thresh: int = 200,
                    dilate_iter: int = 3,
                    crop_top: int = 25) -> np.ndarray:
    """
    Remove on‐screen overlays and watermarks by:
      - Brightness threshold (text & watermark)
      - Low‐sat, high‐val HSV threshold (pale UI text)
      - Dilating the mask to cover edges
      - Inpainting
      - Cropping a top strip if needed
    """
    # 1. brightness mask
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, m1 = cv2.threshold(gray, bright_thresh, 255, cv2.THRESH_BINARY)

    # 2. HSV pale-color mask
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m2 = cv2.inRange(hsv,
                     np.array([0, 0, val_thresh]),
                     np.array([179, sat_thresh, 255]))

    # 3. combine & dilate
    mask = cv2.bitwise_or(m1, m2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # 4. inpaint
    img = cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 5. optional top‐crop of UI bar
    if crop_top > 0:
        img[:crop_top, :] = cv2.blur(img[:crop_top, :], (7,7))

    return img

def center_crop_square(img: np.ndarray) -> np.ndarray:
    """
    Crop the largest possible square centered in the image.
    """
    h, w = img.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    return img[top:top+side, left:left+side]

def suppress_specular(img: np.ndarray, spec_thresh: int = 255) -> np.ndarray:
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
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16,16))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def preprocess_pil(img_pil: Image.Image) -> Image.Image:
    # PIL → BGR numpy
    img = np.array(img_pil)[:,:,::-1].copy()

    # 1. Watermark removal
    img = remove_overlays(img,
                          bright_thresh=240,
                          sat_thresh=50,
                          val_thresh=180,
                          dilate_iter=4,
                          crop_top=30)


    # 2. Specular highlight suppression
    #img = suppress_specular(img)

    # 3. Denoising
    #img = denoise_nlmeans(img)

    # 4. Contrast enhancement
    img = apply_clahe(img)

    # 5. Center-crop square
    img = center_crop_square(img)

    # BGR → RGB PIL
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)
