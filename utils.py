import cv2
from rembg.bg import remove
from PIL import Image
import numpy as np
from io import BytesIO


def thresh_image(image, blur_kernel=(5,5)):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11, 
        2
    )
    
    return thresh

def smart_crop(
    image,
    bbox,
    threshold=0.85,
    min_aspect=2.0,  # favor wider boxes (was 1.5)
    padding_x=65,    # horizontal padding
    padding_y=10     # vertical padding
):
    H, W = image.shape[:2]
    x, y, w, h = bbox

    height_ratio = h / H
    width_ratio = w / W
    aspect = w / h

    # Flag for whether to pad
    apply_padding = True

    # If crop height is too close to image height, reduce height
    if height_ratio > threshold:
        print("Warning: Crop height is too close to original. Reducing height.")
        h = int(H * 0.35)
        y = max(0, y + int((H - h) / 2))
        apply_padding = False

    # Improve aspect ratio: enforce width dominance
    if aspect < min_aspect:
        new_w = int(h * min_aspect)
        x = max(0, x - (new_w - w) // 2)
        w = new_w
        if x + w > W:
            w = W - x

    # Apply asymmetric padding (more horizontal, less vertical)
    if apply_padding:
        x = max(0, x - padding_x)
        w = min(w + 2 * padding_x, W - x)

        y = max(0, y - padding_y)
        h = min(h + 2 * padding_y, H - y)

    return image[y:y+h, x:x+w]

def remove_background(img_array: np.ndarray, output_path = 'output.png', color=(0, 0, 0), path = False):

    """
    Removes background, makes it transparent, and colors the foreground (e.g. signature) black.
    
    Parameters:
    - img_array: Input image as a NumPy array (OpenCV format BGR).
    - output_path: File path to save the processed image.
    - color: RGB tuple for the signature color. Default is black.
    
    """

    # Convert OpenCV BGR to RGB
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_rgb = img_array
    pil_image = Image.fromarray(img_rgb)

    # Convert to bytes
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    input_bytes = buffer.getvalue()

    # Step 1: Remove background
    output_image = remove(input_bytes)

    # Step 2: Convert to RGBA PIL Image
    img = Image.open(BytesIO(output_image)).convert("RGBA")

    # Step 3: Convert to numpy array
    data = np.array(img)

    # Step 4: Modify signature color
    r, g, b, a = data.T
    dark_areas = (r < 100) & (g < 100) & (b < 100) & (a > 0)
    data[..., :-1][dark_areas.T] = color

    # Step 5: Convert back to image and save
    result = Image.fromarray(data)
    if path:
      result.save(output_path, format="PNG")
      print(f"Processed image saved to: {output_path}")
    else:
      return result 
