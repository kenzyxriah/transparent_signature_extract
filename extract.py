import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from utils import thresh_image, smart_crop #, remove_background

def rembg_extract_signature(image_path, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    thresh = thresh_image(img)

    # Hierachical contour extraction
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        print("No contours found.")
        return None

    # Find largest parent contour (likely the main signature)
    max_area = 0
    main_idx = -1
    for i, (cnt, hier) in enumerate(zip(contours, hierarchy[0])):
        if hier[3] == -1:  # Only top-level contours
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                main_idx = i

    if main_idx == -1:
        print("No valid outer contour found.")
        return None

    # Build mask from main contour and its children
    mask = np.zeros_like(thresh)
    indices_to_draw = [main_idx]  # Start with main
    
    # Add children of main_idx (holes)
    for i, h in enumerate(hierarchy[0]):
        if h[3] == main_idx:
            indices_to_draw.append(i)

    for i in indices_to_draw:
        color = 255 if hierarchy[0][i][3] == -1 else 0  # Fill parent, erase hole
        cv2.drawContours(mask, contours, i, color, -1)

    # Smart crop from original image using bounding box
    bbox = cv2.boundingRect(contours[main_idx])
    cropped_img = smart_crop(img, bbox)

    # Re-threshold and repeat contour extraction inside cropped region
    thresh2 = thresh_image(cropped_img)
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy2 is None:
        print("No contours in cropped image.")
        return None

    # Find largest outer again
    max_idx2 = max(
        [(i, cv2.contourArea(cnt)) for i, cnt in enumerate(contours2) if hierarchy2[0][i][3] == -1],
        key=lambda x: x[1],
        default=(None, 0)
    )[0]

    if max_idx2 is None:
        print("No valid contour found in cropped image.")
        return None

    # Draw main and its holes again
    mask2 = np.zeros_like(thresh2)
    draw_indices = [max_idx2]
    for i, h in enumerate(hierarchy2[0]):
        if h[3] == max_idx2:
            draw_indices.append(i)

    for i in draw_indices:
        color = 255 if hierarchy2[0][i][3] == -1 else 0
        cv2.drawContours(mask2, contours2, i, color, -1)

    # Cleanup
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=1)

    # Transparent RGBA image
    h, w = clean_mask.shape
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[clean_mask == 255] = [0, 0, 0, 255]  # Ink = opaque black
    result[clean_mask == 0] = [0, 0, 0, 0]      # Background = transparent

    if output_path:
        cv2.imwrite(output_path, result)
        print(f"Signature extracted and saved to {output_path}")
    else:
        plt.figure(figsize=(8, 4))
        plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')

        plt.subplot(132), plt.imshow(result), plt.title('Transparent Signature')
        plt.tight_layout()
        plt.show()

    # return result


if __name__ == "__main__":
    rembg_extract_signature(r'C:\Users\Admin\Downloads\test11.jpeg', 'output1.png')
    # rembg_extract_signature(r'C:\Users\Admin\Downloads\test11.jpeg', 'output2.png')
    # rembg_extract_signature(r'C:\Users\Admin\Downloads\test12.jpeg', 'output.png')
    # rembg_extract_signature(r'C:\Users\Admin\Downloads\2.jpg', 'output4.png')
    # rembg_extract_signature(r'C:\Users\Admin\Downloads\ref.png', 'output5.png')
    # rembg_extract_signature(r'C:\Users\Admin\Downloads\test14.jpeg', 'output14.png')
