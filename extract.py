import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

from utils import thresh_image, smart_crop

def rembg_extract_signature(image_path, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None

    thresh = thresh_image(img)
    
    #----- First step is to identify the largest external contour(s)-----
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the signature
    mask = np.zeros_like(thresh)

    # Find the largest contour (assumed signature)
    main_contour = max(contours, key = cv2.contourArea)

    if main_contour is None:
        print("No significant contour found.")
        return None

    # Draw the main signature contour
    cv2.drawContours(mask, [main_contour], -1, 255, -1)


    bbox = cv2.boundingRect(main_contour)
    cropped_img = smart_crop(img, bbox)

# 2. ----Second step is to perform hierachical masking-----
# After smart cropping, we find the largest contour once more on the cropped image.
# But we use Retr_comp, where we get a parent(outer) and child(inner) boundaries
# we make sure that anything that isnt the parent boundary in hierachy, is white/transparent

    # Re-threshold and repeat contour extraction inside cropped region
    thresh2 = thresh_image(cropped_img)
    contours2, hierarchy2 = cv2.findContours(thresh2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Find largest/maximum outer/parent layer (which we assume to be the outline of the signature)
    # hierarchy2[0][i][3] == -1 is the parent
    max_idx2 = max(
        [(i, cv2.contourArea(cnt)) for i, cnt in enumerate(contours2) if hierarchy2[0][i][3] == -1],
        key=lambda x: x[1],
        default=(None, 0)
    )[0]

    if max_idx2 is None:
        print("No valid contour found in cropped image.")
        return None

    # Create an empty black mask, and draw only the max contour as black, and others like the child, white
    mask2 = np.zeros_like(thresh2)
    draw_indices = [max_idx2]
    
    # get other contours now not only on the parent level matching our max parent index
    # that should be their children
    for i, h in enumerate(hierarchy2[0]):
        if h[3] == max_idx2:
            draw_indices.append(i)

    # ensure only the case of the max parent is colored black
    # then draw all contours, which will have just the max parent contour colored
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



if __name__ == "__main__":
    rembg_extract_signature(r'C:\Users\Admin\Downloads\2.jpg') #, 'output4.png')

