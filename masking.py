import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def extract_signature(image_path, output_path=None, threshold=127, blur_kernel=(5,5)):

    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
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
    
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask for the signature
    mask = np.zeros_like(thresh)
    
    # Filter contours by area to keep only the signature
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Adjust this threshold as needed
            cv2.drawContours(mask, [contour], -1, 255, -1)
    
    # Extract signature
    signature = cv2.bitwise_and(thresh, mask)
    
    # Create result with transparent background
    result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
    result[signature == 255] = [0, 0, 0, 255]  # Black signature with alpha channel
    
    if output_path:
        # Save the result
        cv2.imwrite(output_path, result)
        print(f"Signature extracted and saved to {output_path}")
    else:
        # Check out the masking accuracy and display results
        plt.figure(figsize=(10, 8))
        plt.subplot(221), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
        # plt.subplot(222), plt.imshow(thresh, cmap='gray'), plt.title('Thresholded')
        plt.subplot(222), plt.imshow(mask, cmap='gray'), plt.title('Mask')
        # plt.subplot(224), plt.imshow(result, cmap='gray'), plt.title('Extracted Signature')
        plt.tight_layout()
        plt.show()
    
    return result


# Example usage
if __name__ == "__main__":
    # Process images and get a mask
    extract_signature(r'C:\Users\Admin\Downloads\2.jpg') # , "extracted_signature.png")
    extract_signature(r'C:\Users\Admin\Downloads\2.png')
    extract_signature(r'C:\Users\Admin\Downloads\1.png')
    extract_signature(r'C:\Users\Admin\Downloads\2.jiff')
    extract_signature(r'C:\Users\Admin\Downloads\test.png')
    extract_signature(r'C:\Users\Admin\Downloads\ref.png')