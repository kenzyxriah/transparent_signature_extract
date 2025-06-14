import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def extract_signature(image_path: str, output_path: str=None, blur_kernel=(5,5)):

    # Read the image as an np.ndarray
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert BGR Image to GrayScale
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
    
    # Create a mask for the image
    mask = np.zeros_like(thresh)
    
    # Filter out and draw only good enough contours by area 
    #normally we'd just do, 
    # cv2.drawContours(mask, contours, -1, 255, -1)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Adjust this threshold as needed
            cv2.drawContours(
                    image=mask,          # Where to draw
                    contours=[contour],  # The contour(s) to draw
                    contourIdx=-1,       # -1 means draw **all** contours in the list
                    color=255,           # Draw color: 255 is white (in grayscale)
                    thickness=-1         # -1 means **fill** the contour
                )
    
    
    if output_path:
        # Save the result
        cv2.imwrite(output_path, mask)
        print(f"Masking done and saved to {output_path}")
    else:
        # Check out the masking accuracy and display results
        plt.figure(figsize=(8, 5))
        plt.subplot(121), plt.imshow(img), plt.title('Original')
        plt.subplot(122), plt.imshow(mask, cmap='gray'), plt.title('Mask')
        plt.tight_layout()
        plt.show()
    
    


# Example usage
if __name__ == "__main__":
    # Process images and get a mask
    # extract_signature(r'C:\Users\Admin\Downloads\2.jpg') # , "extracted_signature.png")
    extract_signature(r'C:\Users\Admin\Downloads\test11.jpeg')