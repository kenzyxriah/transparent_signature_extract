import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def extract_signature(image_path, output_path=None, blur_kernel=(5,5)):

    # Read the image as 
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
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
    
    # # thresholds for RBG values (when masking our actual image)
    # lower, higher = (np.array([50,50,50]), 
    #                  np.array([250,250,250]))
    
    # thresholds for RBG values
    lower, higher = 50,255
    
    # Create a mask for the image
    mask = cv2.inRange(thresh, lower, higher)
    
    # Find contours
    contours, _ = cv2.findContours(mask, 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_NONE) # could be approx_simple too
    

    
    # Filter out and draw only good enough contours 
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Adjust this threshold as needed
            cv2.drawContours(
                        image=mask,          
                        contours=[contour],  
                        contourIdx=-1,       
                        color=255,           # Draw color: 255 is white (in grayscale)
                        thickness=-1         
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