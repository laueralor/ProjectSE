import pydicom
import os
import matplotlib.pyplot as plt
from src.database.db_manager import save_patient
import numpy as np
import cv2

def resize_image(image, target_size=(256, 256)):
    """
    Resizes the image to the target dimensions for the AI model.
    Professional standard for medical imaging.
    """
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    return resized

def normalize_image(pixel_array):
    """
    Normalizes the image to a 0.0 - 1.0 range.
    Part of Module 3: Preprocessing.
    """
    # 1. Eliminar valores extremos (ruido)
    min_val = np.min(pixel_array)
    max_val = np.max(pixel_array)
    
    # 2. Aplicar la fórmula de normalización: (x - min) / (max - min)
    normalized = (pixel_array - min_val) / (max_val - min_val)
    
    return normalized
def read_medical_data(file_path):
    """
    Reads a DICOM file, extracts metadata and saves the pixel array as a PNG image.
    Following Module 2 and 3 specifications.
    """
    # 1. Integrity check: Verify if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    try:
        # 2. Read the DICOM file using pydicom library
        dataset = pydicom.dcmread(file_path)
        
        # 3. Metadata extraction (Module 2)
        print(f"--- DICOM Metadata Extracted ---")
        print(f"Patient ID: {dataset.PatientID}")
        print(f"Modality: {dataset.Modality}")
        print(f"Manufacturer: {dataset.Manufacturer}")
        print(f"Dimensions: {dataset.Rows} x {dataset.Columns} pixels")

        # 4. Pixel data extraction (Module 3)
        pixel_data = dataset.pixel_array

        normalized_pixels = normalize_image(pixel_data)
        print(f"Brillo normalizado: {normalized_pixels.min()} a {normalized_pixels.max()}")

        plt.imshow(normalized_pixels, cmap='gray')
        plt.title("Normalized Scan (0.0 - 1.0)")
        plt.savefig("samples/test_normalized.png")

        final_image = resize_image(normalized_pixels)
        print(f"Final dimensions for AI: {final_image.shape} pixels")

        print(f"Max Pixel Intensity: {pixel_data.max()}")
        print(f"Min Pixel Intensity: {pixel_data.min()}")
        print("Image loaded successfully into memory.")

        # 5. Image Visualization & Saving (Module 6 Preparation)
        plt.imshow(pixel_data, cmap='gray')
        plt.title(f"Patient Scan: {dataset.PatientID}")
        
        # Saving the result as a physical file in the samples folder
        output_path = "samples/test_result.png"
        plt.savefig(output_path)
        
        print(f"Success! Image saved as: {output_path}")

        save_patient(dataset.PatientID, dataset.Modality)
        
    except Exception as e:
        print(f"Error reading medical file: {e}")

# Main execution
if __name__ == "__main__":
    sample_file = "samples/test_scan.dcm"
    read_medical_data(sample_file)