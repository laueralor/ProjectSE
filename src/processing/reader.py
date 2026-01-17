import pydicom
import numpy as np
from PIL import Image
import os
from src.processing.ai_model import predict_cancer # Importamos tu IA

def process_dicom(file_path):
    """Extrae metadatos y prepara la imagen."""
    ds = pydicom.dcmread(file_path)
    patient_id = ds.PatientID
    
    image_data = ds.pixel_array.astype(float)
    rescaled_image = (np.maximum(image_data, 0) / image_data.max()) * 255
    final_image = Image.fromarray(np.uint8(rescaled_image))
    
    output_path = "samples/test_result.png"
    final_image.save(output_path)

    # LLAMADA A LA IA
    score = predict_cancer(output_path)

    metadata = {
        "patient_id": patient_id,
        "modality": ds.Modality,
        "ai_score": f"{score}%" if score is not None else "Error",
        "image_path": output_path
    }
    
    return metadata