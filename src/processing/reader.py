import pydicom
import numpy as np
from PIL import Image
import os
import sqlite3

def process_dicom(file_path):
    """Extrae metadatos y prepara la imagen para la IA."""
    ds = pydicom.dcmread(file_path)
    
    # Extraer metadatos básicos
    metadata = {
        "patient_id": ds.PatientID,
        "modality": ds.Modality,
        "manufacturer": ds.Manufacturer,
        "pixel_data": ds.pixel_array
    }
    
    # Normalizar y guardar imagen de muestra
    image_data = ds.pixel_array.astype(float)
    rescaled_image = (np.maximum(image_data, 0) / image_data.max()) * 255
    final_image = Image.fromarray(np.uint8(rescaled_image)).resize((256, 256))
    
    output_path = "samples/test_result.png"
    final_image.save(output_path)
    
    # Guardar en base de datos
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO patients (patient_id, status) VALUES (?, ?)", 
                   (metadata["patient_id"], "Processed"))
    conn.commit()
    conn.close()
    
    metadata["image_path"] = output_path
    return metadata