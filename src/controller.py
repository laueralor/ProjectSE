import os
from src.processing.reader import process_dicom
from src.processing.ai_model import predict_cancer # Nueva importación

def process_and_analyze(file_path):
    try:
        # 1. Procesar DICOM y guardar imagen temporal
        result = process_dicom(file_path)
        
        # 2. Llamar a la IA real con la imagen generada
        score = predict_cancer(result["image_path"])
        
        if score is not None:
            result["score"] = f"{score}%"
            result["status"] = "success"
        else:
            result["status"] = "error"
            result["message"] = "Error en el análisis de IA"
            
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}