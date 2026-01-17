import os
from src.processing.reader import process_dicom

def process_and_analyze(file_path):
    """
    Recibe la ruta de un archivo, lo procesa y devuelve los resultados.
    """
    try:
        # 1. Procesamos la imagen usando el reader que ya tienes
        # Esto nos devolverá los metadatos y guardará la imagen en samples/
        result = process_dicom(file_path)
        
        # 2. Por ahora, añadimos un resultado de IA "falso" para probar
        result["score"] = "85%" 
        result["status"] = "success"
        
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}