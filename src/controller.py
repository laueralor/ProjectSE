import os
import sqlite3
from src.processing.reader import process_dicom
from src.processing.ai_model import predict_cancer

def process_and_analyze(file_path):
    try:
        # 1. Procesar imagen
        result = process_dicom(file_path)
        
        # 2. Obtener score de IA
        score = predict_cancer(result["image_path"])
        score_str = f"{score}%" if score is not None else "N/A"
        
        # 3. GUARDAR EN HISTORIAL (Base de datos)
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO patients (patient_id, status, ai_score) VALUES (?, ?, ?)", 
            (result["patient_id"], "Completado", score_str)
        )
        conn.commit()
        conn.close()
        
        # 4. Devolver resultados a la UI
        result["score"] = score_str
        result["status"] = "success"
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def get_all_history():
    """Trae todos los registros de pacientes de la base de datos."""
    try:
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        # Traemos ID, Fecha y Resultado de IA
        cursor.execute("SELECT patient_id, timestamp, ai_score FROM patients ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Error al consultar historial: {e}")
        return []