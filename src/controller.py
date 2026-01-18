import os
import sqlite3
from src.processing.reader import process_dicom
from src.processing.ai_model import predict_cancer

def process_and_analyze(file_path):
    try:
        # 1. Process image
        result = process_dicom(file_path)
        
        # 2. Get AI score
        score = predict_cancer(result["image_path"])
        score_str = f"{score}%" if score is not None else "N/A"
        
        # 3. SAVE TO HISTORY (Database)
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO patients (patient_id, status, ai_score) VALUES (?, ?, ?)", 
            (result["patient_id"], "Completed", score_str)
        )
        conn.commit()
        conn.close()
        
        # 4. Return results to the UI
        result["score"] = score_str
        result["status"] = "success"
        return result
        
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
def get_all_history():
    try:
        conn = sqlite3.connect('hospital.db')
        cursor = conn.cursor()
        # IMPORTANT: Explicitly requesting the 4 columns
        cursor.execute("SELECT patient_id, timestamp, ai_score, report_path FROM patients ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        conn.close()
        return rows
    except Exception as e:
        print(f"Database query error: {e}")
        return []
    
def search_patient_history(patient_id):
    conn = sqlite3.connect('hospital.db')
    cursor = conn.cursor()
    # Make sure to request the 4 columns (ID, Date, Score, PATH)
    cursor.execute("SELECT patient_id, timestamp, ai_score, report_path FROM patients WHERE patient_id = ? ORDER BY timestamp DESC", (patient_id,))
    rows = cursor.fetchall()
    conn.close()
    return rows