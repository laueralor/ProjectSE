import sqlite3
import os

def init_database():
    """
    Initializes the SQLite database and creates the necessary tables.
    Matches the requirements for Module 1 and 4.
    """
    # This will create a file named 'hospital.db' in the root folder
    db_path = "hospital.db"
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create Patients table based on your design specs
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_external_id TEXT UNIQUE NOT NULL,
            modality TEXT,
            last_scan_date DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'doctor'
        )
    ''')

    conn.commit()
    conn.close()
    print(f"Database initialized successfully at: {os.path.abspath(db_path)}")

def save_patient(patient_id, modality):
    """
    Saves or updates patient information in the database.
    Part of Module 1: Patient Management.
    """
    db_path = "hospital.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # We use INSERT OR IGNORE so that if the patient already exists, it doesn't cause an error
        cursor.execute('''
            INSERT OR IGNORE INTO patients (patient_external_id, modality)
            VALUES (?, ?)
        ''', (patient_id, modality))
        
        conn.commit()
        print(f"Database: Patient {patient_id} processed successfully.")
    except Exception as e:
        print(f"Database Error: {e}")
    finally:
        conn.close()
if __name__ == "__main__":
    init_database()