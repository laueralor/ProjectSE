import sqlite3

def register_user(username, password):
    """
    Registers a new doctor in the system.
    Part of Module 1: User Management.
    """
    db_path = "hospital.db"
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        
        conn.commit()
        print(f"User {username} registered successfully.")
        conn.close()
        return True
    except sqlite3.IntegrityError:
        print("Error: Username already exists.")
        return False
    except Exception as e:
        print(f"Auth Error: {e}")
        return False
    
def login_user(username, password):
    """
    Checks if the doctor exists and the password matches.
    """
    db_path = "hospital.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Buscamos al usuario en la tabla 'users'
    cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
    user = cursor.fetchone()
    
    conn.close()

    if user:
        print(f"Login Success: Welcome, Dr. {username}!")
        return True
    else:
        print("Login Failed: Invalid username or password.")
        return False

if __name__ == "__main__":
    # Test registration
    print("--- Registering a new doctor ---")
    register_user("laura_doctor", "password123")

    print("--- Testing Login System ---")
    login_user("laura_doctor", "secure_pass_2026")