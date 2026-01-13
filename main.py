import customtkinter as ctk
from src.ui.login_window import LoginWindow
from src.ui.main_window import MedicalApp

def launch_dashboard():
    """Starts the main application after successful login."""
    app = MedicalApp()
    app.mainloop()

def start_system():
    """Initializes the security layer first."""
    # We pass the launch_dashboard function as the 'on_login_success' callback
    login_screen = LoginWindow(on_login_success=launch_dashboard)
    login_screen.mainloop()

if __name__ == "__main__":
    start_system()