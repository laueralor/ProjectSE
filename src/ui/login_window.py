import customtkinter as ctk
from tkinter import messagebox
from src.auth.auth_manager import login_user

# Global Appearance Settings
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class LoginWindow(ctk.CTk):
    def __init__(self, on_login_success):
        super().__init__()
        
        # We pass a function to call when login is successful
        self.on_login_success = on_login_success

        self.title("CD AI | Secure Access")
        self.geometry("400x550")
        self.resizable(False, False)

        # Main container
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(expand=True, fill="both", padx=40)

        # Header Section
        self.icon_label = ctk.CTkLabel(self.main_frame, text="🔒", font=ctk.CTkFont(size=50))
        self.icon_label.pack(pady=(40, 10))

        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="Medical Portal", 
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.title_label.pack(pady=(0, 30))

        # Input Fields
        self.username_input = ctk.CTkEntry(
            self.main_frame, 
            placeholder_text="Username", 
            width=280, 
            height=45
        )
        self.username_input.pack(pady=10)

        self.password_input = ctk.CTkEntry(
            self.main_frame, 
            placeholder_text="Password", 
            show="*", 
            width=280, 
            height=45
        )
        self.password_input.pack(pady=10)

        # Action Button
        self.login_button = ctk.CTkButton(
            self.main_frame, 
            text="Sign In", 
            command=self.attempt_login, 
            width=280, 
            height=45,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        self.login_button.pack(pady=40)

        # Footer
        self.footer_label = ctk.CTkLabel(
            self, 
            text="Authorized Personnel Only", 
            font=ctk.CTkFont(size=10),
            text_color="gray"
        )
        self.footer_label.pack(side="bottom", pady=20)

    def attempt_login(self):
        user = self.username_input.get()
        pw = self.password_input.get()

        # Database Verification (Module 1 Connection)
        if login_user(user, pw):
            messagebox.showinfo("Access Granted", f"Welcome, Dr. {user}")
            self.destroy() # Close login
            self.on_login_success() # Open Main Dashboard
        else:
            messagebox.showerror("Access Denied", "Invalid credentials. Please try again.")

if __name__ == "__main__":
    # Test stub (just for testing this window independently)
    def dummy_success(): print("Transitioning to Dashboard...")
    app = LoginWindow(dummy_success)
    app.mainloop()