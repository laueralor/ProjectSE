import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import subprocess
from src.controller import process_and_analyze, get_all_history, search_patient_history
from customtkinter import CTkImage, CTkInputDialog
import sqlite3

# Appearance configuration
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MedicalApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CANCER DETECT AI | Diagnostic Suite")
        self.geometry("900x700")

        # Grid configuration to make it responsive
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Professional side menu) ---
        # --- Sidebar (Professional side menu) ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1) # Flexible spacer

        self.logo_label = ctk.CTkLabel(self.sidebar, text="CD AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=30)

        self.btn_upload = ctk.CTkButton(self.sidebar, text="Loading DICOM", command=self.handle_upload)
        self.btn_upload.grid(row=1, column=0, padx=20, pady=10)

        self.history_button = ctk.CTkButton(self.sidebar, text="Show Historial", command=self.show_history)
        self.history_button.grid(row=2, column=0, padx=20, pady=10)

        self.btn_export = ctk.CTkButton(self.sidebar, text="Export Report", 
                                        command=self.export_report)
        self.btn_export.grid(row=3, column=0, padx=20, pady=10)
        
        self.btn_logout = ctk.CTkButton(self.sidebar, text="Logout", 
                                        fg_color="#e74c3c", hover_color="#c0392b", 
                                        command=self.handle_logout)
        self.btn_logout.grid(row=7, column=0, padx=20, pady=20)

        # --- Main Area ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.main_frame, text="Diagnostic Analysis Panel", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(pady=(10, 20))

        # Image viewer with rounded corners
        self.image_label = ctk.CTkLabel(self.main_frame, text="Waiting for scanner...", 
                                         fg_color=("#ebebeb", "#212121"), corner_radius=15,
                                         width=500, height=400)
        self.image_label.pack(expand=True, fill="both", padx=20, pady=20)

        # Bottom status bar
        self.status_bar = ctk.CTkLabel(self, text="System connected to hospital.db", font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=20, pady=5)

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def handle_upload(self):
        initial_dir = os.path.join(os.getcwd(), "samples")
        
        # 1. Open the file browser restricted to .dcm files and the samples folder
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Seleccionar Escáner DICOM",
            filetypes=[("Archivos Médicos DICOM", "*.dcm")]
        )
        
        if file_path:
            # 2. Call the controller we just tested
            result = process_and_analyze(file_path)
            
            if result["status"] == "success":
                # 3. Display the patient ID in the status bar
                self.status_bar.configure(
                    text=f"Patient: {result['patient_id']} | AI Score: {result['score']}",
                    text_color="#2ecc71"
                )
                messagebox.showinfo("Success", "Image processed successfully")
                self.display_image(result["image_path"])
            else:
                messagebox.showerror("Error", f"Failed to process: {result['message']}")
    def display_image(self, image_path):
        # 1. We load the image with PIL
        img = Image.open(image_path)
        
        # 2. We convert it to the CustomTkinter format
        # The size (400, 400) is so that it fits well in your panel
        ctk_img = CTkImage(light_image=img, dark_image=img, size=(400, 400))
        
        # 3. We place it in the label you have in the center
        self.image_label.configure(image=ctk_img, text="") 
        self.image_label.image = ctk_img # We keep a reference so it doesn't get garbage collected

    def show_history(self):
        # 1. We ask for the ID to know which report to open
        dialog = CTkInputDialog(text="Enter patient ID to VIEW their report\n(Or leave empty to see the full list):", title="Report Manager")
        search_id = dialog.get_input()
        
        # 2. We retrieve the data (using the search function we already have)
        if search_id:
            data = search_patient_history(search_id)
        else:
            data = get_all_history()

        if not data:
            messagebox.showinfo("History", "No records found.")
            return

        # 3. IF THE DOCTOR SEARCHED FOR AN ID: We try to open the report automatically
        if search_id:
            # Check if the first result has a valid report path
            if len(data[0]) > 3 and data[0][3]:
                report_path = data[0][3]
                try:
                    # We use wslpath so Windows recognizes the Linux/WSL path
                    win_path = subprocess.check_output(['wslpath', '-w', report_path]).decode().strip()
                    # We open Notepad (it's the way to "view" it from the app)
                    subprocess.run(['notepad.exe', win_path], check=False)
                except Exception as e:
                    messagebox.showerror("Error", f"Could not open the physical file: {e}")
            else:
                messagebox.showwarning("Warning", f"Patient {search_id} does not have a generated report yet.")

        # 4. In any case, we show the summary list to confirm data
        history_text = "ID PACIENTE | FECHA | SCORE IA\n" + "-"*40 + "\n"
        for row in data:
            has_report = " [OK]" if len(row) > 3 and row[3] else " [X]"
            history_text += f"{row[0]} | {row[1]} | {row[2]} {has_report}\n"
        
        messagebox.showinfo("History Viewer", history_text)
    
    def handle_logout(self):
        if messagebox.askyesno("Logout", "Are you sure you want to logout?"):
            self.destroy() #We close the current window
    def export_report(self):
        status_text = self.status_bar.cget("text")
        if "IA Score" not in status_text:
            messagebox.showwarning("Warning", "There is no active analysis to export.")
            return

        try:
            # 1. Extract patient ID
            p_id = status_text.split("|")[0].split(":")[1].strip()
            
            # 2. Create reports folder
            reports_dir = os.path.join(os.getcwd(), "reports")
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)

            # 3. Generate automatic name
            file_name = os.path.join(reports_dir, f"Report_{p_id}.txt")

            # 4. Write the report
            with open(file_name, "w") as f:
                f.write(f"MEDICAL REPORT - CANCER DETECT AI\n{'='*40}\n")
                f.write(f"PATIENT ID: {p_id}\n")
                f.write(f"DETAILS: {status_text}\n")

            # 5. Link in Database
            conn = sqlite3.connect('hospital.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE patients SET report_path = ? WHERE patient_id = ? AND timestamp = (SELECT MAX(timestamp) FROM patients WHERE patient_id = ?)", 
                         (file_name, p_id, p_id))
            conn.commit()
            conn.close()

            messagebox.showinfo("Success", "Report saved automatically.")
            # ------------------------------------------

        except Exception as e:
            messagebox.showerror("Error", f"Failed to open report: {e}")
if __name__ == "__main__":
    app = MedicalApp()
    app.mainloop()