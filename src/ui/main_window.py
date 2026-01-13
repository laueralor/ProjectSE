import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

# Configuración de apariencia
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class MedicalApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("CANCER DETECT AI | Diagnostic Suite")
        self.geometry("900x700")

        # Configuración de cuadrícula (Grid) para que sea responsivo
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- Sidebar (Menú lateral profesional) ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo_label = ctk.CTkLabel(self.sidebar, text="CD AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=30)

        self.btn_upload = ctk.CTkButton(self.sidebar, text="Cargar DICOM", command=self.handle_upload)
        self.btn_upload.grid(row=1, column=0, padx=20, pady=10)
        
        self.appearance_mode_label = ctk.CTkLabel(self.sidebar, text="Tema:", anchor="w")
        self.appearance_mode_label.grid(row=5, column=0, padx=20, pady=(10, 0))
        self.appearance_mode_optionemenu = ctk.CTkOptionMenu(self.sidebar, values=["Light", "Dark"], command=self.change_appearance_mode)
        self.appearance_mode_optionemenu.grid(row=6, column=0, padx=20, pady=(10, 20))

        # --- Área Principal ---
        self.main_frame = ctk.CTkFrame(self, corner_radius=15, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.title_label = ctk.CTkLabel(self.main_frame, text="Panel de Análisis de Diagnóstico", font=ctk.CTkFont(size=22, weight="bold"))
        self.title_label.pack(pady=(10, 20))

        # Visualizador de imagen con bordes redondeados
        self.image_display = ctk.CTkLabel(self.main_frame, text="Esperando escáner...", 
                                         fg_color=("#ebebeb", "#212121"), corner_radius=15,
                                         width=500, height=400)
        self.image_display.pack(expand=True, fill="both", padx=20, pady=20)

        # Barra de estado inferior
        self.status_bar = ctk.CTkLabel(self, text="Sistema conectado a hospital.db", font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=20, pady=5)

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def handle_upload(self):
        file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
        if file_path:
            if os.path.exists("samples/test_result.png"):
                img = Image.open("samples/test_result.png")
                # El tamaño del botón de visualización
                img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(450, 450))
                
                self.image_display.configure(image=img_ctk, text="")
                self.status_bar.configure(text=f"Analizando: {os.path.basename(file_path)}", text_color="#2ecc71")

if __name__ == "__main__":
    app = MedicalApp()
    app.mainloop()