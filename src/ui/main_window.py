import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from src.controller import process_and_analyze
from tkinter import filedialog, messagebox
import os
from PIL import Image
from customtkinter import CTkImage
from src.controller import get_all_history

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
        # --- Sidebar (Menú lateral profesional) ---
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(4, weight=1) # Espaciador flexible

        self.logo_label = ctk.CTkLabel(self.sidebar, text="CD AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=30)

        self.btn_upload = ctk.CTkButton(self.sidebar, text="Cargar DICOM", command=self.handle_upload)
        self.btn_upload.grid(row=1, column=0, padx=20, pady=10)

        self.history_button = ctk.CTkButton(self.sidebar, text="Ver Historial", command=self.show_history)
        self.history_button.grid(row=2, column=0, padx=20, pady=10)
        
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
        self.image_label = ctk.CTkLabel(self.main_frame, text="Esperando escáner...", 
                                         fg_color=("#ebebeb", "#212121"), corner_radius=15,
                                         width=500, height=400)
        self.image_label.pack(expand=True, fill="both", padx=20, pady=20)

        # Barra de estado inferior
        self.status_bar = ctk.CTkLabel(self, text="Sistema conectado a hospital.db", font=ctk.CTkFont(size=12))
        self.status_bar.grid(row=1, column=0, columnspan=2, padx=20, pady=5)

    def change_appearance_mode(self, new_appearance_mode: str):
        ctk.set_appearance_mode(new_appearance_mode)

    def handle_upload(self):
        # 1. Abrir el buscador de archivos
        file_path = filedialog.askopenfilename(filetypes=[("DICOM files", "*.dcm")])
        
        if file_path:
            # 2. Llamar al controlador que acabamos de probar
            result = process_and_analyze(file_path)
            
            if result["status"] == "success":
                # 3. Mostrar el ID del paciente en la barra de estado
                self.status_bar.configure(
                    text=f"Paciente: {result['patient_id']} | IA Score: {result['score']}",
                    text_color="#2ecc71"
                )
                messagebox.showinfo("Éxito", "Imagen procesada correctamente")
                self.display_image(result["image_path"])
            else:
                messagebox.showerror("Error", f"Fallo al procesar: {result['message']}")
    def display_image(self, image_path):
        # 1. Cargamos la imagen con PIL
        img = Image.open(image_path)
        
        # 2. La convertimos al formato de CustomTkinter
        # El tamaño (400, 400) es para que encaje bien en tu panel
        ctk_img = CTkImage(light_image=img, dark_image=img, size=(400, 400))
        
        # 3. La ponemos en el label que tienes en el centro
        # Nota: Asegúrate de que tu label de imagen se llame self.image_label
        self.image_label.configure(image=ctk_img, text="") 
        self.image_label.image = ctk_img # Guardamos referencia para que no desaparezca

    def show_history(self):
        data = get_all_history()
        if not data:
            messagebox.showinfo("Historial", "No hay registros todavía.")
            return
        
        # Creamos un resumen de texto con los datos
        history_text = "ID Paciente | Fecha | Resultado IA\n"
        history_text += "-"*40 + "\n"
        for row in data:
            history_text += f"{row[0]} | {row[1]} | {row[2]}\n"
        
        # Lo mostramos en una ventana
        messagebox.showinfo("Historial de Diagnósticos", history_text)
if __name__ == "__main__":
    app = MedicalApp()
    app.mainloop()