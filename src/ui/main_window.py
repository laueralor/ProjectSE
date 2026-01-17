import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import subprocess
from src.controller import process_and_analyze, get_all_history, search_patient_history
from customtkinter import CTkImage, CTkInputDialog

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

        self.btn_export = ctk.CTkButton(self.sidebar, text="Exportar Informe", 
                                        command=self.export_report)
        self.btn_export.grid(row=3, column=0, padx=20, pady=10)
        
        self.btn_logout = ctk.CTkButton(self.sidebar, text="Cerrar Sesión", 
                                        fg_color="#e74c3c", hover_color="#c0392b", 
                                        command=self.handle_logout)
        self.btn_logout.grid(row=7, column=0, padx=20, pady=20)

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
        initial_dir = os.path.join(os.getcwd(), "samples")
        
        # 1. Abrir el buscador restringido a archivos .dcm y a la carpeta de muestras
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Seleccionar Escáner DICOM",
            filetypes=[("Archivos Médicos DICOM", "*.dcm")]
        )
        
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
        # 1. Preguntar si quiere ver todo o buscar uno
        dialog = CTkInputDialog(text="Introduce ID de paciente para buscar (o deja vacío para ver todos):", title="Buscar en Historial")
        search_id = dialog.get_input()
        
        # 2. Obtener los datos según la elección
        if search_id: # Si escribió algo, buscamos ese ID
            data = search_patient_history(search_id)
            title = f"Resultados para Paciente: {search_id}"
        else: # Si no escribió nada, traemos todo
            data = get_all_history()
            title = "Historial Completo"

        if not data:
            messagebox.showinfo("Historial", "No se encontraron registros.")
            return
        
        # 3. Formatear y mostrar
        history_text = "ID Paciente | Fecha | Resultado IA\n"
        history_text += "-"*40 + "\n"
        for row in data:
            history_text += f"{row[0]} | {row[1]} | {row[2]}\n"
        
        messagebox.showinfo(title, history_text)
    
    def handle_logout(self):
        if messagebox.askyesno("Cerrar Sesión", "¿Estás seguro de que quieres salir?"):
            self.destroy() # Cerramos la ventana actual
            # Aquí, dependiendo de cómo lances la app, podrías volver a llamar a Login
            # Por ahora, es la forma segura de finalizar la sesión del médico.

    def export_report(self):
        status_text = self.status_bar.cget("text")
        if "IA Score" not in status_text:
            messagebox.showwarning("Aviso", "No hay ningún análisis activo.")
            return
            
        file_name = filedialog.asksaveasfilename(defaultextension=".txt",
                                               filetypes=[("Archivo de texto", "*.txt")])
        if file_name:
            try:
                # Escribimos el archivo
                with open(file_name, "w") as f:
                    f.write(f"INFORME MÉDICO - CANCER DETECT AI\n{'='*30}\n")
                    f.write(f"Detalles: {status_text}\n")
                
                messagebox.showinfo("Éxito", "Informe guardado.")

                # INTENTO DE APERTURA AUTOMÁTICA
                # Si estás en Windows (nativo)
                if os.name == 'nt':
                    os.startfile(file_name)
                # Si estás en WSL/Linux/Mac
                else:
                    try:
                        # Intento 1: Comando estándar de Linux
                        subprocess.run(['xdg-open', file_name], check=True)
                    except:
                        # Intento 2: Comando específico para WSL (abre el bloc de notas de Windows)
                        subprocess.run(['notepad.exe', file_name], check=False)

            except Exception as e:
                print(f"Nota: El informe se guardó pero no se pudo abrir automáticamente: {e}")
if __name__ == "__main__":
    app = MedicalApp()
    app.mainloop()