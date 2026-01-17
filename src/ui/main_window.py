import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import subprocess
from src.controller import process_and_analyze, get_all_history, search_patient_history
from customtkinter import CTkImage, CTkInputDialog
import sqlite3

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
        # 1. Pedimos el ID para saber qué informe abrir
        dialog = CTkInputDialog(text="Introduce ID de paciente para VER su informe\n(O deja vacío para ver la lista completa):", title="Gestor de Informes")
        search_id = dialog.get_input()
        
        # 2. Obtenemos los datos (usando la función de búsqueda que ya tenemos)
        if search_id:
            data = search_patient_history(search_id)
        else:
            data = get_all_history()

        if not data:
            messagebox.showinfo("Historial", "No se encontraron registros.")
            return

        # 3. SI EL MÉDICO BUSCÓ UN ID: Intentamos abrir el informe automáticamente
        if search_id:
            # Comprobamos si el primer resultado tiene una ruta de informe válida
            if len(data[0]) > 3 and data[0][3]:
                report_path = data[0][3]
                try:
                    # Usamos wslpath para que Windows reconozca la ruta de Linux/WSL
                    win_path = subprocess.check_output(['wslpath', '-w', report_path]).decode().strip()
                    # Abrimos el Bloc de notas (es la forma de "verlo" desde la app)
                    subprocess.run(['notepad.exe', win_path], check=False)
                except Exception as e:
                    messagebox.showerror("Error", f"No se pudo abrir el archivo físico: {e}")
            else:
                messagebox.showwarning("Aviso", f"El paciente {search_id} no tiene un informe generado todavía.")

        # 4. En cualquier caso, mostramos la lista resumen para confirmar datos
        history_text = "ID PACIENTE | FECHA | SCORE IA\n" + "-"*40 + "\n"
        for row in data:
            has_report = " [OK]" if len(row) > 3 and row[3] else " [X]"
            history_text += f"{row[0]} | {row[1]} | {row[2]} {has_report}\n"
        
        messagebox.showinfo("Visor de Historial", history_text)
    
    def handle_logout(self):
        if messagebox.askyesno("Cerrar Sesión", "¿Estás seguro de que quieres salir?"):
            self.destroy() # Cerramos la ventana actual
            # Aquí, dependiendo de cómo lances la app, podrías volver a llamar a Login
            # Por ahora, es la forma segura de finalizar la sesión del médico.
    def export_report(self):
        status_text = self.status_bar.cget("text")
        if "IA Score" not in status_text:
            messagebox.showwarning("Aviso", "No hay análisis activo para exportar.")
            return

        try:
            # 1. Extraer ID del paciente
            p_id = status_text.split("|")[0].split(":")[1].strip()
            
            # 2. Crear carpeta de reportes
            reports_dir = os.path.join(os.getcwd(), "reports")
            if not os.path.exists(reports_dir):
                os.makedirs(reports_dir)

            # 3. Generar nombre automático
            file_name = os.path.join(reports_dir, f"Reporte_{p_id}.txt")

            # 4. Escribir el informe
            with open(file_name, "w") as f:
                f.write(f"INFORME MÉDICO - CANCER DETECT AI\n{'='*40}\n")
                f.write(f"ID PACIENTE: {p_id}\n")
                f.write(f"DETALLES: {status_text}\n")

            # 5. Vincular en Base de Datos
            conn = sqlite3.connect('hospital.db')
            cursor = conn.cursor()
            cursor.execute("UPDATE patients SET report_path = ? WHERE patient_id = ? AND timestamp = (SELECT MAX(timestamp) FROM patients WHERE patient_id = ?)", 
                         (file_name, p_id, p_id))
            conn.commit()
            conn.close()

            messagebox.showinfo("Éxito", "Informe guardado automáticamente.")
            # ------------------------------------------

        except Exception as e:
            messagebox.showerror("Error", f"Fallo al abrir el informe: {e}")
if __name__ == "__main__":
    app = MedicalApp()
    app.mainloop()