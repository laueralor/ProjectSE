import pydicom
import os

def leer_datos_medicos(ruta_archivo):
    # 1. Comprobar si el archivo existe (Requisito de integridad)
    if not os.path.exists(ruta_archivo):
        print(f"Error: No se encuentra el archivo en {ruta_archivo}")
        return

    # 2. Leer el archivo DICOM usando la librería pydicom
    try:
        ds = pydicom.dcmread(ruta_archivo)
        
        # 3. Extraer metadatos definidos en vuestro diseño (Módulo 2)
        print(f"--- Datos extraídos del archivo DICOM ---")
        print(f"ID del Paciente: {ds.PatientID}")
        print(f"Modalidad: {ds.Modality}")
        print(f"Fabricante del equipo: {ds.Manufacturer}")
        print(f"Dimensiones: {ds.Rows} x {ds.Columns} píxeles")
        
    except Exception as e:
        print(f"Error al leer el archivo médico: {e}")

# Ejecutar la prueba apuntando a vuestra carpeta samples
# Nota: La ruta depende de desde dónde lances el script
path_al_archivo = "samples/test_scan.dcm"
leer_datos_medicos(path_al_archivo)