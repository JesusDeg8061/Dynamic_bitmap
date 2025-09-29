import os
from keras.models import load_model

# -----------------------------
# Configura la ruta de tu modelo
# -----------------------------
ruta_modelo = r"C:\Users\jesus\OneDrive\Desktop\DynamicBitMap\Dynamic_bitmap\ejemplos\models_tf\modelo.keras"

# -----------------------------
# Verificar si el archivo existe
# -----------------------------
if not os.path.exists(ruta_modelo):
    print(f" Archivo NO encontrado en la ruta:\n{ruta_modelo}")
    exit()  # Termina el script si no encuentra el archivo
else:
    print(" Archivo encontrado, intentando cargar el modelo...")

# -----------------------------
# Cargar el modelo y manejar errores
# -----------------------------
try:
    model = load_model(ruta_modelo)
    print("Modelo cargado correctamente.\n")
    print("----- Resumen del modelo -----")
    model.summary()  # Mostrar arquitectura del modelo
except OSError as e:
    print(f" Error al cargar el modelo: {e}")
except ValueError as e:
    print(f" El archivo no parece ser un modelo válido de Keras: {e}")
except Exception as e:
    print(f" Error inesperado: {e}")
