#!/usr/bin/env python
# pip install sounddevice soundfile numpy keyboard
import argparse
import sounddevice as sd
import soundfile as sf
import numpy as np
import keyboard  # Instala con: pip install keyboard
import os

# --- Configuración ---
DEFAULT_FS = 16000
CHANNELS = 2
DEFAULT_OUTPUT_FILE = 'grabacion_larga.ogg' # OGG ocupa muy poco y es nativo
DEFAULT_DEVICE_NAME = "CABLE Output" # Asegúrate que se llama así en tu sistema

parser = argparse.ArgumentParser(description="Graba audio desde un dispositivo de entrada.")
parser.add_argument(
    "--output",
    default=DEFAULT_OUTPUT_FILE,
    help=f"Nombre del archivo de salida (default: {DEFAULT_OUTPUT_FILE})",
)
parser.add_argument(
    "--device",
    default=DEFAULT_DEVICE_NAME,
    help=f"Nombre del dispositivo de entrada (default: {DEFAULT_DEVICE_NAME})",
)
parser.add_argument(
    "--fs",
    type=int,
    default=DEFAULT_FS,
    help=f"Frecuencia de muestreo en Hz (default: {DEFAULT_FS})",
)
args = parser.parse_args()

OUTPUT_FILE = args.output
DEVICE_NAME = args.device
FS = args.fs

print(f"--- Buscando el corral (dispositivo): {DEVICE_NAME} ---")

# Buscar el índice del Cable Virtual
devices = sd.query_devices()
device_id = None
for i, dev in enumerate(devices):
    if DEVICE_NAME in dev['name'] and dev['max_input_channels'] > 0:
        device_id = i
        break

if device_id is None:
    print("¡No encuentro el Cable! Revisa el nombre en el Panel de Control.")
    exit()

print(f"Grabando desde dispositivo {device_id}. Pulsa 'ESC' para soltar la vara y cerrar.")

# --- La faena de grabación ---
try:
    # Abrimos el archivo para ir escribiendo en el disco "poco a poco"
    with sf.SoundFile(OUTPUT_FILE, mode='w', samplerate=FS, channels=CHANNELS) as file:
        with sd.InputStream(samplerate=FS, device=device_id, channels=CHANNELS) as stream:
            while True:
                # Leemos un trozo (chunk) de audio
                data, overflowed = stream.read(FS) 
                file.write(data)
                
                # Un pequeño indicador de volumen para no aburrirse
                volume_norm = np.linalg.norm(data) * 10
                print(f"|{'#' * int(volume_norm)}".ljust(50) + "|", end="\r")

                if keyboard.is_pressed('esc'):
                    print("\n¡Alto! Guardando y cerrando...")
                    break
except Exception as e:
    print(f"Se ha descarriado el proceso: {e}")

print(f"Fichero guardado: {os.path.abspath(OUTPUT_FILE)}")
