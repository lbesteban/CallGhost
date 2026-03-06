# convert audio to srt/txt

Script principal: `CallGhost.py`

Resumen:
- `file`: transcribe un archivo y genera `TXT` + `SRT`.
- `live`: captura audio de un dispositivo de entrada y genera `TXT` + `SRT` en incremental.
- Compatible con `Get-Content -Wait` por flush en disco.
- Si se ejecuta sin argumentos, entra en modo interactivo: lista dispositivos, monitoriza actividad 3s (Active/Silent) y pide indice.
- Genera automaticamente `config.ini` si no existe. Los flags CLI siempre tienen prioridad sobre el config.

## setup

```powershell
# Crear entorno virtual
python -m venv .venv

# Activarlo
.\.venv\Scripts\Activate.ps1

# Actualizar pip
python -m pip install --upgrade pip setuptools wheel

# Instalar PyTorch SOLO CPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Instalar dependencias
pip install openai-whisper sounddevice numpy

# ffmpeg (necesario para modo file)
# Crear carpeta para binarios
mkdir bin -ErrorAction SilentlyContinue

# Descargar el zip de FFmpeg (Build esencial de gyan.dev, que es la pata negra)
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
Invoke-WebRequest -Uri $ffmpegUrl -OutFile "ffmpeg.zip"

# Extraer solo el ffmpeg.exe
Expand-Archive -Path "ffmpeg.zip" -DestinationPath "temp_ffmpeg" -Force
Move-Item "temp_ffmpeg\ffmpeg-*\bin\ffmpeg.exe" "bin\ffmpeg.exe"

# Limpieza
Remove-Item "ffmpeg.zip"
Remove-Item "temp_ffmpeg" -Recurse

```

## ejemplos de uso

```powershell
# Listar dispositivos de entrada para modo live
python .\CallGhost.py live --list-devices

# Modo live con autoseleccion por nombre o substring
python .\CallGhost.py live --input-device "CABLE" --output-dir .\recording

# Modo live por indice de dispositivo
python .\CallGhost.py live --device-index 3 --output-dir .\recording --output-prefix reunion

# Modo archivo
python .\CallGhost.py file ".\audio.ogg" --output-dir .\salida

# Modo interactivo (sin argumentos)
python .\CallGhost.py
```

## parametros CLI

Globales:
- `--config`: ruta al `config.ini` (se auto-crea si no existe).
- `--model`: modelo Whisper local (`tiny`, `base`, `small`, `medium`, `large`).
- `--language`: idioma ISO (ej. `es`, `en`).
- `--device-compute`: `cpu` o `cuda`.
- `--ffmpeg-path`: ruta explicita de `ffmpeg`.

Subcomando `file`:
- `audio`: ruta de audio de entrada.
- `--output-dir`: directorio de salida para `TXT/SRT`.
- `--txt-out`: ruta completa para el `TXT`.
- `--srt-out`: ruta completa para el `SRT`.

Subcomando `live`:
- `--list-devices`: lista dispositivos de entrada y termina.
- `--input-device`: substring del nombre del dispositivo.
- `--device-index`: indice numerico del dispositivo.
- `--output-dir`: carpeta de salida de `TXT/SRT` (default `recording`).
- `--output-prefix`: prefijo del nombre de archivos de salida.
- `--chunk-seconds`: tamano del chunk (default `6.0`).
- `--overlap-seconds`: solape entre chunks (default `1.0`).
- `--heartbeat-seconds`: frecuencia de estado en consola (default `1.0`).
- `--rms-threshold`: umbral para saltar silencios (default `0.003`).
- `--sample-rate`: sample rate de captura opcional.
- `--channels`: canales de captura opcional.
- `--stop_when_silent_for`: umbral para entrar en pausa por silencio.
