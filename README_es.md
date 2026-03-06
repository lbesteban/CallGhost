# CallGhost

`CallGhost.py` es una utilidad práctica de voz a texto que transcribe:
- archivos de audio (modo `file`), y
- dispositivos de entrada en vivo (modo `live`) con grabación por capítulos.

Genera salidas `.txt` y `.srt` y encaja bien en flujos donde luego procesas el contenido con herramientas de IA.

## Para Qué Sirve Este Proyecto

CallGhost es una prueba de concepto enfocada en dos objetivos:
1. Construir algo útil con vibe coding.
2. Demostrar un flujo de ingeniería disciplinado (alcance claro, código legible, iteración pragmática).

Está pensado para experimentación local y práctica, no como una herramienta de cumplimiento normativo en producción.

## Funcionalidades Principales

- Transcripción de archivos a TXT/SRT.
- Captura en vivo desde dispositivo de entrada.
- Arranque interactivo sin argumentos (lista de dispositivos + sondeo rápido de actividad).
- Modo en vivo por capítulos:
  - `SPACE`: cierra el capítulo actual y cambia al siguiente.
  - `ESC`: cierra la sesión de forma segura.
- Cierre automático de capítulo por silencio prolongado (`--stop_when_silent_for`) con pitidos.
- Procesamiento secuencial de capítulos en segundo plano (evita solapes en el cambio de capítulo).
- Conversión opcional con ffmpeg a formatos finales (`ogg`, `mp3`, `wav`).

## Instalación

```powershell
# Crear entorno virtual
python -m venv .venv

# Activar
.\.venv\Scripts\Activate.ps1

# Actualizar herramientas de empaquetado
python -m pip install --upgrade pip setuptools wheel

# Instalar dependencias
pip install openai-whisper sounddevice numpy

# Opcional: build CPU-only de PyTorch (si no usas CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ffmpeg para conversión de audio (necesario para algunas salidas)
mkdir bin -ErrorAction SilentlyContinue
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
Invoke-WebRequest -Uri $ffmpegUrl -OutFile "ffmpeg.zip"
Expand-Archive -Path "ffmpeg.zip" -DestinationPath "temp_ffmpeg" -Force
Move-Item "temp_ffmpeg\ffmpeg-*\bin\ffmpeg.exe" "bin\ffmpeg.exe"
Remove-Item "ffmpeg.zip"
Remove-Item "temp_ffmpeg" -Recurse
```

## Uso Básico

```powershell
# Listar dispositivos de entrada
python .\CallGhost.py live --list-devices

# Modo live (auto-coincidencia por nombre)
python .\CallGhost.py live --input-device "CABLE" --output-dir .\recording

# Modo live (índice de dispositivo concreto)
python .\CallGhost.py live --device-index 3 --output-dir .\recording --output-prefix class_notes

# Modo file
python .\CallGhost.py file ".\audio.ogg" --output-dir .\output

# Modo interactivo (sin argumentos)
python .\CallGhost.py
```

## Flujo de Capítulos en Live

Durante `live`, la app muestra:
- `press spacebar to stop and switch chapter`

Comportamiento:
- La sesión empieza en el capítulo `..._001`.
- Pulsa `SPACE` para cerrar el capítulo activo y procesarlo.
- Pulsa `SPACE` otra vez (cuando termine el procesamiento) para iniciar el siguiente (`..._002`, `..._003`, ...).
- Si el silencio supera `--stop_when_silent_for`, el capítulo activo se cierra automáticamente, se procesa y la app espera `SPACE` (nuevo capítulo) o `ESC` (salir).
- Pulsa `ESC` en cualquier momento para cerrar la sesión de forma limpia.

## Casos de Uso Prácticos

1. Grabar una clase -> apuntes con IA
- Captura cada bloque temático como capítulo.
- Pasa los TXT a GPT Projects o NotebookLM para resúmenes, Q&A, glosarios y plan de repaso.

2. Grabar reuniones -> acciones estructuradas
- Segmenta por agenda en capítulos.
- Usa GPT/Perplexity Spaces para extraer decisiones, responsables, fechas y siguientes pasos.

3. Transcribir video/curso/charla online
- Enruta audio de sistema a un dispositivo virtual y captura por capítulos temáticos.
- Genera SRT+TXT buscables para síntesis y creación de notas.

4. Entrevistas o sesiones de investigación
- Mantén fronteras de capítulo alineadas con temas.
- Exporta texto por capítulo para síntesis, chequeo de contradicciones y clustering temático.

## Uso Responsable

Usa este proyecto de forma responsable:
- Obtén consentimiento antes de grabar a otras personas.
- Cumple leyes locales y políticas internas sobre grabación/transcripción.
- No subas datos sensibles a servicios externos de IA sin autorización.
- Redacta datos personales o confidenciales antes de procesar fuera.

Este proyecto es una prueba de concepto para explorar resultados útiles con vibe coding, no un marco legal/compliance.

## CPU vs CUDA (Cuándo Usar Cada Uno)

`--device-compute cpu`
- Mejor para portabilidad y setups simples.
- Inferencia más lenta, sobre todo con audios largos o modelos grandes.
- Menor complejidad de instalación y menos problemas de compatibilidad.

`--device-compute cuda`
- Mejor para mayor velocidad de transcripción.
- Suele ser bastante más rápido con GPU NVIDIA compatible.
- Requiere drivers/CUDA compatibles y build CUDA de PyTorch.

Regla rápida:
- Usa `cpu` si priorizas simplicidad y fiabilidad.
- Usa `cuda` si priorizas velocidad y tu entorno ya está preparado.

## Parámetros CLI (Resumen)

Globales:
- `--config`: ruta a `config.ini` (se auto-crea si no existe).
- `--model`: modelo Whisper (`tiny`, `base`, `small`, `medium`, `large`).
- `--language`: código de idioma ISO.
- `--device-compute`: `cpu` o `cuda`.
- `--ffmpeg-path`: ruta explícita a ffmpeg.

Modo `file`:
- `audio`
- `--output-dir`
- `--txt-out`
- `--srt-out`

Modo `live`:
- `--list-devices`
- `--input-device`
- `--device-index`
- `--output-dir`
- `--output-prefix`
- `--chunk-seconds`
- `--overlap-seconds`
- `--heartbeat-seconds`
- `--rms-threshold`
- `--sample-rate`
- `--channels`
- `--stop_when_silent_for`
- `--live-strategy`
- `--live-audio-out`
- `--live-audio-format`

## Lecciones Aprendidas (desde `AGENTS.md`)

Este proyecto refuerza la intención de `AGENTS.md`:
- Claridad por encima de “soluciones ingeniosas” mejora mantenibilidad.
- Cambios pequeños y acotados reducen regresiones y aceleran validación.
- La documentación forma parte de “done”, no es opcional.
- Restricciones explícitas (typing, manejo de errores, control de alcance) mejoran fiabilidad al trabajar con IA.
- La autonomía “trust but notify” funciona cuando cada cambio se valida y se reporta con precisión.

Objetivo final:
- Usar vibe coding para crear herramientas útiles, testeables y comprensibles por humanos.

## Cómo Escribir Mejores Prompts para Herramientas de Vibe Coding

Usa esta estructura (válida para Codex u otros agentes):
1. Objetivo: qué hay que construir/arreglar.
2. Límites de alcance: qué archivos/componentes sí y no.
3. Contrato de comportamiento: comportamiento visible exacto y casos límite.
4. Restricciones: estilo, dependencias, rendimiento y compatibilidad.
5. Verificación: comandos/tests que deben pasar.
6. Formato de salida: resumen y referencias obligatorias.

Consejos:
- Define criterios de aceptación concretos.
- Incluye ejemplos y casos de fallo.
- Pide cambios mínimos y reversibles cuando sea posible.
- Exige validación por comandos en cada ciclo.

## Licencia

Este proyecto está licenciado bajo **GNU GPL v3.0**.
Consulta [`LICENSE`](LICENSE) para más detalles.
