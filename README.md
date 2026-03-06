# CallGhost

`CallGhost.py` is a practical speech-to-text utility that transcribes:
- audio files (`file` mode), and
- live input devices (`live` mode) with chapter-based recording.

It generates `.txt` and `.srt` outputs and supports workflows where recordings are later processed with AI tools.

Spanish documentation is available at [`README_es.md`](README_es.md).

## What This Project Is For

CallGhost is a proof of concept focused on two goals:
1. Build something useful with vibe coding.
2. Demonstrate a disciplined engineering workflow (clear scope, readable code, pragmatic iteration).

It is intended for local, practical experimentation and not as a production-grade compliance tool.

## Core Features

- File transcription to TXT/SRT.
- Live capture from an input device.
- Interactive startup when no arguments are provided (device list + quick activity probe).
- Chapter-based live mode:
  - `SPACE`: stop current chapter and switch to the next chapter.
  - `ESC`: exit the session gracefully.
- Auto chapter close on prolonged silence (`--stop_when_silent_for`) with audible beeps.
- Sequential chapter processing in background threads (prevents overlapping chapter transitions).
- Optional ffmpeg conversion to final audio formats (`ogg`, `mp3`, `wav`).

## Setup

```powershell
# Create virtual environment
python -m venv .venv

# Activate
.\.venv\Scripts\Activate.ps1

# Upgrade packaging tools
python -m pip install --upgrade pip setuptools wheel

# Install dependencies
pip install openai-whisper sounddevice numpy

# Optional: CPU-only PyTorch build (if you do not use CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# ffmpeg for audio conversion (needed for some outputs)
mkdir bin -ErrorAction SilentlyContinue
$ffmpegUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
Invoke-WebRequest -Uri $ffmpegUrl -OutFile "ffmpeg.zip"
Expand-Archive -Path "ffmpeg.zip" -DestinationPath "temp_ffmpeg" -Force
Move-Item "temp_ffmpeg\ffmpeg-*\bin\ffmpeg.exe" "bin\ffmpeg.exe"
Remove-Item "ffmpeg.zip"
Remove-Item "temp_ffmpeg" -Recurse
```

## Basic Usage

```powershell
# List input devices
python .\CallGhost.py live --list-devices

# Live mode (device auto-match by name)
python .\CallGhost.py live --input-device "CABLE" --output-dir .\recording

# Live mode (specific device index)
python .\CallGhost.py live --device-index 3 --output-dir .\recording --output-prefix class_notes

# File mode
python .\CallGhost.py file ".\audio.ogg" --output-dir .\output

# Interactive mode (no args)
python .\CallGhost.py
```

## Live Chapter Workflow

During `live` mode, the app shows:
- `press spacebar to stop and switch chapter`

Behavior:
- Session starts with chapter `..._001`.
- Press `SPACE` to close the active chapter and process it.
- Press `SPACE` again (once processing is done) to start the next chapter (`..._002`, `..._003`, ...).
- If silence exceeds `--stop_when_silent_for`, the active chapter closes automatically, gets processed, and the app waits for `SPACE` (new chapter) or `ESC` (exit).
- Press `ESC` at any time to end the session cleanly.

## Practical Use Cases

1. Class recording -> AI study notes
- Capture each section as a separate chapter.
- Feed chapter TXT files into GPT Projects or NotebookLM for summaries, Q&A, glossary extraction, and revision plans.

2. Meeting recording -> structured action items
- Capture team meetings and segment by agenda chapters.
- Use GPT/Perplexity Spaces to generate decisions, owners, deadlines, and follow-up prompts.

3. Online video / course / talk transcription
- Route system audio to a virtual input device and capture chapter-by-topic.
- Build searchable SRT+TXT materials for later synthesis and note generation.

4. Research sessions and interviews
- Keep chapter boundaries aligned with topics.
- Export chapter text into AI tools for synthesis, contradiction checks, and thematic clustering.

## Responsible Use

Use this project responsibly:
- Obtain consent before recording people.
- Follow local laws and organizational policies for recording and transcription.
- Do not upload sensitive data to third-party AI tools unless policy allows it.
- Redact personal or confidential content before external processing.

This project is a proof of concept designed to explore useful outcomes with vibe coding, not a legal/compliance framework.

## CPU vs CUDA (When To Use Each)

`--device-compute cpu`
- Best for portability and simple setups.
- Slower inference, especially on longer recordings or larger models.
- Lower setup complexity and fewer compatibility issues.

`--device-compute cuda`
- Best for faster transcription throughput.
- Typically much faster on supported NVIDIA GPUs.
- Requires compatible GPU drivers/CUDA stack and PyTorch CUDA build.

Quick rule of thumb:
- Use `cpu` if reliability and simplicity are the priority.
- Use `cuda` when speed matters and your environment is GPU-ready.

## CLI Parameters (Summary)

Global:
- `--config`: path to `config.ini` (auto-created if missing).
- `--model`: Whisper model (`tiny`, `base`, `small`, `medium`, `large`).
- `--language`: ISO language code.
- `--device-compute`: `cpu` or `cuda`.
- `--ffmpeg-path`: explicit ffmpeg path.

`file` mode:
- `audio`
- `--output-dir`
- `--txt-out`
- `--srt-out`

`live` mode:
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

## Lessons Learned (from `AGENTS.md` Practices)

This project reinforces the intent of `AGENTS.md`:
- Clarity over cleverness produces maintainable code under iteration pressure.
- Small, scoped changes reduce regressions and speed up validation.
- Documentation quality is part of done-ness, not optional polish.
- Explicit constraints (typing, error handling, scope control) improve AI-assisted coding reliability.
- “Trust but notify” autonomy works when changes are verified and reported clearly.

Target outcome:
- Use vibe coding to produce practical tools that are actually usable, testable, and understandable by humans.

## How To Write Better Prompts for Vibe Coding Agents

Use this structure (works for Codex or other agents):
1. Objective: what must be built/fixed.
2. Scope boundaries: files/components allowed and forbidden.
3. Behavior contract: exact user-visible behavior and edge cases.
4. Constraints: style, dependencies, performance, and compatibility limits.
5. Verification: what command/tests must pass.
6. Output expectations: summary format and required references.

Prompt quality tips:
- Be specific about acceptance criteria.
- Include concrete examples and failure cases.
- Ask for minimal, reversible changes when possible.
- Require validation commands in every implementation cycle.

## License

This project is licensed under **GNU GPL v3.0**.
See [`LICENSE`](LICENSE) for details.
