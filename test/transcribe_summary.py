import argparse
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import whisper

try:
    import sounddevice as sd
except Exception:
    sd = None

# Resumen de arquitectura:
# - Modo "file": transcribe un archivo completo y genera TXT/SRT una sola vez.
# - Modo "live": captura audio por dispositivo, transcribe por chunks con solape y
#   escribe TXT/SRT de forma incremental para monitorizar con Get-Content -Wait.
# - La captura y la transcripcion se desacoplan con una cola para evitar bloqueos.

TARGET_SR = 16000
DEFAULT_CHUNK_SECONDS = 6.0
DEFAULT_OVERLAP_SECONDS = 1.0
DEFAULT_HEARTBEAT_SECONDS = 1.0
DEFAULT_RMS_THRESHOLD = 0.003
LIVE_ACTIVITY_SLOT_SECONDS = 15.0
LIVE_ACTIVITY_SLOTS_PER_LINE = 10
DEFAULT_MERGE_GAP_SECONDS = 0.6
DEFAULT_MIN_SEGMENT_SECONDS = 1.2
DEFAULT_MIN_FINAL_WORDS = 4


def format_srt_timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000.0))
    hours = millis // 3600000
    millis %= 3600000
    minutes = millis // 60000
    millis %= 60000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def normalize_token(token: str) -> str:
    return re.sub(r"[^\w]+", "", token.lower(), flags=re.UNICODE)


def dedupe_prefix_by_words(previous_text: str, new_text: str, max_words: int = 25) -> str:
    prev_words = previous_text.strip().split()
    new_words = new_text.strip().split()
    if not new_words:
        return ""

    best = 0
    upper = min(max_words, len(prev_words), len(new_words))
    for k in range(upper, 0, -1):
        prev_slice = [normalize_token(w) for w in prev_words[-k:]]
        new_slice = [normalize_token(w) for w in new_words[:k]]
        if prev_slice == new_slice:
            best = k
            break

    return " ".join(new_words[best:]).strip()


def tail_words(text: str, max_words: int = 60) -> str:
    words = text.strip().split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[-max_words:])


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float32), dtype=np.float32)))


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32, copy=False)
    if audio.size == 0:
        return np.array([], dtype=np.float32)

    duration = len(audio) / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    out = np.interp(x_new, x_old, audio).astype(np.float32)
    return out


def to_mono_float32(block: np.ndarray) -> np.ndarray:
    arr = np.asarray(block, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    if arr.shape[1] == 1:
        return arr[:, 0]
    return arr.mean(axis=1, dtype=np.float32)


def list_input_devices() -> List[Tuple[int, Dict]]:
    if sd is None:
        raise RuntimeError("sounddevice no esta instalado. Instala con: pip install sounddevice")

    devices = sd.query_devices()
    result: List[Tuple[int, Dict]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            result.append((idx, dev))
    return result


def print_input_devices(devices: List[Tuple[int, Dict]]) -> None:
    if not devices:
        print("No se encontraron dispositivos de entrada.")
        return

    print("Dispositivos de entrada disponibles:")
    for idx, dev in devices:
        name = dev.get("name", "(sin nombre)")
        max_ch = dev.get("max_input_channels", 0)
        sr = int(dev.get("default_samplerate", 0) or 0)
        print(f"  [{idx}] {name} | in_channels={max_ch} | default_sr={sr}")


def pick_input_device(
    devices: List[Tuple[int, Dict]],
    input_device: Optional[str],
    device_index: Optional[int],
) -> Tuple[int, Dict]:
    if not devices:
        raise RuntimeError("No hay dispositivos de entrada disponibles.")

    by_index = {idx: dev for idx, dev in devices}

    if device_index is not None:
        if device_index in by_index:
            return device_index, by_index[device_index]
        raise ValueError(f"No existe device-index {device_index} o no es de entrada.")

    if input_device:
        needle = input_device.lower()
        matches = [(i, d) for i, d in devices if needle in str(d.get("name", "")).lower()]
        if matches:
            return matches[0]
        raise ValueError(f"No se encontro dispositivo que contenga: {input_device}")

    auto_needles = ["cable output", "vb-audio", "vb audio", "cable"]
    for needle in auto_needles:
        matches = [(i, d) for i, d in devices if needle in str(d.get("name", "")).lower()]
        if matches:
            return matches[0]

    default_in = None
    default_device = sd.default.device if sd else None
    if isinstance(default_device, (list, tuple)) and default_device:
        default_in = default_device[0]

    if default_in is not None and default_in in by_index:
        return default_in, by_index[default_in]

    return devices[0]


@dataclass
class LiveStats:
    capture_samples_16k: int = 0
    transcribed_seconds: float = 0.0
    chunks_processed: int = 0
    chunks_skipped_silence: int = 0
    dropped_blocks: int = 0
    state: str = "grabando"


@dataclass
class TimedSegment:
    start: float
    end: float
    text: str


def resolve_stop_when_silent_for(args: argparse.Namespace) -> float:
    """Resolve silence timeout giving priority to the correctly named argument."""
    if getattr(args, "stop_when_silent_for", None) is not None:
        return float(args.stop_when_silent_for)
    legacy = getattr(args, "stop_when_siltent_for", None)
    if legacy is None:
        return 60.0
    return float(legacy)


def resolve_require_punctuation(args: argparse.Namespace) -> bool:
    """Resolve punctuation policy based on strategy and explicit CLI flags."""
    value = getattr(args, "require_punctuation", None)
    if value is not None:
        return bool(value)
    return str(args.live_strategy) in {"final-pass", "hybrid"}


def should_regenerate_final_outputs(live_strategy: str) -> bool:
    """Return whether outputs must be regenerated from full captured audio."""
    return live_strategy in {"final-pass", "hybrid"}


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 mono audio in [-1, 1] to PCM16 bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    return (clipped * 32767.0).astype(np.int16).tobytes()


def extract_timed_segments(result: Dict, offset_sec: float = 0.0) -> List[TimedSegment]:
    """Extract text segments with absolute timestamps from a Whisper result."""
    segments = result.get("segments", []) or []
    extracted: List[TimedSegment] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = offset_sec + float(seg.get("start", 0.0))
        end = offset_sec + float(seg.get("end", start + 0.1))
        extracted.append(TimedSegment(start=start, end=end, text=text))
    if extracted:
        return extracted
    full_text = str(result.get("text", "") or "").strip()
    if not full_text:
        return []
    return [TimedSegment(start=offset_sec, end=offset_sec + 0.1, text=full_text)]


def _join_segments(left: TimedSegment, right: TimedSegment) -> TimedSegment:
    merged_tail = dedupe_prefix_by_words(left.text, right.text, max_words=40)
    merged_text = left.text if not merged_tail else f"{left.text} {merged_tail}".strip()
    return TimedSegment(start=left.start, end=max(left.end, right.end), text=merged_text)


def merge_for_readability(
    segments: List[TimedSegment],
    merge_gap_seconds: float,
    min_segment_seconds: float,
    min_final_words: int,
    require_punctuation: bool,
) -> List[TimedSegment]:
    """Merge short/choppy segments into readable final phrases."""
    if not segments:
        return []

    merged: List[TimedSegment] = [segments[0]]
    for segment in segments[1:]:
        last = merged[-1]
        gap = max(0.0, segment.start - last.end)
        if gap <= merge_gap_seconds:
            merged[-1] = _join_segments(last, segment)
            continue

        words = len(last.text.split())
        short_duration = (last.end - last.start) < min_segment_seconds
        lacks_words = words < min_final_words
        needs_punctuation = require_punctuation and not re.search(r"[.!?…]$", last.text)
        if short_duration or lacks_words or needs_punctuation:
            merged[-1] = _join_segments(last, segment)
            continue

        merged.append(segment)

    normalized: List[TimedSegment] = []
    for segment in merged:
        text = dedupe_prefix_by_words(normalized[-1].text, segment.text, max_words=40) if normalized else segment.text
        text = text.strip()
        if not text:
            continue
        start = max(0.0, float(segment.start))
        end = max(start + 0.1, float(segment.end))
        normalized.append(TimedSegment(start=start, end=end, text=text))
    return normalized


def write_final_outputs(txt_path: Path, srt_path: Path, segments: List[TimedSegment]) -> None:
    """Write final TXT and SRT files from normalized segments."""
    with open(txt_path, "w", encoding="utf-8", newline="\n") as txt_fh:
        for segment in segments:
            txt_fh.write(segment.text + "\n")

    with open(srt_path, "w", encoding="utf-8", newline="\n") as srt_fh:
        for index, segment in enumerate(segments, start=1):
            srt_fh.write(f"{index}\n")
            srt_fh.write(f"{format_srt_timestamp(segment.start)} --> {format_srt_timestamp(segment.end)}\n")
            srt_fh.write(segment.text + "\n\n")


def convert_wav_to_audio(wav_path: Path, output_path: Path, audio_format: str) -> None:
    """Convert a WAV file to the requested compressed audio format."""
    if audio_format == "wav":
        if wav_path != output_path:
            wav_path.replace(output_path)
        return
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg no esta disponible para convertir audio final.")
    cmd = [ffmpeg_path, "-y", "-i", str(wav_path)]
    if audio_format == "ogg":
        cmd.extend(["-c:a", "libopus", "-b:a", "64k"])
    elif audio_format == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "128k"])
    else:
        raise ValueError(f"Formato de audio no soportado: {audio_format}")
    cmd.append(str(output_path))
    subprocess.run(cmd, check=True, capture_output=True, text=True)


class IncrementalWriters:
    def __init__(self, txt_path: Path, srt_path: Path):
        self.txt_path = txt_path
        self.srt_path = srt_path
        self._txt_fh = open(self.txt_path, "a", encoding="utf-8", newline="\n")
        self._srt_fh = open(self.srt_path, "a", encoding="utf-8", newline="\n")
        self.srt_index = 1
        self.last_srt_end = 0.0
        self.txt_tail = ""
        self.srt_tail = ""

    def close(self) -> None:
        self._txt_fh.close()
        self._srt_fh.close()

    def append_txt(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        deduped = dedupe_prefix_by_words(self.txt_tail, text)
        if not deduped:
            return
        # Doble salto para mejorar legibilidad del stream en modo live.
        self._txt_fh.write(deduped + "\n\n")
        self._txt_fh.flush()
        self.txt_tail = tail_words((self.txt_tail + " " + deduped).strip())

    def append_srt_segment(self, start_abs: float, end_abs: float, text: str) -> None:
        text = text.strip()
        if not text:
            return

        deduped = dedupe_prefix_by_words(self.srt_tail, text)
        if not deduped:
            return

        start = max(start_abs, self.last_srt_end)
        end = max(end_abs, start + 0.1)

        self._srt_fh.write(f"{self.srt_index}\n")
        self._srt_fh.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
        self._srt_fh.write(deduped + "\n\n")
        self._srt_fh.flush()

        self.srt_index += 1
        self.last_srt_end = end
        self.srt_tail = tail_words((self.srt_tail + " " + deduped).strip())


def run_file_mode(args: argparse.Namespace) -> int:
    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"Archivo no encontrado: {audio_path}")
        return 1

    if shutil.which("ffmpeg") is None:
        print("Error: 'ffmpeg' no esta disponible en PATH. Instala ffmpeg para modo archivo.")
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = audio_path.stem
    txt_path = (Path(args.txt_out).expanduser().resolve() if args.txt_out else out_dir / f"{base}.txt")
    srt_path = (Path(args.srt_out).expanduser().resolve() if args.srt_out else out_dir / f"{base}.srt")

    print(f"SRT: {srt_path}")
    print(f"TXT: {txt_path}")

    print(f"Cargando modelo Whisper ({args.model}, {args.device_compute})...")
    model = whisper.load_model(args.model, device=args.device_compute)

    print("Transcribiendo archivo...")
    result = model.transcribe(
        str(audio_path),
        language=args.language,
        fp16=(args.device_compute == "cuda"),
        verbose=False,
    )

    segments = result.get("segments", []) or []
    with open(txt_path, "w", encoding="utf-8", newline="\n") as fh:
        if segments:
            for seg in segments:
                text = str(seg.get("text", "")).strip()
                if text:
                    fh.write(text + "\n")
        else:
            full_text = (result.get("text") or "").strip()
            if full_text:
                fh.write(full_text + "\n")

    with open(srt_path, "w", encoding="utf-8", newline="\n") as fh:
        for i, seg in enumerate(segments, start=1):
            start = float(seg.get("start", 0.0))
            end = float(seg.get("end", start + 0.1))
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            fh.write(f"{i}\n")
            fh.write(f"{format_srt_timestamp(start)} --> {format_srt_timestamp(end)}\n")
            fh.write(text + "\n\n")

    print("Listo.")
    return 0


def run_live_mode(args: argparse.Namespace) -> int:
    if sd is None:
        print("Error: sounddevice no esta instalado. Ejecuta: pip install sounddevice")
        return 1

    devices = list_input_devices()

    if args.list_devices:
        print_input_devices(devices)
        return 0

    if args.input_device is None and args.device_index is None:
        print_input_devices(devices)

    try:
        selected_index, selected_dev = pick_input_device(devices, args.input_device, args.device_index)
    except ValueError as ex:
        print(str(ex))
        print_input_devices(devices)
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.output_prefix:
        base = args.output_prefix
    else:
        base = datetime.now().strftime("live_%Y%m%d_%H%M%S")

    txt_path = out_dir / f"{base}.txt"
    srt_path = out_dir / f"{base}.srt"
    final_audio_format = str(args.live_audio_format)
    if args.live_audio_out:
        final_audio_path = Path(args.live_audio_out).expanduser().resolve()
    else:
        final_audio_path = out_dir / f"{base}.{final_audio_format}"
    temp_wav_path = out_dir / f"{base}.capture_tmp.wav"

    print(f"SRT: {srt_path.resolve()}")
    print(f"TXT: {txt_path.resolve()}")
    print(f"AUDIO: {final_audio_path.resolve()}")
    print(f"Dispositivo: [{selected_index}] {selected_dev.get('name', '(sin nombre)')}")

    strategy = str(args.live_strategy)
    stop_when_silent_for = resolve_stop_when_silent_for(args)
    require_punctuation = resolve_require_punctuation(args)
    needs_streaming = strategy in {"streaming", "hybrid"}
    needs_final_pass = should_regenerate_final_outputs(strategy)

    model = None
    if needs_streaming:
        print(f"Cargando modelo Whisper ({args.model}, {args.device_compute})...")
        model = whisper.load_model(args.model, device=args.device_compute)

    audio_q: queue.Queue = queue.Queue(maxsize=128)
    stop_event = threading.Event()
    stats = LiveStats()
    stats_lock = threading.Lock()
    activity_lock = threading.Lock()
    wav_lock = threading.Lock()
    disk_full_event = threading.Event()
    disk_error_message = ""

    chunk_samples = int(args.chunk_seconds * TARGET_SR)
    overlap_samples = int(args.overlap_seconds * TARGET_SR)
    overlap_samples = min(overlap_samples, max(0, chunk_samples - 1))
    stride_samples = max(1, chunk_samples - overlap_samples)

    stream_sr = int(args.sample_rate)
    stream_channels = int(args.channels)

    temp_wav_path.parent.mkdir(parents=True, exist_ok=True)
    wav_fh = wave.open(str(temp_wav_path), "wb")
    wav_fh.setnchannels(1)
    wav_fh.setsampwidth(2)
    wav_fh.setframerate(TARGET_SR)

    writers = IncrementalWriters(txt_path=txt_path, srt_path=srt_path) if needs_streaming else None
    start_monotonic = time.monotonic()
    last_sound_monotonic = start_monotonic
    slot_has_sound = False

    def callback(indata, frames, time_info, status):
        nonlocal last_sound_monotonic, slot_has_sound, disk_error_message
        del frames, time_info
        mono = to_mono_float32(indata)
        mono_16k = resample_linear(mono, stream_sr, TARGET_SR)
        if rms(mono_16k) >= args.rms_threshold:
            now = time.monotonic()
            with activity_lock:
                last_sound_monotonic = now
                slot_has_sound = True
        pcm = float32_to_pcm16(mono_16k)
        try:
            with wav_lock:
                wav_fh.writeframesraw(pcm)
        except OSError as ex:
            disk_error_message = f"Error escribiendo WAV temporal: {ex}"
            disk_full_event.set()
            stop_event.set()
            return

        try:
            if needs_streaming:
                audio_q.put_nowait(mono_16k.copy())
        except queue.Full:
            with stats_lock:
                stats.dropped_blocks += 1

    def transcription_worker():
        assert model is not None
        assert writers is not None
        pending = np.array([], dtype=np.float32)
        chunk_start_sec = 0.0

        while not stop_event.is_set() or not audio_q.empty():
            try:
                block = audio_q.get(timeout=0.5)
            except queue.Empty:
                continue

            with stats_lock:
                stats.capture_samples_16k += len(block)
                stats.state = "grabando"

            if block.size == 0:
                continue

            pending = np.concatenate([pending, block])

            while len(pending) >= chunk_samples:
                chunk = pending[:chunk_samples]
                pending = pending[stride_samples:]

                if rms(chunk) < args.rms_threshold:
                    with stats_lock:
                        stats.chunks_skipped_silence += 1
                        stats.transcribed_seconds = chunk_start_sec + (chunk_samples / TARGET_SR)
                    chunk_start_sec += stride_samples / TARGET_SR
                    continue

                with stats_lock:
                    stats.state = "transcribiendo"

                result = model.transcribe(
                    audio=chunk,
                    language=args.language,
                    fp16=(args.device_compute == "cuda"),
                    verbose=False,
                    condition_on_previous_text=False,
                    temperature=0.0,
                )

                with stats_lock:
                    stats.state = "escribiendo"

                segments = result.get("segments", []) or []
                if not segments:
                    txt = (result.get("text") or "").strip()
                    if txt:
                        writers.append_txt(txt)
                        start = chunk_start_sec
                        end = chunk_start_sec + args.chunk_seconds
                        writers.append_srt_segment(start, end, txt)
                else:
                    for seg in segments:
                        seg_text = str(seg.get("text", "")).strip()
                        if not seg_text:
                            continue
                        start = chunk_start_sec + float(seg.get("start", 0.0))
                        end = chunk_start_sec + float(seg.get("end", 0.0))
                        writers.append_txt(seg_text)
                        writers.append_srt_segment(start, end, seg_text)

                with stats_lock:
                    stats.chunks_processed += 1
                    stats.transcribed_seconds = chunk_start_sec + (chunk_samples / TARGET_SR)
                    stats.state = "grabando"

                chunk_start_sec += stride_samples / TARGET_SR

        if len(pending) >= TARGET_SR:
            chunk = pending
            if rms(chunk) >= args.rms_threshold:
                with stats_lock:
                    stats.state = "transcribiendo"
                result = model.transcribe(
                    audio=chunk,
                    language=args.language,
                    fp16=(args.device_compute == "cuda"),
                    verbose=False,
                    condition_on_previous_text=False,
                    temperature=0.0,
                )
                with stats_lock:
                    stats.state = "escribiendo"
                segments = result.get("segments", []) or []
                if segments:
                    for seg in segments:
                        seg_text = str(seg.get("text", "")).strip()
                        if not seg_text:
                            continue
                        start = chunk_start_sec + float(seg.get("start", 0.0))
                        end = chunk_start_sec + float(seg.get("end", 0.0))
                        writers.append_txt(seg_text)
                        writers.append_srt_segment(start, end, seg_text)
                else:
                    txt = (result.get("text") or "").strip()
                    if txt:
                        writers.append_txt(txt)
                        writers.append_srt_segment(chunk_start_sec, chunk_start_sec + len(chunk) / TARGET_SR, txt)

            with stats_lock:
                stats.state = "grabando"

    worker = None
    if needs_streaming:
        worker = threading.Thread(target=transcription_worker, daemon=True)
        worker.start()

    device_name = selected_dev.get("name", "(sin nombre)")
    print("Iniciando captura live. Pulsa Ctrl+C para detener.")
    print(
        f"Escuchando [{device_name}] | marca cada {LIVE_ACTIVITY_SLOT_SECONDS:.0f}s "
        f"(###=sonido, ___=silencio)"
    )
    activity_markers: List[str] = []
    next_activity_tick = start_monotonic + LIVE_ACTIVITY_SLOT_SECONDS
    last_line_len = 0

    try:
        with sd.InputStream(
            device=selected_index,
            channels=stream_channels,
            samplerate=stream_sr,
            dtype="float32",
            blocksize=0,
            callback=callback,
        ):
            while not stop_event.is_set():
                time.sleep(args.heartbeat_seconds)
                if disk_full_event.is_set():
                    print()
                    print(disk_error_message or "Error de escritura en disco. Finalizando captura live.")
                    stop_event.set()
                    break
                now = time.monotonic()
                with activity_lock:
                    silent_for = now - last_sound_monotonic

                if args.min_free_mb_stop > 0:
                    free_bytes = shutil.disk_usage(temp_wav_path.parent).free
                    if free_bytes < int(args.min_free_mb_stop * 1024 * 1024):
                        print()
                        print(
                            "Espacio libre insuficiente en disco "
                            f"(< {args.min_free_mb_stop:.0f} MB). Finalizando captura live."
                        )
                        stop_event.set()
                        break

                if silent_for >= stop_when_silent_for:
                    print()
                    print(
                        "Silencio detectado durante "
                        f"{stop_when_silent_for:.0f}s. Finalizando captura live."
                    )
                    stop_event.set()
                    break

                while now >= next_activity_tick and not stop_event.is_set():
                    with activity_lock:
                        marker = "###" if slot_has_sound else "___"
                        slot_has_sound = False
                    activity_markers.append(marker)
                    if len(activity_markers) > LIVE_ACTIVITY_SLOTS_PER_LINE:
                        activity_markers = activity_markers[-LIVE_ACTIVITY_SLOTS_PER_LINE:]

                    line = f"Escuchando [{device_name}] {' '.join(activity_markers)}"
                    clear = "\r" + (" " * last_line_len) + "\r"
                    print(clear + line, end="", flush=True)
                    last_line_len = len(line)
                    next_activity_tick += LIVE_ACTIVITY_SLOT_SECONDS

                    if len(activity_markers) == LIVE_ACTIVITY_SLOTS_PER_LINE:
                        time.sleep(0.1)
                        print("\r" + (" " * last_line_len) + "\r", end="", flush=True)
                        last_line_len = 0
                        activity_markers = []

    except KeyboardInterrupt:
        print("Deteniendo captura...")
    finally:
        print()
        stop_event.set()
        if worker is not None:
            worker.join(timeout=15)
        if writers is not None:
            writers.close()
        with wav_lock:
            wav_fh.close()

    if needs_final_pass:
        if model is None:
            print(f"Cargando modelo Whisper ({args.model}, {args.device_compute})...")
            model = whisper.load_model(args.model, device=args.device_compute)
        print("Generando transcripcion final desde WAV temporal...")
        result = model.transcribe(
            str(temp_wav_path),
            language=args.language,
            fp16=(args.device_compute == "cuda"),
            verbose=False,
        )
        raw_segments = extract_timed_segments(result)
        merged = merge_for_readability(
            segments=raw_segments,
            merge_gap_seconds=float(args.merge_gap_seconds),
            min_segment_seconds=float(args.min_segment_seconds),
            min_final_words=int(args.min_final_words),
            require_punctuation=require_punctuation,
        )
        write_final_outputs(txt_path=txt_path, srt_path=srt_path, segments=merged)
    elif not txt_path.exists():
        txt_path.write_text("", encoding="utf-8")
        srt_path.write_text("", encoding="utf-8")

    try:
        convert_wav_to_audio(temp_wav_path, final_audio_path, final_audio_format)
    except Exception as ex:
        print(f"Advertencia: no se pudo convertir audio final ({ex}). Se conserva WAV temporal.")
        final_audio_path = temp_wav_path
    else:
        if final_audio_path != temp_wav_path and temp_wav_path.exists():
            temp_wav_path.unlink(missing_ok=True)

    print("Finalizado.")
    print(f"SRT: {srt_path.resolve()}")
    print(f"TXT: {txt_path.resolve()}")
    print(f"AUDIO: {final_audio_path.resolve()}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio en modo archivo o live y genera salidas TXT/SRT. "
            "Incluye deduplicacion por solape, heartbeat de estado y escritura incremental."
        ),
        epilog=(
            "Modo file (parametros): audio, --output-dir, --txt-out, --srt-out. \n"
            "Modo live (parametros): --list-devices, --input-device, --device-index, \n"
            "--output-dir, --output-prefix, --chunk-seconds, --overlap-seconds, \n"
            "--heartbeat-seconds, --rms-threshold, --sample-rate, --channels, \n"
            "--live-strategy, --live-audio-out, --live-audio-format, --stop_when_silent_for, \n"
            "--merge-gap-seconds, --min-free-mb-stop. \n"
            "Ejemplo file: transcribe_summary.py file audio.ogg --output-dir ./salida. \n"
            "Ejemplo live calidad: transcribe_summary.py live --live-strategy final-pass --output-dir ./salida \n"
            "(captura live y genera TXT/SRT finales al terminar). \n"
            "Ejemplo live hybrid: transcribe_summary.py live --live-strategy hybrid --output-dir ./salida \n"
            "(provisional en vivo + regeneracion final limpia). \n"
            "Nota: streaming puede degradar legibilidad final por fragmentacion en tiempo real.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--model", default="small", help="Modelo Whisper local (tiny, base, small, medium, large)")
    parser.add_argument("--language", default="es", help="Idioma ISO (ej. es, en, fr).")
    parser.add_argument(
        "--device-compute",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Dispositivo de inferencia para Whisper.",
    )

    subparsers = parser.add_subparsers(dest="mode")

    p_file = subparsers.add_parser(
        "file",
        help="Transcribe un archivo de audio.",
        description=(
            "Modo archivo: procesa un audio completo y guarda TXT/SRT. "
            "Si no se indica --output-dir, usa la carpeta del audio de entrada."
        ),
    )
    p_file.add_argument("audio", help="Ruta al archivo de audio (ej. archivo.ogg)")
    p_file.add_argument("--output-dir", help="Directorio de salida para TXT/SRT")
    p_file.add_argument("--txt-out", help="Ruta completa del TXT de salida")
    p_file.add_argument("--srt-out", help="Ruta completa del SRT de salida")

    p_live = subparsers.add_parser(
        "live",
        help="Transcribe en casi tiempo real desde dispositivo de entrada.",
        description=(
            "Modo live orientado a calidad: captura audio en WAV mono 16k y, por defecto, "
            "genera TXT/SRT finales al terminar (final-pass)."
        ),
    )
    p_live.add_argument("--list-devices", action="store_true", help="Lista dispositivos de entrada y sale.")
    p_live.add_argument(
        "--input-device",
        default="CABLE Output",
        help=(
            "Texto a buscar en el nombre del dispositivo de entrada. "
            "Default: 'CABLE Output'."
        ),
    )
    p_live.add_argument("--device-index", type=int, help="Indice numerico del dispositivo de entrada.")
    p_live.add_argument("--output-dir", default=".", help="Directorio donde se guardaran TXT/SRT.")
    p_live.add_argument("--output-prefix", help="Prefijo base para los archivos live.")
    p_live.add_argument(
        "--live-strategy",
        choices=["final-pass", "hybrid", "streaming"],
        default="final-pass",
        help="Estrategia live. Default: final-pass (maxima calidad final).",
    )
    p_live.add_argument(
        "--live-audio-out",
        help="Ruta del audio final capturado. Si no se indica, se autogenera con --live-audio-format.",
    )
    p_live.add_argument(
        "--live-audio-format",
        choices=["ogg", "mp3", "wav"],
        default="ogg",
        help="Formato del audio final guardado. Default: ogg.",
    )
    p_live.add_argument(
        "--chunk-seconds",
        type=float,
        default=DEFAULT_CHUNK_SECONDS,
        help="Tamano de chunk (s) para streaming/hybrid. Default: 6.0.",
    )
    p_live.add_argument(
        "--overlap-seconds",
        type=float,
        default=DEFAULT_OVERLAP_SECONDS,
        help="Solape entre chunks (s) para streaming/hybrid. Default: 1.0.",
    )
    p_live.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=DEFAULT_HEARTBEAT_SECONDS,
        help="Refresco de consola en live (s). Default: 1.0.",
    )
    p_live.add_argument("--rms-threshold", type=float, default=DEFAULT_RMS_THRESHOLD, help="Umbral RMS para saltar silencio.")
    p_live.add_argument(
        "--sample-rate",
        type=int,
        default=TARGET_SR,
        help="Sample rate del stream de captura. Default: 16000 Hz.",
    )
    p_live.add_argument(
        "--channels",
        type=int,
        default=1,
        help="Canales de captura. Default: 1 (mono).",
    )
    p_live.add_argument(
        "--stop_when_silent_for",
        type=float,
        default=60.0,
        help="Detiene live tras este tiempo continuo de silencio (s). Default: 60.",
    )
    p_live.add_argument(
        "--stop_when_siltent_for",
        type=float,
        default=None,
        help="Alias legacy con typo de --stop_when_silent_for.",
    )
    p_live.add_argument(
        "--min-free-mb-stop",
        type=float,
        default=512.0,
        help="Detiene captura si el espacio libre baja de este umbral (MB). 0 para desactivar.",
    )
    p_live.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=DEFAULT_MERGE_GAP_SECONDS,
        help="En final-pass/hybrid, fusiona segmentos separados por este gap maximo (s).",
    )
    p_live.add_argument(
        "--min-segment-seconds",
        type=float,
        default=DEFAULT_MIN_SEGMENT_SECONDS,
        help="Duracion minima objetivo de segmento final (s).",
    )
    p_live.add_argument(
        "--min-final-words",
        type=int,
        default=DEFAULT_MIN_FINAL_WORDS,
        help="Numero minimo de palabras por segmento final cuando sea posible.",
    )
    p_live.add_argument(
        "--require-punctuation",
        dest="require_punctuation",
        action="store_true",
        help="En final-pass/hybrid, intenta cerrar segmentos con puntuacion final.",
    )
    p_live.add_argument(
        "--no-require-punctuation",
        dest="require_punctuation",
        action="store_false",
        help="No exige puntuacion final para cerrar segmentos.",
    )
    p_live.set_defaults(require_punctuation=None)

    parser.add_argument(
        "legacy_audio",
        nargs="?",
        help=argparse.SUPPRESS,
    )

    return parser


def normalize_legacy_args(argv: List[str]) -> List[str]:
    if not argv:
        return argv

    known = {"file", "live"}
    first = argv[0]
    if first in known or first.startswith("-"):
        return argv

    return ["file"] + argv


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = normalize_legacy_args(argv)

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.mode == "file":
        return run_file_mode(args)
    if args.mode == "live":
        return run_live_mode(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
