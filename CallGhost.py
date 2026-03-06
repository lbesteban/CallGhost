"""CallGhost: transcription utility with file and live capture modes.

Architecture:
    - `file` mode transcribes an existing audio file and writes TXT/SRT outputs.
    - `live` mode captures microphone/loopback input, writes incremental outputs,
      and optionally regenerates final high-quality outputs from captured WAV.
    - Startup supports both CLI-driven execution and interactive device selection
      when no arguments are provided.
"""

import argparse
import configparser
import os
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import whisper

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import msvcrt
except Exception:
    msvcrt = None

try:
    import winsound
except Exception:
    winsound = None

# Architecture summary:
# - "file" mode transcribes a complete audio file and generates TXT/SRT once.
# - "live" mode captures device audio and processes each chapter into TXT/SRT/audio.
# - Chapter processing runs in a background worker while the main loop remains interactive.

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
DEFAULT_STOP_WHEN_SILENT_FOR = 60.0
DEFAULT_OUTPUT_DIR = "recording"
DEFAULT_INPUT_DEVICE = "CABLE Output"
DEFAULT_OUTPUT_PREFIX_TEMPLATE = "{date}_recording"
DEFAULT_CONFIG_PATH = Path("config.ini")
DEVICE_MONITOR_SECONDS = 3.0
DEVICE_MONITOR_SAMPLE_SECONDS = 0.18

CONFIG_TEMPLATE = """# CallGhost config file
# This file is auto-generated on first run. CLI flags always override these defaults.

[app]
# Whisper model: tiny/base/small/medium/large
model = small
# Recognition language (ISO-639-1)
language = es
# Compute device: cpu/cuda
device_compute = cpu
# Optional absolute path to ffmpeg. Empty means auto-detect.
ffmpeg_path =

[capture]
# Default execution mode when explicitly requested by CLI.
default_mode = live
# Preferred live input device name fragment.
input_device = CABLE Output
# Optional numeric device index. Empty means auto-select or interactive pick.
device_index =
# Silence threshold before entering paused state (seconds)
stop_when_silent_for = 60
# Keep legacy typo key for compatibility if needed.
stop_when_siltent_for =

[audio]
# Default capture format
sample_rate = 16000
channels = 1
rms_threshold = 0.003
chunk_seconds = 6.0
overlap_seconds = 1.0
heartbeat_seconds = 1.0
live_strategy = final-pass
live_audio_format = ogg

[output]
# Default output location and naming
output_dir = recording
output_prefix = {date}_recording
live_audio_out =
txt_out =
srt_out =
"""


def ensure_config_file(config_path: Path) -> None:
    """Create config.ini with comments/defaults when it does not exist."""
    if config_path.exists():
        return
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8", newline="\n")


def _parse_bool(value: str, fallback: bool) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return fallback


def _config_value(config: configparser.ConfigParser, section: str, key: str, fallback: str) -> str:
    if not config.has_section(section):
        return fallback
    value = config.get(section, key, fallback=fallback).strip()
    return value if value else fallback


def _optional_int(config: configparser.ConfigParser, section: str, key: str) -> Optional[int]:
    raw = config.get(section, key, fallback="").strip()
    if not raw:
        return None
    return int(raw)


def _optional_float(config: configparser.ConfigParser, section: str, key: str) -> Optional[float]:
    raw = config.get(section, key, fallback="").strip()
    if not raw:
        return None
    return float(raw)


def load_config_defaults(config_path: Path) -> Dict[str, Any]:
    """Read config defaults and return parser-ready values."""
    ensure_config_file(config_path)
    config = configparser.ConfigParser(interpolation=None)
    config.read(config_path, encoding="utf-8")
    today = datetime.now().strftime("%Y_%m_%d")
    output_prefix_template = _config_value(
        config,
        "output",
        "output_prefix",
        DEFAULT_OUTPUT_PREFIX_TEMPLATE,
    )
    output_prefix_default = output_prefix_template.replace("{date}", today)
    return {
        "model": _config_value(config, "app", "model", "small"),
        "language": _config_value(config, "app", "language", "es"),
        "device_compute": _config_value(config, "app", "device_compute", "cpu"),
        "ffmpeg_path": _config_value(config, "app", "ffmpeg_path", ""),
        "default_mode": _config_value(config, "capture", "default_mode", "live"),
        "input_device": _config_value(config, "capture", "input_device", DEFAULT_INPUT_DEVICE),
        "device_index": _optional_int(config, "capture", "device_index"),
        "stop_when_silent_for": config.getfloat(
            "capture",
            "stop_when_silent_for",
            fallback=DEFAULT_STOP_WHEN_SILENT_FOR,
        ),
        "stop_when_siltent_for": _optional_float(config, "capture", "stop_when_siltent_for"),
        "sample_rate": config.getint("audio", "sample_rate", fallback=TARGET_SR),
        "channels": config.getint("audio", "channels", fallback=1),
        "rms_threshold": config.getfloat("audio", "rms_threshold", fallback=DEFAULT_RMS_THRESHOLD),
        "chunk_seconds": config.getfloat("audio", "chunk_seconds", fallback=DEFAULT_CHUNK_SECONDS),
        "overlap_seconds": config.getfloat("audio", "overlap_seconds", fallback=DEFAULT_OVERLAP_SECONDS),
        "heartbeat_seconds": config.getfloat("audio", "heartbeat_seconds", fallback=DEFAULT_HEARTBEAT_SECONDS),
        "live_strategy": _config_value(config, "audio", "live_strategy", "final-pass"),
        "live_audio_format": _config_value(config, "audio", "live_audio_format", "ogg"),
        "output_dir": _config_value(config, "output", "output_dir", DEFAULT_OUTPUT_DIR),
        "output_prefix": output_prefix_default,
        "live_audio_out": _config_value(config, "output", "live_audio_out", ""),
        "txt_out": _config_value(config, "output", "txt_out", ""),
        "srt_out": _config_value(config, "output", "srt_out", ""),
        "require_punctuation": _parse_bool(
            _config_value(config, "audio", "require_punctuation", ""),
            fallback=False,
        ),
    }


def resolve_ffmpeg_path(ffmpeg_hint: str = "") -> Optional[str]:
    """Resolve ffmpeg executable from hint, local bin, or PATH."""
    if ffmpeg_hint:
        hinted = Path(ffmpeg_hint).expanduser().resolve()
        if hinted.exists():
            return str(hinted)
    local_bin = Path("bin") / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if local_bin.exists():
        return str(local_bin.resolve())
    return shutil.which("ffmpeg")


def emit_pause_beeps(count: int = 3) -> None:
    """Emit audible notifications when silence pause is activated."""
    for _ in range(max(1, count)):
        if winsound is not None:
            winsound.Beep(880, 180)
        else:
            print("\a", end="", flush=True)
            time.sleep(0.18)


def read_pressed_key() -> str:
    """Read one non-blocking key on platforms that support msvcrt."""
    if msvcrt is None or not msvcrt.kbhit():
        return ""
    key = msvcrt.getwch()
    while msvcrt.kbhit():
        msvcrt.getwch()
    return key


def sample_device_rms(device_index: int, device: Dict, rms_threshold: float) -> float:
    """Capture a short sample from a device and return max RMS observed."""
    if sd is None:
        return 0.0
    max_rms = 0.0
    channels = max(1, min(int(device.get("max_input_channels", 1) or 1), 2))
    sample_rate = int(device.get("default_samplerate", TARGET_SR) or TARGET_SR)

    def callback(indata, frames, time_info, status):
        nonlocal max_rms
        del frames, time_info
        if status:
            return
        level = rms(to_mono_float32(indata))
        if level > max_rms:
            max_rms = level

    try:
        with sd.InputStream(
            device=device_index,
            channels=channels,
            samplerate=sample_rate,
            dtype="float32",
            callback=callback,
            blocksize=0,
        ):
            sd.sleep(int(DEVICE_MONITOR_SAMPLE_SECONDS * 1000))
    except Exception:
        return 0.0

    if max_rms < 0.0:
        return 0.0
    return max_rms if max_rms >= rms_threshold else max_rms


def render_activity_indicator(level: float, threshold: float) -> Tuple[str, str]:
    """Render a compact ASCII activity bar and active/silent label."""
    if threshold <= 0:
        threshold = DEFAULT_RMS_THRESHOLD
    ratio = min(1.0, max(0.0, level / threshold))
    bar_slots = 5
    filled = int(round(ratio * bar_slots))
    bar = "[" + ("#" * filled) + (" " * (bar_slots - filled)) + "]"
    status = "Active" if level >= threshold else "Silent"
    return bar, status


def monitor_devices_once(
    devices: List[Tuple[int, Dict]],
    duration_seconds: float,
    rms_threshold: float,
) -> Dict[int, float]:
    """Monitor available devices for a short time window and store max RMS per device."""
    levels: Dict[int, float] = {idx: 0.0 for idx, _ in devices}
    end_time = time.monotonic() + max(0.5, duration_seconds)
    while time.monotonic() < end_time:
        for idx, dev in devices:
            level = sample_device_rms(idx, dev, rms_threshold)
            if level > levels[idx]:
                levels[idx] = level
    return levels


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
        raise RuntimeError("sounddevice is not installed. Install it with: pip install sounddevice")

    devices = sd.query_devices()
    result: List[Tuple[int, Dict]] = []
    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0:
            result.append((idx, dev))
    return result


def print_input_devices(devices: List[Tuple[int, Dict]]) -> None:
    if not devices:
        print("No input devices were found.")
        return

    print("Available input devices:")
    for idx, dev in devices:
        name = dev.get("name", "(unnamed)")
        max_ch = dev.get("max_input_channels", 0)
        sr = int(dev.get("default_samplerate", 0) or 0)
        print(f"  [{idx}] {name} | in_channels={max_ch} | default_sr={sr}")


def pick_input_device(
    devices: List[Tuple[int, Dict]],
    input_device: Optional[str],
    device_index: Optional[int],
) -> Tuple[int, Dict]:
    if not devices:
        raise RuntimeError("No input devices are available.")

    by_index = {idx: dev for idx, dev in devices}

    if device_index is not None:
        if device_index in by_index:
            return device_index, by_index[device_index]
        raise ValueError(f"Device index {device_index} does not exist or is not an input device.")

    if input_device:
        needle = input_device.lower()
        matches = [(i, d) for i, d in devices if needle in str(d.get("name", "")).lower()]
        if matches:
            return matches[0]
        raise ValueError(f"No device name contains: {input_device}")

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


def interactive_startup(args: argparse.Namespace) -> argparse.Namespace:
    """Run interactive live startup when no CLI arguments are provided."""
    devices = list_input_devices()
    if not devices:
        raise RuntimeError("No input devices are available.")

    print("Detected input devices:")
    for idx, dev in devices:
        print(f"  [{idx}] {dev.get('name', '(unnamed)')}")

    print()
    print("Monitoring activity for 3 seconds...")
    levels = monitor_devices_once(
        devices=devices,
        duration_seconds=DEVICE_MONITOR_SECONDS,
        rms_threshold=float(args.rms_threshold),
    )
    for idx, dev in devices:
        level = levels.get(idx, 0.0)
        bar, status = render_activity_indicator(level, float(args.rms_threshold))
        print(f"  [{idx}] {bar} {status:<6} | {dev.get('name', '(unnamed)')}")

    print()
    raw = input("Select device index to record (ESC/Ctrl+C to exit): ").strip()
    if not raw:
        raise ValueError("Empty input. You must provide a numeric device index.")
    args.device_index = int(raw)
    args.input_device = None
    args.mode = "live"
    return args


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


def convert_wav_to_audio(
    wav_path: Path,
    output_path: Path,
    audio_format: str,
    ffmpeg_hint: str = "",
) -> None:
    """Convert a WAV file to the requested compressed audio format."""
    if audio_format == "wav":
        if wav_path != output_path:
            wav_path.replace(output_path)
        return
    ffmpeg_path = resolve_ffmpeg_path(ffmpeg_hint)
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg is not available to convert the final audio.")
    cmd = [ffmpeg_path, "-y", "-i", str(wav_path)]
    if audio_format == "ogg":
        cmd.extend(["-c:a", "libopus", "-b:a", "64k"])
    elif audio_format == "mp3":
        cmd.extend(["-c:a", "libmp3lame", "-b:a", "128k"])
    else:
        raise ValueError(f"Unsupported audio format: {audio_format}")
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
        # Add a blank line for better readability in live incremental output.
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
        print(f"Audio file not found: {audio_path}")
        return 1

    ffmpeg_path = resolve_ffmpeg_path(getattr(args, "ffmpeg_path", ""))
    if ffmpeg_path is None:
        print(
            "Error: ffmpeg was not found. Configure app.ffmpeg_path in config.ini, "
            "place bin/ffmpeg.exe, or add ffmpeg to PATH."
        )
        return 1

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else audio_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    base = audio_path.stem
    txt_path = (Path(args.txt_out).expanduser().resolve() if args.txt_out else out_dir / f"{base}.txt")
    srt_path = (Path(args.srt_out).expanduser().resolve() if args.srt_out else out_dir / f"{base}.srt")

    print(f"SRT: {srt_path}")
    print(f"TXT: {txt_path}")

    print(f"Loading Whisper model ({args.model}, {args.device_compute})...")
    model = whisper.load_model(args.model, device=args.device_compute)

    print("Transcribing file...")
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

    print("Done.")
    return 0


def run_live_mode(args: argparse.Namespace) -> int:
    if sd is None:
        print("Error: sounddevice is not installed. Run: pip install sounddevice")
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
        base = str(args.output_prefix)
    else:
        base = datetime.now().strftime("%Y_%m_%d_recording")

    final_audio_format = str(args.live_audio_format)
    stop_when_silent_for = resolve_stop_when_silent_for(args)
    require_punctuation = resolve_require_punctuation(args)

    stream_sr = int(args.sample_rate)
    stream_channels = int(args.channels)
    device_name = selected_dev.get("name", "(unnamed)")

    chapter_lock = threading.Lock()
    model_lock = threading.Lock()
    processing_done = threading.Event()
    processing_done.set()
    stop_event = threading.Event()
    disk_full_event = threading.Event()
    processing_error: Optional[str] = None
    disk_error_message = ""
    chapter_index = 0
    chapter_active = False
    chapter_wav_fh: Optional[wave.Wave_write] = None
    chapter_wav_path: Optional[Path] = None
    chapter_txt_path: Optional[Path] = None
    chapter_srt_path: Optional[Path] = None
    chapter_audio_path: Optional[Path] = None
    chapter_last_sound = time.monotonic()
    active_outputs: List[Tuple[Path, Path, Path]] = []
    loaded_model = None
    processing_thread: Optional[threading.Thread] = None

    def build_chapter_paths(index: int) -> Tuple[Path, Path, Path, Path]:
        suffix = f"{index:03d}"
        txt_path = out_dir / f"{base}_{suffix}.txt"
        srt_path = out_dir / f"{base}_{suffix}.srt"
        wav_path = out_dir / f"{base}_{suffix}.capture_tmp.wav"
        if args.live_audio_out:
            configured = Path(args.live_audio_out).expanduser().resolve()
            ext = configured.suffix if configured.suffix else f".{final_audio_format}"
            audio_path = configured.with_name(f"{configured.stem}_{suffix}{ext}")
        else:
            audio_path = out_dir / f"{base}_{suffix}.{final_audio_format}"
        return txt_path, srt_path, audio_path, wav_path

    def close_active_chapter(reason: str) -> Optional[Tuple[Path, Path, Path, Path]]:
        nonlocal chapter_active, chapter_wav_fh
        nonlocal chapter_wav_path, chapter_txt_path, chapter_srt_path, chapter_audio_path
        with chapter_lock:
            if not chapter_active or chapter_wav_fh is None:
                return None
            wav_fh = chapter_wav_fh
            wav_path = chapter_wav_path
            txt_path = chapter_txt_path
            srt_path = chapter_srt_path
            audio_path = chapter_audio_path
            chapter_active = False
            chapter_wav_fh = None
            chapter_wav_path = None
            chapter_txt_path = None
            chapter_srt_path = None
            chapter_audio_path = None
        assert wav_path is not None and txt_path is not None and srt_path is not None and audio_path is not None
        wav_fh.close()
        print()
        print(f"Chapter closed ({reason}): {txt_path.stem}")
        return txt_path, srt_path, audio_path, wav_path

    def start_new_chapter() -> bool:
        nonlocal chapter_index, chapter_active, chapter_wav_fh
        nonlocal chapter_wav_path, chapter_txt_path, chapter_srt_path, chapter_audio_path
        nonlocal chapter_last_sound
        if not processing_done.is_set():
            print("The previous chapter is still processing. Wait before starting the next one.")
            return False
        with chapter_lock:
            if chapter_active:
                return False
            chapter_index += 1
            txt_path, srt_path, audio_path, wav_path = build_chapter_paths(chapter_index)
            wav_path.parent.mkdir(parents=True, exist_ok=True)
            wav_fh = wave.open(str(wav_path), "wb")
            wav_fh.setnchannels(1)
            wav_fh.setsampwidth(2)
            wav_fh.setframerate(TARGET_SR)
            chapter_active = True
            chapter_wav_fh = wav_fh
            chapter_wav_path = wav_path
            chapter_txt_path = txt_path
            chapter_srt_path = srt_path
            chapter_audio_path = audio_path
            chapter_last_sound = time.monotonic()

        print()
        print(f"Chapter started: {base}_{chapter_index:03d}")
        print(f"SRT: {srt_path.resolve()}")
        print(f"TXT: {txt_path.resolve()}")
        print(f"AUDIO: {audio_path.resolve()}")
        return True

    def process_chapter(txt_path: Path, srt_path: Path, audio_path: Path, wav_path: Path) -> None:
        nonlocal loaded_model
        with model_lock:
            if loaded_model is None:
                print(f"Loading Whisper model ({args.model}, {args.device_compute})...")
                loaded_model = whisper.load_model(args.model, device=args.device_compute)
            local_model = loaded_model

        print(f"Processing chapter: {txt_path.stem}")
        result = local_model.transcribe(
            str(wav_path),
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

        try:
            audio_format = audio_path.suffix.lstrip(".").lower() or final_audio_format
            convert_wav_to_audio(
                wav_path,
                audio_path,
                audio_format,
                ffmpeg_hint=getattr(args, "ffmpeg_path", ""),
            )
        except Exception as ex:
            print(f"Warning: failed to convert final audio ({ex}). Temporary WAV kept: {wav_path}")
            active_outputs.append((txt_path, srt_path, wav_path))
            return

        wav_path.unlink(missing_ok=True)
        active_outputs.append((txt_path, srt_path, audio_path))
        print(f"Chapter ready: {txt_path.stem}")

    def launch_processing(paths: Tuple[Path, Path, Path, Path]) -> None:
        nonlocal processing_thread, processing_error
        if not processing_done.is_set():
            print("Processing is in progress; another chapter cannot start yet.")
            return
        processing_done.clear()

        def _target() -> None:
            nonlocal processing_error
            try:
                process_chapter(*paths)
            except Exception as ex:
                processing_error = str(ex)
                print(f"Error while processing chapter: {ex}")
            finally:
                processing_done.set()

        processing_thread = threading.Thread(target=_target, daemon=True)
        processing_thread.start()

    def callback(indata, frames, time_info, status):
        nonlocal chapter_last_sound, disk_error_message
        del frames, time_info
        if status:
            return
        mono = to_mono_float32(indata)
        mono_16k = resample_linear(mono, stream_sr, TARGET_SR)
        level = rms(mono_16k)
        pcm = float32_to_pcm16(mono_16k)
        try:
            with chapter_lock:
                if not chapter_active or chapter_wav_fh is None:
                    return
                if level >= args.rms_threshold:
                    chapter_last_sound = time.monotonic()
                chapter_wav_fh.writeframesraw(pcm)
        except OSError as ex:
            disk_error_message = f"Error writing temporary WAV: {ex}"
            disk_full_event.set()
            stop_event.set()

    print(f"Input device: [{selected_index}] {device_name}")
    print("Starting live chapter mode.")
    print("press spacebar to stop and switch chapter")
    print("Press ESC to exit.")
    start_new_chapter()

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
                pressed_key = read_pressed_key()
                if pressed_key == "\x1b":
                    print()
                    print("ESC detected. Closing session...")
                    paths = close_active_chapter("exit")
                    if paths is not None:
                        launch_processing(paths)
                    stop_event.set()
                    break

                if pressed_key == " ":
                    if processing_done.is_set():
                        paths = close_active_chapter("spacebar")
                        if paths is None:
                            start_new_chapter()
                        else:
                            launch_processing(paths)
                    else:
                        print("Previous chapter is still processing. SPACE is temporarily blocked.")

                time.sleep(max(0.1, float(args.heartbeat_seconds)))

                if disk_full_event.is_set():
                    print()
                    print(disk_error_message or "Disk write error. Ending live capture.")
                    paths = close_active_chapter("disk error")
                    if paths is not None and processing_done.is_set():
                        launch_processing(paths)
                    stop_event.set()
                    break

                if args.min_free_mb_stop > 0:
                    free_bytes = shutil.disk_usage(out_dir).free
                    if free_bytes < int(args.min_free_mb_stop * 1024 * 1024):
                        print()
                        print(
                            "Insufficient free disk space "
                            f"(< {args.min_free_mb_stop:.0f} MB). Ending live capture."
                        )
                        paths = close_active_chapter("insufficient disk space")
                        if paths is not None and processing_done.is_set():
                            launch_processing(paths)
                        stop_event.set()
                        break

                with chapter_lock:
                    active = chapter_active
                    silent_for = time.monotonic() - chapter_last_sound

                if active and silent_for >= stop_when_silent_for:
                    print()
                    emit_pause_beeps(3)
                    paths = close_active_chapter("silence")
                    if paths is not None:
                        launch_processing(paths)
                    print("Chapter is processing. Press SPACE to start another when done, or ESC to exit.")

    except KeyboardInterrupt:
        print()
        print("Stopping capture due to Ctrl+C...")
        paths = close_active_chapter("ctrl+c")
        if paths is not None:
            launch_processing(paths)
    finally:
        if processing_thread is not None:
            processing_thread.join()

    if processing_error:
        print(f"Finished with errors: {processing_error}")
        return 1

    print("Finished.")
    if not active_outputs:
        print("No chapters were generated.")
        return 0

    print("Generated outputs:")
    for txt_path, srt_path, audio_path in active_outputs:
        print(f"TXT: {txt_path.resolve()}")
        print(f"SRT: {srt_path.resolve()}")
        print(f"AUDIO: {audio_path.resolve()}")
    return 0


def build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe audio in file or live mode and generate TXT/SRT outputs. "
            "Includes overlap-aware deduplication, status heartbeat, and incremental writes."
        ),
        epilog=(
            "File mode (parameters): audio, --output-dir, --txt-out, --srt-out. \n"
            "Live mode (parameters): --list-devices, --input-device, --device-index, \n"
            "--output-dir, --output-prefix, --chunk-seconds, --overlap-seconds, \n"
            "--heartbeat-seconds, --rms-threshold, --sample-rate, --channels, \n"
            "--live-strategy, --live-audio-out, --live-audio-format, --stop_when_silent_for, \n"
            "--merge-gap-seconds, --min-free-mb-stop. \n"
            "Live chapter mode: SPACE closes/opens chapters (only when processing is idle), \n"
            "ESC exits the session after finalizing the active chapter. \n"
            "File example: CallGhost.py file audio.ogg --output-dir ./output. \n"
            "Live quality example: CallGhost.py live --live-strategy final-pass --output-dir ./output \n"
            "(captures live audio and generates final TXT/SRT at the end). \n"
            "Live hybrid example: CallGhost.py live --live-strategy hybrid --output-dir ./output \n"
            "(live provisional output + final clean regeneration). \n"
            "Note: streaming may reduce final readability due to real-time segmentation.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to config.ini. If missing, it is auto-generated.",
    )
    parser.add_argument(
        "--model",
        default=defaults["model"],
        help="Local Whisper model (tiny, base, small, medium, large).",
    )
    parser.add_argument("--language", default=defaults["language"], help="ISO language code (e.g., es, en, fr).")
    parser.add_argument(
        "--device-compute",
        default=defaults["device_compute"],
        choices=["cpu", "cuda"],
        help="Compute device for Whisper inference.",
    )
    parser.add_argument(
        "--ffmpeg-path",
        default=defaults["ffmpeg_path"],
        help="Explicit ffmpeg path. If omitted, it is resolved from bin/ or PATH.",
    )

    subparsers = parser.add_subparsers(dest="mode")

    p_file = subparsers.add_parser(
        "file",
        help="Transcribe an audio file.",
        description=(
            "File mode: process a full audio file and write TXT/SRT outputs. "
            "If --output-dir is omitted, the input file directory is used."
        ),
    )
    p_file.add_argument("audio", help="Path to the input audio file (e.g., audio.ogg).")
    p_file.add_argument("--output-dir", default=defaults["output_dir"], help="Output directory for TXT/SRT.")
    p_file.add_argument("--txt-out", default=defaults["txt_out"], help="Full output path for TXT.")
    p_file.add_argument("--srt-out", default=defaults["srt_out"], help="Full output path for SRT.")

    p_live = subparsers.add_parser(
        "live",
        help="Transcribe near real-time audio from an input device.",
        description=(
            "Quality-oriented live mode: capture 16k mono WAV and, by default, "
            "generate final TXT/SRT at the end (final-pass)."
        ),
    )
    p_live.add_argument("--list-devices", action="store_true", help="List input devices and exit.")
    p_live.add_argument(
        "--input-device",
        default=defaults["input_device"],
        help=(
            "Substring to match in the input device name. "
            "Default: 'CABLE Output'."
        ),
    )
    p_live.add_argument("--device-index", type=int, default=defaults["device_index"], help="Numeric input device index.")
    p_live.add_argument("--output-dir", default=defaults["output_dir"], help="Output directory for TXT/SRT.")
    p_live.add_argument("--output-prefix", default=defaults["output_prefix"], help="Base prefix for live output files.")
    p_live.add_argument(
        "--live-strategy",
        choices=["final-pass", "hybrid", "streaming"],
        default=defaults["live_strategy"],
        help="Live strategy. Default: final-pass (best final quality).",
    )
    p_live.add_argument(
        "--live-audio-out",
        default=defaults["live_audio_out"],
        help="Final captured audio path. If omitted, it is auto-generated from --live-audio-format.",
    )
    p_live.add_argument(
        "--live-audio-format",
        choices=["ogg", "mp3", "wav"],
        default=defaults["live_audio_format"],
        help="Output format for final captured audio. Default: ogg.",
    )
    p_live.add_argument(
        "--chunk-seconds",
        type=float,
        default=defaults["chunk_seconds"],
        help="Chunk length (s) for streaming/hybrid. Default: 6.0.",
    )
    p_live.add_argument(
        "--overlap-seconds",
        type=float,
        default=defaults["overlap_seconds"],
        help="Chunk overlap (s) for streaming/hybrid. Default: 1.0.",
    )
    p_live.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=defaults["heartbeat_seconds"],
        help="Live console heartbeat interval (s). Default: 1.0.",
    )
    p_live.add_argument("--rms-threshold", type=float, default=defaults["rms_threshold"], help="RMS threshold used to treat audio as silence.")
    p_live.add_argument(
        "--sample-rate",
        type=int,
        default=defaults["sample_rate"],
        help="Capture stream sample rate. Default: 16000 Hz.",
    )
    p_live.add_argument(
        "--channels",
        type=int,
        default=defaults["channels"],
        help="Capture channel count. Default: 1 (mono).",
    )
    p_live.add_argument(
        "--stop_when_silent_for",
        type=float,
        default=defaults["stop_when_silent_for"],
        help="Auto-close the active chapter after this continuous silence duration (s). Default: 60.",
    )
    p_live.add_argument(
        "--stop_when_siltent_for",
        type=float,
        default=defaults["stop_when_siltent_for"],
        help="Legacy typo alias for --stop_when_silent_for.",
    )
    p_live.add_argument(
        "--min-free-mb-stop",
        type=float,
        default=512.0,
        help="Stop capture when free disk space drops below this threshold (MB). Use 0 to disable.",
    )
    p_live.add_argument(
        "--merge-gap-seconds",
        type=float,
        default=DEFAULT_MERGE_GAP_SECONDS,
        help="In final-pass/hybrid, merge segments separated by up to this gap (s).",
    )
    p_live.add_argument(
        "--min-segment-seconds",
        type=float,
        default=DEFAULT_MIN_SEGMENT_SECONDS,
        help="Target minimum duration for final segments (s).",
    )
    p_live.add_argument(
        "--min-final-words",
        type=int,
        default=DEFAULT_MIN_FINAL_WORDS,
        help="Target minimum words per final segment when possible.",
    )
    p_live.add_argument(
        "--require-punctuation",
        dest="require_punctuation",
        action="store_true",
        help="In final-pass/hybrid, prefer closing segments on final punctuation.",
    )
    p_live.add_argument(
        "--no-require-punctuation",
        dest="require_punctuation",
        action="store_false",
        help="Do not require final punctuation to close segments.",
    )
    p_live.set_defaults(require_punctuation=defaults["require_punctuation"])

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


def extract_config_path(argv: List[str]) -> Path:
    """Extract --config path from argv before full parser initialization."""
    for i, token in enumerate(argv):
        if token == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1]).expanduser()
        if token.startswith("--config="):
            return Path(token.split("=", 1)[1]).expanduser()
    return DEFAULT_CONFIG_PATH


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    raw_argv = list(argv)
    config_path = extract_config_path(raw_argv)
    defaults = load_config_defaults(config_path)
    argv = normalize_legacy_args(argv)

    parser = build_parser(defaults)
    if not raw_argv:
        try:
            # argparse only accepts global flags before subcommands.
            args = parser.parse_args(["--config", str(config_path), "live"])
            args = interactive_startup(args)
        except (KeyboardInterrupt, EOFError):
            print()
            print("Cancelled by user.")
            return 1
        except Exception as ex:
            print(f"Interactive mode error: {ex}")
            return 1
    else:
        args = parser.parse_args(argv)

    if args.mode == "file":
        return run_file_mode(args)
    if args.mode == "live":
        return run_live_mode(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
