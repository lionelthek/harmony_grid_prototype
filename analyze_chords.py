#!/usr/bin/env python3
"""
analyze_chords.py

Prototype CLI for turning a local audio file into a simple jazz/pop chord grid.

This is not a perfect transcription system. It is a first-pass harmonic assistant:
- estimates tempo and beat positions;
- extracts chroma features;
- classifies frames against simple chord templates;
- smooths and quantizes chords into measures;
- exports Markdown and JSON.

Usage:
    python analyze_chords.py path/to/song.mp3 --out grid.md --json analysis.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


NOTE_NAMES_SHARP = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
FRENCH_NOTE_TO_ENGLISH = {
    "do": "C",
    "do#": "C#",
    "réb": "C#",
    "reb": "C#",
    "ré": "D",
    "re": "D",
    "ré#": "Eb",
    "re#": "Eb",
    "mib": "Eb",
    "mi": "E",
    "fa": "F",
    "fa#": "F#",
    "solb": "F#",
    "sol": "G",
    "sol#": "Ab",
    "lab": "Ab",
    "la": "A",
    "la#": "Bb",
    "sib": "Bb",
    "si": "B",
}


@dataclass
class ChordSegment:
    start: float
    end: float
    symbol: str
    confidence: float


@dataclass
class ChordBlock:
    symbol: str
    beat_start: int
    beat_duration: float
    confidence: float


@dataclass
class GridMeasure:
    measure: int
    start: float
    end: float
    chords: list[ChordBlock]


@dataclass
class AnalysisResult:
    source_file: str
    duration: float
    tempo: float
    meter: str
    key_guess: str
    chord_segments: list[ChordSegment]
    grid: list[GridMeasure]


def require_dependencies() -> None:
    missing = []
    for package in ["librosa", "soundfile"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing Python dependencies: {joined}\n"
            "Install them with:\n"
            "    pip install -r requirements.txt\n"
        )


def convert_to_wav(input_path: Path, sample_rate: int) -> Path:
    """Convert any FFmpeg-readable file to mono WAV for stable loading."""
    tmp = tempfile.NamedTemporaryFile(prefix="analyze_chords_", suffix=".wav", delete=False)
    tmp.close()
    wav_path = Path(tmp.name)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        str(wav_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise SystemExit("FFmpeg is required but was not found in PATH.") from exc
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Could not decode audio file with FFmpeg: {input_path}") from exc
    return wav_path


def normalize_vector(v: np.ndarray) -> np.ndarray:
    v = np.maximum(v.astype(float), 0.0)
    total = float(np.sum(v))
    if total <= 1e-9:
        return np.zeros_like(v)
    return v / total


def rotate_template(intervals: Iterable[int], root: int, weights: dict[int, float] | None = None) -> np.ndarray:
    template = np.zeros(12, dtype=float)
    weights = weights or {}
    for interval in intervals:
        template[(root + interval) % 12] = weights.get(interval, 1.0)
    return normalize_vector(template)


def chord_templates() -> list[tuple[str, np.ndarray]]:
    """Build a compact jazz/pop vocabulary.

    The prototype intentionally keeps the vocabulary small to avoid pretending
    that it can reliably detect very fine tensions in dense mixes.
    """
    qualities = [
        ("", [0, 4, 7], {0: 1.0, 4: 0.85, 7: 0.9}),
        ("m", [0, 3, 7], {0: 1.0, 3: 0.85, 7: 0.9}),
        ("7", [0, 4, 7, 10], {0: 1.0, 4: 0.8, 7: 0.85, 10: 0.65}),
        ("maj7", [0, 4, 7, 11], {0: 1.0, 4: 0.8, 7: 0.85, 11: 0.6}),
        ("m7", [0, 3, 7, 10], {0: 1.0, 3: 0.8, 7: 0.85, 10: 0.65}),
        ("dim", [0, 3, 6], {0: 1.0, 3: 0.8, 6: 0.75}),
        ("m7b5", [0, 3, 6, 10], {0: 1.0, 3: 0.75, 6: 0.75, 10: 0.6}),
        ("sus4", [0, 5, 7], {0: 1.0, 5: 0.8, 7: 0.9}),
        ("add9", [0, 2, 4, 7], {0: 1.0, 2: 0.45, 4: 0.8, 7: 0.85}),
    ]
    templates: list[tuple[str, np.ndarray]] = []
    for root in range(12):
        for suffix, intervals, weights in qualities:
            templates.append((f"{NOTE_NAMES_SHARP[root]}{suffix}", rotate_template(intervals, root, weights)))
    return templates


def classify_chroma(chroma_vector: np.ndarray, templates: list[tuple[str, np.ndarray]]) -> tuple[str, float]:
    v = normalize_vector(chroma_vector)
    if np.sum(v) <= 1e-9:
        return "N", 0.0

    scores = []
    for symbol, template in templates:
        denom = (np.linalg.norm(v) * np.linalg.norm(template)) + 1e-9
        score = float(np.dot(v, template) / denom)
        scores.append((symbol, score))
    scores.sort(key=lambda item: item[1], reverse=True)
    best_symbol, best_score = scores[0]
    second_score = scores[1][1] if len(scores) > 1 else 0.0
    confidence = max(0.0, min(1.0, best_score - (0.45 * second_score)))
    if best_score < 0.48 or confidence < 0.12:
        return "?", confidence
    return best_symbol, confidence


def median_filter_symbols(symbols: list[str], width: int = 7) -> list[str]:
    if width <= 1 or not symbols:
        return symbols
    half = width // 2
    out = []
    for i in range(len(symbols)):
        lo = max(0, i - half)
        hi = min(len(symbols), i + half + 1)
        window = symbols[lo:hi]
        out.append(max(set(window), key=window.count))
    return out


def segments_from_frames(
    times: np.ndarray,
    symbols: list[str],
    confidences: list[float],
    duration: float,
    min_segment_seconds: float,
) -> list[ChordSegment]:
    if len(symbols) == 0:
        return []

    raw: list[ChordSegment] = []
    start = float(times[0])
    current = symbols[0]
    confs = [confidences[0]]

    for i in range(1, len(symbols)):
        if symbols[i] != current:
            end = float(times[i])
            raw.append(ChordSegment(start=start, end=end, symbol=current, confidence=float(np.mean(confs))))
            start = end
            current = symbols[i]
            confs = [confidences[i]]
        else:
            confs.append(confidences[i])
    raw.append(ChordSegment(start=start, end=duration, symbol=current, confidence=float(np.mean(confs))))

    if not raw:
        return []

    merged: list[ChordSegment] = []
    for seg in raw:
        if (
            merged
            and (seg.end - seg.start) < min_segment_seconds
            and seg.symbol not in {"?", "N"}
        ):
            prev = merged[-1]
            prev.end = seg.end
            prev.confidence = float((prev.confidence + seg.confidence) / 2)
        else:
            merged.append(seg)
    return merged


def guess_key(chroma_mean: np.ndarray) -> str:
    """Very rough key estimate using major/minor Krumhansl-style profiles."""
    major_profile = normalize_vector(np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]))
    minor_profile = normalize_vector(np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]))
    v = normalize_vector(chroma_mean)
    best = ("Unknown", -1.0)
    for root in range(12):
        maj = np.roll(major_profile, root)
        minor = np.roll(minor_profile, root)
        maj_score = float(np.dot(v, maj) / ((np.linalg.norm(v) * np.linalg.norm(maj)) + 1e-9))
        min_score = float(np.dot(v, minor) / ((np.linalg.norm(v) * np.linalg.norm(minor)) + 1e-9))
        if maj_score > best[1]:
            best = (f"{NOTE_NAMES_SHARP[root]} major", maj_score)
        if min_score > best[1]:
            best = (f"{NOTE_NAMES_SHARP[root]} minor", min_score)
    return best[0]


def parse_chord_root_and_quality(symbol: str) -> tuple[int | None, str]:
    if symbol in {"?", "N", ""}:
        return None, "unknown"
    root_text = symbol[:2] if len(symbol) >= 2 and symbol[1] in {"#", "b"} else symbol[:1]
    if root_text not in NOTE_NAMES_SHARP:
        return None, "unknown"
    root = NOTE_NAMES_SHARP.index(root_text)
    suffix = symbol[len(root_text):]
    if suffix.startswith("m") and not suffix.startswith("maj"):
        quality = "minor"
    elif suffix.startswith("dim") or suffix.startswith("m7b5"):
        quality = "diminished"
    elif suffix.startswith("7"):
        quality = "dominant"
    else:
        quality = "major"
    return root, quality


def guess_key_from_chords(segments: list[ChordSegment], chroma_guess: str) -> str:
    """Estimate key from detected chord durations.

    This helps with the common relative major/minor confusion: for example,
    E minor and G major can have very similar global pitch-class content.
    """
    if not segments:
        return chroma_guess

    major_degrees = {
        0: {"major", "dominant"},
        2: {"minor"},
        4: {"minor"},
        5: {"major"},
        7: {"major", "dominant"},
        9: {"minor"},
        11: {"diminished"},
    }
    minor_degrees = {
        0: {"minor"},
        2: {"diminished", "minor"},
        3: {"major"},
        5: {"minor"},
        7: {"minor", "major", "dominant"},
        8: {"major"},
        10: {"major", "dominant"},
        11: {"diminished", "dominant"},
    }

    parsed: list[tuple[int, str, float, float]] = []
    for index, seg in enumerate(segments):
        root, quality = parse_chord_root_and_quality(seg.symbol)
        if root is None:
            continue
        duration = max(0.0, seg.end - seg.start)
        # First and last harmonies often carry tonic information in pop songs.
        edge_bonus = 1.35 if index == 0 or index == len(segments) - 1 else 1.0
        parsed.append((root, quality, duration * edge_bonus, seg.confidence))

    if not parsed:
        return chroma_guess

    best_key = chroma_guess
    best_score = -1.0
    for tonic in range(12):
        for mode, degrees in [("major", major_degrees), ("minor", minor_degrees)]:
            score = 0.0
            for root, quality, duration, confidence in parsed:
                degree = (root - tonic) % 12
                weight = duration * max(0.25, confidence)
                allowed_qualities = degrees.get(degree)
                if allowed_qualities is None:
                    score -= 0.25 * weight
                elif quality in allowed_qualities:
                    score += 1.0 * weight
                else:
                    score += 0.35 * weight
                if degree == 0:
                    score += 0.55 * weight
                    if mode == "minor" and quality == "minor":
                        score += 0.65 * weight
                    if mode == "major" and quality in {"major", "dominant"}:
                        score += 0.65 * weight
            if score > best_score:
                best_score = score
                best_key = f"{NOTE_NAMES_SHARP[tonic]} {mode}"

    return best_key


def parse_forced_key(key_label: str | None) -> tuple[int | None, str | None]:
    """Parse labels such as 'E minor', 'Em', 'Mi mineur', 'mi min'."""
    if not key_label:
        return None, None

    raw = key_label.strip()
    lower = raw.lower().replace("♭", "b").replace("♯", "#")
    mode = None
    if any(token in lower for token in ["minor", "mineur", " min", "moll"]):
        mode = "minor"
    elif any(token in lower for token in ["major", "majeur", " maj", "dur"]):
        mode = "major"
    elif len(raw) >= 2 and raw[1].lower() == "m":
        mode = "minor"

    first_token = lower.split()[0] if lower.split() else lower
    first_token = first_token.rstrip(":")

    # English note names.
    english_candidate = raw.split()[0].replace("♭", "b").replace("♯", "#").rstrip(":")
    if english_candidate.endswith("m") and len(english_candidate) <= 3:
        english_candidate = english_candidate[:-1]
    english_candidate = english_candidate[:1].upper() + english_candidate[1:]
    if english_candidate in NOTE_NAMES_SHARP:
        return NOTE_NAMES_SHARP.index(english_candidate), mode

    # French note names.
    french_candidate = FRENCH_NOTE_TO_ENGLISH.get(first_token)
    if french_candidate in NOTE_NAMES_SHARP:
        return NOTE_NAMES_SHARP.index(french_candidate), mode

    return None, mode


def minor_tonic_symbol(symbol: str) -> str:
    root, quality = parse_chord_root_and_quality(symbol)
    if root is None:
        return symbol
    root_name = NOTE_NAMES_SHARP[root]
    if quality == "major":
        if symbol.endswith("maj7"):
            return f"{root_name}m7"
        if symbol.endswith("add9"):
            return f"{root_name}m"
        return f"{root_name}m"
    if quality == "dominant":
        return f"{root_name}m7"
    return symbol


def apply_forced_key_corrections(segments: list[ChordSegment], force_key: str | None) -> list[ChordSegment]:
    """Use forced minor key as musical context, not only display text.

    In this MVP, the most common failure is tonic-major instead of tonic-minor.
    When the user explicitly forces a minor key, we correct tonic-root major
    detections into minor labels. This is intentionally conservative: it only
    touches chords whose root is the forced tonic.
    """
    tonic, mode = parse_forced_key(force_key)
    if tonic is None or mode != "minor":
        return segments

    corrected: list[ChordSegment] = []
    for seg in segments:
        root, quality = parse_chord_root_and_quality(seg.symbol)
        new_symbol = seg.symbol
        if root == tonic and quality in {"major", "dominant"}:
            new_symbol = minor_tonic_symbol(seg.symbol)
        corrected.append(ChordSegment(start=seg.start, end=seg.end, symbol=new_symbol, confidence=seg.confidence))
    return corrected


def segment_at_time(segments: list[ChordSegment], t: float) -> ChordSegment | None:
    for seg in segments:
        if seg.start <= t < seg.end:
            return seg
    return segments[-1] if segments else None


def build_grid(
    segments: list[ChordSegment],
    beat_times: np.ndarray,
    duration: float,
    beats_per_bar: int,
) -> list[GridMeasure]:
    if beat_times.size < beats_per_bar + 1:
        # Fallback to fixed beat grid if beat tracking failed.
        beat_times = np.linspace(0, duration, max(8, math.ceil(duration * 2)))

    grid: list[GridMeasure] = []
    measure_number = 1
    for i in range(0, len(beat_times) - beats_per_bar, beats_per_bar):
        start = float(beat_times[i])
        end = float(beat_times[i + beats_per_bar]) if i + beats_per_bar < len(beat_times) else duration
        if end <= start:
            continue

        beat_symbols: list[tuple[int, str, float]] = []
        for beat_offset in range(beats_per_bar):
            beat_index = i + beat_offset
            beat_time = float(beat_times[beat_index])
            if beat_index + 1 < len(beat_times):
                next_beat_time = float(beat_times[beat_index + 1])
                probe_time = beat_time + ((next_beat_time - beat_time) * 0.5)
            else:
                probe_time = beat_time
            seg = segment_at_time(segments, probe_time)
            if seg is None:
                symbol, conf = "?", 0.0
            else:
                symbol, conf = seg.symbol, seg.confidence
            beat_symbols.append((beat_offset + 1, symbol, conf))

        blocks: list[ChordBlock] = []
        block_symbol = beat_symbols[0][1]
        block_start = beat_symbols[0][0]
        block_confs = [beat_symbols[0][2]]
        for beat_start, symbol, conf in beat_symbols[1:]:
            if symbol != block_symbol:
                blocks.append(
                    ChordBlock(
                        symbol=block_symbol,
                        beat_start=block_start,
                        beat_duration=beat_start - block_start,
                        confidence=float(np.mean(block_confs)),
                    )
                )
                block_symbol = symbol
                block_start = beat_start
                block_confs = [conf]
            else:
                block_confs.append(conf)
        blocks.append(
            ChordBlock(
                symbol=block_symbol,
                beat_start=block_start,
                beat_duration=(beats_per_bar + 1) - block_start,
                confidence=float(np.mean(block_confs)),
            )
        )

        grid.append(GridMeasure(measure=measure_number, start=start, end=end, chords=blocks))
        measure_number += 1

    return grid


def format_measure(measure: GridMeasure, previous_single_symbol: str | None) -> tuple[str, str | None]:
    symbols = [block.symbol for block in measure.chords if block.symbol not in {"N"}]
    if not symbols:
        text = "?"
        return f"| {text:<10} ", text

    if len(symbols) == 1:
        symbol = symbols[0]
        if previous_single_symbol == symbol:
            return f"| {'%':<10} ", symbol
        return f"| {symbol:<10} ", symbol

    joined = " ".join(symbols)
    return f"| {joined:<10} ", None


def write_markdown(result: AnalysisResult, out_path: Path) -> None:
    lines = [
        "# Grille harmonique",
        "",
        f"- Fichier : `{Path(result.source_file).name}`",
        f"- Durée : {result.duration:.1f} s",
        f"- Tempo estimé : {result.tempo:.1f} BPM",
        f"- Mesure : {result.meter}",
        f"- Tonalité probable : {result.key_guess}",
        "",
        "## Grille",
        "",
    ]

    previous_single: str | None = None
    row: list[str] = []
    for measure in result.grid:
        cell, previous_single = format_measure(measure, previous_single)
        row.append(cell)
        if len(row) == 4:
            lines.append("".join(row) + "|")
            row = []
    if row:
        lines.append("".join(row) + "|")

    lines += [
        "",
        "## Segments détectés",
        "",
        "| Début | Fin | Accord | Confiance |",
        "|---:|---:|---|---:|",
    ]
    for seg in result.chord_segments:
        lines.append(f"| {seg.start:.2f} | {seg.end:.2f} | {seg.symbol} | {seg.confidence:.2f} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_audio(
    input_path: Path,
    sample_rate: int,
    hop_length: int,
    beats_per_bar: int,
    min_segment_seconds: float,
    fallback_bpm: float,
    force_tempo: float | None,
    force_key: str | None,
) -> AnalysisResult:
    require_dependencies()
    import librosa

    wav_path = convert_to_wav(input_path, sample_rate)
    try:
        y, sr = librosa.load(str(wav_path), sr=sample_rate, mono=True)
    finally:
        wav_path.unlink(missing_ok=True)

    if y.size == 0:
        raise SystemExit("The decoded audio file is empty.")

    duration = float(librosa.get_duration(y=y, sr=sr))
    y_harmonic, _ = librosa.effects.hpss(y)

    tempo_raw, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, trim=False)
    tempo = float(np.asarray(tempo_raw).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    if force_tempo is not None and force_tempo > 0:
        tempo = float(force_tempo)
        beat_step = 60.0 / tempo
        beat_times = np.arange(0.0, duration + beat_step, beat_step)
    elif tempo <= 1.0 or beat_times.size < beats_per_bar + 1:
        tempo = float(fallback_bpm)
        beat_step = 60.0 / max(1.0, fallback_bpm)
        beat_times = np.arange(0.0, duration + beat_step, beat_step)
    elif beat_times[0] > 0.5:
        beat_times = np.insert(beat_times, 0, 0.0)

    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=hop_length)
    chroma = librosa.decompose.nn_filter(chroma, aggregate=np.median, metric="cosine")
    chroma = np.maximum(chroma, 0)
    frame_times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    templates = chord_templates()
    frame_symbols: list[str] = []
    frame_confidences: list[float] = []
    for i in range(chroma.shape[1]):
        symbol, confidence = classify_chroma(chroma[:, i], templates)
        frame_symbols.append(symbol)
        frame_confidences.append(confidence)

    frame_symbols = median_filter_symbols(frame_symbols, width=9)
    chord_segments = segments_from_frames(
        frame_times,
        frame_symbols,
        frame_confidences,
        duration=duration,
        min_segment_seconds=min_segment_seconds,
    )
    chord_segments = apply_forced_key_corrections(chord_segments, force_key)
    chroma_key_guess = guess_key(np.mean(chroma, axis=1))
    key_guess = force_key if force_key else guess_key_from_chords(chord_segments, chroma_key_guess)
    grid = build_grid(chord_segments, beat_times, duration, beats_per_bar)

    return AnalysisResult(
        source_file=str(input_path),
        duration=duration,
        tempo=tempo,
        meter=f"{beats_per_bar}/4",
        key_guess=key_guess,
        chord_segments=chord_segments,
        grid=grid,
    )


def result_to_json_dict(result: AnalysisResult) -> dict:
    return {
        "source_file": result.source_file,
        "duration": result.duration,
        "analysis": {
            "tempo": result.tempo,
            "meter": result.meter,
            "key_guess": result.key_guess,
        },
        "chord_segments": [asdict(seg) for seg in result.chord_segments],
        "grid": [
            {
                "measure": measure.measure,
                "start": measure.start,
                "end": measure.end,
                "chords": [asdict(block) for block in measure.chords],
            }
            for measure in result.grid
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a local audio file and export a simple jazz/pop chord grid.")
    parser.add_argument("audio_file", type=Path, help="Path to a local audio file.")
    parser.add_argument("--out", type=Path, default=Path("grid.md"), help="Markdown grid output path.")
    parser.add_argument("--json", type=Path, default=Path("analysis.json"), help="JSON analysis output path.")
    parser.add_argument("--sample-rate", type=int, default=22050, help="Analysis sample rate.")
    parser.add_argument("--hop-length", type=int, default=2048, help="Feature hop length.")
    parser.add_argument("--beats-per-bar", type=int, default=4, help="Assumed meter numerator. Default: 4.")
    parser.add_argument("--min-segment", type=float, default=0.45, help="Merge chord segments shorter than this many seconds.")
    parser.add_argument("--fallback-bpm", type=float, default=120.0, help="BPM used if beat tracking fails.")
    parser.add_argument("--tempo", type=float, default=None, help="Override detected BPM, e.g. --tempo 80.5.")
    parser.add_argument("--key", type=str, default=None, help='Override key label, e.g. "E minor" or "Mi mineur".')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.audio_file.exists():
        raise SystemExit(f"Audio file not found: {args.audio_file}")

    result = analyze_audio(
        input_path=args.audio_file,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        beats_per_bar=args.beats_per_bar,
        min_segment_seconds=args.min_segment,
        fallback_bpm=args.fallback_bpm,
        force_tempo=args.tempo,
        force_key=args.key,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    write_markdown(result, args.out)
    args.json.write_text(json.dumps(result_to_json_dict(result), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote Markdown grid: {args.out}")
    print(f"Wrote JSON analysis: {args.json}")
    print(f"Tempo: {result.tempo:.1f} BPM | Meter: {result.meter} | Key guess: {result.key_guess}")


if __name__ == "__main__":
    main()
