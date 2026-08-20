#!/usr/bin/env python3
"""Generate a tiny synthetic chord progression for smoke testing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


NOTE_TO_MIDI = {
    "C": 60,
    "C#": 61,
    "D": 62,
    "Eb": 63,
    "E": 64,
    "F": 65,
    "F#": 66,
    "G": 67,
    "Ab": 68,
    "A": 69,
    "Bb": 70,
    "B": 71,
}


CHORDS = {
    "C": ["C", "E", "G"],
    "Am": ["A", "C", "E"],
    "F": ["F", "A", "C"],
    "G7": ["G", "B", "D", "F"],
}


def midi_to_hz(midi: int) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def synth_chord(notes: list[str], seconds: float, sr: int) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = np.zeros_like(t)
    for note in notes:
        midi = NOTE_TO_MIDI[note]
        for octave_shift, amp in [(-12, 0.35), (0, 0.45), (12, 0.2)]:
            freq = midi_to_hz(midi + octave_shift)
            audio += amp * np.sin(2 * np.pi * freq * t)
    # Light amplitude envelope to avoid clicks.
    attack = max(1, int(0.02 * sr))
    release = max(1, int(0.04 * sr))
    env = np.ones_like(audio)
    env[:attack] = np.linspace(0, 1, attack)
    env[-release:] = np.linspace(1, 0, release)
    return audio * env


def main() -> None:
    sr = 22050
    progression = ["C", "Am", "F", "G7"] * 2
    audio = np.concatenate([synth_chord(CHORDS[chord], 2.0, sr) for chord in progression])
    audio = audio / max(1e-9, np.max(np.abs(audio)))
    out = Path("test_progression.wav")
    wavfile.write(out, sr, (audio * 0.8 * 32767).astype(np.int16))
    print(out)


if __name__ == "__main__":
    main()
