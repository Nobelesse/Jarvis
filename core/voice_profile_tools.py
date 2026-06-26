# core/voice_profile_tools.py

from __future__ import annotations

import math
import shutil
import wave
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    VOICE_AUTH_PROFILE_DIR,
    VOICE_AUTH_REQUIRED_MATCHES,
)
from core.voice_auth import (
    VOICE_VERIFICATION_THRESHOLD,
    _compare_voice_files,
    _get_voice_verifier,
    _load_wav_for_voice_verification,
    _score_to_float,
)


VOICE_AUTH_MATCH_THRESHOLD = VOICE_VERIFICATION_THRESHOLD
MIN_RMS = 0.006
MIN_ACTIVE_RATIO = 0.35
MAX_PEAK = 0.995


def get_profile_files() -> list[Path]:
    VOICE_AUTH_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    return sorted(
        file_path
        for file_path in VOICE_AUTH_PROFILE_DIR.glob("profile_*.wav")
        if file_path.is_file()
    )


def write_wav(audio: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    mono_audio = np.clip(mono_audio, -1.0, 1.0)

    pcm_audio = (mono_audio * 32767.0).astype(np.int16)

    with wave.open(str(destination), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm_audio.tobytes())


def record_wav(destination: Path, seconds: float) -> np.ndarray:
    frame_count = max(1, int(SAMPLE_RATE * seconds))

    print(f"[Recording for {seconds:.1f} seconds...]")

    audio = sd.rec(
        frame_count,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    write_wav(mono_audio, destination)

    return mono_audio


def analyse_audio(audio: np.ndarray) -> dict[str, float]:
    mono_audio = np.asarray(audio, dtype=np.float32).reshape(-1)

    if mono_audio.size == 0:
        return {
            "duration_seconds": 0.0,
            "rms": 0.0,
            "peak": 0.0,
            "active_ratio": 0.0,
        }

    rms = float(math.sqrt(float(np.mean(np.square(mono_audio)))))
    peak = float(np.max(np.abs(mono_audio)))

    frame_size = max(1, int(SAMPLE_RATE * 0.25))
    frame_count = mono_audio.size // frame_size
    usable_audio = mono_audio[: frame_count * frame_size]

    if frame_count == 0:
        active_ratio = 0.0
    else:
        frames = usable_audio.reshape(frame_count, frame_size)
        frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))

        activity_floor = max(0.008, rms * 0.60)

        active_ratio = float(
            np.mean(frame_rms >= activity_floor)
        )

    return {
        "duration_seconds": float(mono_audio.size / SAMPLE_RATE),
        "rms": rms,
        "peak": peak,
        "active_ratio": active_ratio,
    }


def quality_issues(metrics: dict[str, float]) -> list[str]:
    issues: list[str] = []

    if metrics["rms"] < MIN_RMS:
        issues.append("audio is too quiet")

    if metrics["active_ratio"] < MIN_ACTIVE_RATIO:
        issues.append("too much silence was detected")

    if metrics["peak"] >= MAX_PEAK:
        issues.append("microphone clipping was detected")

    return issues


def archive_existing_profiles() -> Path | None:
    profiles = get_profile_files()

    if not profiles:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    archive_directory = (
        VOICE_AUTH_PROFILE_DIR.parent
        / "voice_profile_archive"
        / timestamp
    )

    archive_directory.mkdir(parents=True, exist_ok=True)

    for profile in profiles:
        shutil.move(
            str(profile),
            str(archive_directory / profile.name),
        )

    return archive_directory


def similarity_score(
    verifier: Any,
    first_file: Path,
    second_file: Path,
) -> float:
    candidate_waveform = _load_wav_for_voice_verification(
        second_file
    )

    score, _prediction = _compare_voice_files(
        verifier=verifier,
        profile_file=first_file,
        candidate_waveform=candidate_waveform,
    )

    return _score_to_float(score)


def profile_health_report() -> dict[str, Any]:
    profiles = get_profile_files()

    report: dict[str, Any] = {
        "profile_count": len(profiles),
        "profiles": profiles,
        "pair_scores": [],
        "minimum_score": None,
        "average_score": None,
        "maximum_score": None,
        "matching_pairs": 0,
        "ready": False,
    }

    if len(profiles) < 2:
        return report

    verifier = _get_voice_verifier()

    for first_file, second_file in combinations(profiles, 2):
        score = similarity_score(
            verifier=verifier,
            first_file=first_file,
            second_file=second_file,
        )

        report["pair_scores"].append(
            (first_file, second_file, score)
        )

    scores = [item[2] for item in report["pair_scores"]]

    report["minimum_score"] = min(scores)
    report["average_score"] = sum(scores) / len(scores)
    report["maximum_score"] = max(scores)

    report["matching_pairs"] = sum(
        score >= VOICE_AUTH_MATCH_THRESHOLD
        for score in scores
    )

    report["ready"] = (
        report["matching_pairs"] >= VOICE_AUTH_REQUIRED_MATCHES
    )

    return report


def candidate_test_report(candidate_file: Path) -> dict[str, Any]:
    profiles = get_profile_files()

    report: dict[str, Any] = {
        "profile_count": len(profiles),
        "scores": [],
        "matches": 0,
        "best_score": None,
        "accepted": False,
    }

    if not profiles:
        return report

    verifier = _get_voice_verifier()

    candidate_waveform = _load_wav_for_voice_verification(
        candidate_file
    )

    for profile in profiles:
        score, _prediction = _compare_voice_files(
            verifier=verifier,
            profile_file=profile,
            candidate_waveform=candidate_waveform,
        )

        score_value = _score_to_float(score)

        report["scores"].append(
            (profile, score_value)
        )

    scores = [item[1] for item in report["scores"]]

    report["matches"] = sum(
        score >= VOICE_AUTH_MATCH_THRESHOLD
        for score in scores
    )

    report["best_score"] = max(scores)

    report["accepted"] = (
        report["matches"] >= VOICE_AUTH_REQUIRED_MATCHES
    )

    return report