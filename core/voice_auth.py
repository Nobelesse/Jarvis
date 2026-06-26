# core/voice_auth.py

from __future__ import annotations

import shutil
import statistics
import threading
from itertools import combinations
from pathlib import Path
from uuid import uuid4

from config import (
    AUDIO_DIR,
    MODELS_DIR,
    SAMPLE_RATE,
    VOICE_AUTH_ENROLLMENT_SECONDS,
    VOICE_AUTH_MODEL_DIR,
    VOICE_AUTH_MODEL_SOURCE,
    VOICE_AUTH_PROFILE_DIR,
    VOICE_AUTH_PROFILE_SAMPLES,
    VOICE_AUTH_REQUIRED_MATCHES,
    WHISPER_MODEL,
)


# SpeechBrain's default decision threshold for speaker verification.
VOICE_VERIFICATION_THRESHOLD = 0.25

# These validate recording quality. They do not lower security.
VOICE_ENROLLMENT_MINIMUM_SPEECH_SECONDS = 1.80
VOICE_COMMAND_MINIMUM_SPEECH_SECONDS = 0.70
VOICE_MINIMUM_RMS = 0.004
VOICE_MINIMUM_PEAK = 0.015
VOICE_PROFILE_MINIMUM_MEDIAN_SCORE = 0.35
VOICE_PROFILE_MINIMUM_MATCH_RATIO = 0.70

_voice_verifier: object | None = None
_voice_verifier_lock = threading.Lock()

_whisper_model: object | None = None
_whisper_model_lock = threading.Lock()


def get_voice_profile_files() -> list[Path]:
    if not VOICE_AUTH_PROFILE_DIR.exists():
        return []

    return sorted(
        file_path
        for file_path in VOICE_AUTH_PROFILE_DIR.glob("profile_*.wav")
        if file_path.is_file()
    )


def has_voice_profile() -> bool:
    return len(get_voice_profile_files()) >= VOICE_AUTH_PROFILE_SAMPLES


def _record_wav(duration_seconds: float, prefix: str) -> Path:
    import numpy as np
    import sounddevice as sd
    from scipy.io.wavfile import write

    if duration_seconds <= 0:
        raise ValueError("Recording duration must be greater than zero.")

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[Listening for {duration_seconds:.1f} seconds...]")

    recording = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    audio_data = np.clip(recording, -1.0, 1.0)
    audio_data = (audio_data * 32767).astype(np.int16)

    audio_path = AUDIO_DIR / f"{prefix}_{uuid4().hex}.wav"
    write(str(audio_path), SAMPLE_RATE, audio_data)

    return audio_path


def record_voice_profile_sample(sample_number: int) -> Path:
    return _record_wav(
        duration_seconds=VOICE_AUTH_ENROLLMENT_SECONDS,
        prefix=f"voice_profile_sample_{sample_number}",
    )


def record_voice_verification_sample(
    duration_seconds: float | None = None,
) -> Path:
    duration = duration_seconds or VOICE_AUTH_ENROLLMENT_SECONDS

    return _record_wav(
        duration_seconds=duration,
        prefix="voice_auth_test",
    )


def _get_voice_verifier():
    global _voice_verifier

    if _voice_verifier is not None:
        return _voice_verifier

    with _voice_verifier_lock:
        if _voice_verifier is None:
            from speechbrain.inference.speaker import SpeakerRecognition
            from speechbrain.utils.fetching import LocalStrategy

            print("[Loading offline voice-verification model...]")
            VOICE_AUTH_MODEL_DIR.mkdir(parents=True, exist_ok=True)

            _voice_verifier = SpeakerRecognition.from_hparams(
                source=VOICE_AUTH_MODEL_SOURCE,
                savedir=str(VOICE_AUTH_MODEL_DIR),
                run_opts={"device": "cpu"},
                local_strategy=LocalStrategy.COPY,
            )

    return _voice_verifier


def warm_up_voice_verifier() -> None:
    _get_voice_verifier()


def _get_whisper_model():
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    with _whisper_model_lock:
        if _whisper_model is None:
            from faster_whisper import WhisperModel

            print("[Loading local speech-recognition model...]")

            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                download_root=str(MODELS_DIR),
            )

    return _whisper_model


def _score_to_float(score) -> float:
    try:
        value = score

        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "reshape"):
            value = value.reshape(-1)[0]

        if hasattr(value, "item"):
            value = value.item()

        return float(value)

    except (TypeError, ValueError, IndexError):
        return 0.0


def _wav_to_float32(audio_data):
    import numpy as np

    audio = np.asarray(audio_data)

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    elif audio.ndim != 1:
        audio = audio.reshape(-1)

    if audio.dtype.kind == "u":
        info = np.iinfo(audio.dtype)
        midpoint = (float(info.max) + 1.0) / 2.0
        audio = (audio.astype(np.float32) - midpoint) / midpoint

    elif audio.dtype.kind == "i":
        info = np.iinfo(audio.dtype)
        divisor = float(max(abs(int(info.min)), abs(int(info.max))))
        audio = audio.astype(np.float32) / divisor

    else:
        audio = audio.astype(np.float32)

    audio = np.nan_to_num(
        audio,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return np.clip(audio, -1.0, 1.0)


def inspect_voice_recording(audio_path: str | Path) -> dict[str, float]:
    import numpy as np
    from scipy.io import wavfile

    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    sample_rate, audio_data = wavfile.read(str(path))

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in audio file: {path}")

    audio = _wav_to_float32(audio_data)

    absolute_audio = np.abs(audio)
    peak = float(absolute_audio.max())
    rms = float(np.sqrt(np.mean(np.square(audio))))

    if peak <= 0.0:
        activity_threshold = 0.0
        active_ratio = 0.0
        active_seconds = 0.0
    else:
        noise_floor = float(np.percentile(absolute_audio, 20))

        activity_threshold = max(
            0.006,
            min(
                0.050,
                max(peak * 0.08, noise_floor * 3.0),
            ),
        )

        active_ratio = float(
            np.mean(absolute_audio >= activity_threshold)
        )

        active_seconds = float(
            audio.size * active_ratio / float(sample_rate)
        )

    return {
        "duration_seconds": round(
            audio.size / float(sample_rate),
            3,
        ),
        "sample_rate": float(sample_rate),
        "rms": rms,
        "peak": peak,
        "activity_threshold": activity_threshold,
        "active_ratio": active_ratio,
        "active_seconds": active_seconds,
    }


def get_voice_quality_issues(
    audio_info: dict[str, float],
    *,
    minimum_speech_seconds: float,
) -> list[str]:
    issues: list[str] = []

    if audio_info["peak"] < VOICE_MINIMUM_PEAK:
        issues.append(
            "Microphone input is too quiet. Speak closer to the microphone."
        )

    if audio_info["rms"] < VOICE_MINIMUM_RMS:
        issues.append(
            "The recording has very little voice energy."
        )

    if audio_info["active_seconds"] < minimum_speech_seconds:
        issues.append(
            f"Speak continuously for at least "
            f"{minimum_speech_seconds:.1f} seconds."
        )

    return issues


def format_voice_audio_quality(audio_info: dict[str, float]) -> str:
    return (
        "Audio quality: "
        f"duration={audio_info['duration_seconds']:.1f}s, "
        f"rms={audio_info['rms']:.3f}, "
        f"peak={audio_info['peak']:.3f}, "
        f"active={audio_info['active_ratio'] * 100:.0f}%"
    )


def _compare_voice_files(
    verifier,
    first_file: Path,
    second_file: Path,
) -> float:
    score, _prediction = verifier.verify_files(
        str(first_file),
        str(second_file),
    )

    return _score_to_float(score)


def get_voice_profile_diagnostics(
    sample_paths: list[str | Path] | None = None,
) -> dict[str, object]:
    if sample_paths is None:
        samples = get_voice_profile_files()
    else:
        samples = [Path(path) for path in sample_paths]

    report: dict[str, object] = {
        "healthy": False,
        "sample_count": len(samples),
        "pairs": [],
        "audio_issues": {},
        "scores": [],
        "median_score": 0.0,
        "average_score": 0.0,
        "match_ratio": 0.0,
        "message": "",
    }

    if len(samples) < VOICE_AUTH_PROFILE_SAMPLES:
        report["message"] = (
            f"At least {VOICE_AUTH_PROFILE_SAMPLES} profile samples "
            "are required."
        )
        return report

    for sample in samples:
        if not sample.is_file():
            report["message"] = (
                f"Profile sample was not found: {sample.name}"
            )
            return report

        try:
            audio_info = inspect_voice_recording(sample)

            issues = get_voice_quality_issues(
                audio_info,
                minimum_speech_seconds=VOICE_ENROLLMENT_MINIMUM_SPEECH_SECONDS,
            )
        except Exception as error:
            report["message"] = (
                f"Could not inspect {sample.name}: {error}"
            )
            return report

        if issues:
            audio_issues = report["audio_issues"]

            if isinstance(audio_issues, dict):
                audio_issues[sample.name] = issues

    if report["audio_issues"]:
        report["message"] = (
            "One or more profile recordings have poor audio quality."
        )
        return report

    try:
        verifier = _get_voice_verifier()
    except Exception as error:
        report["message"] = (
            f"Voice-verification model could not start: {error}"
        )
        return report

    scores: list[float] = []
    pairs: list[dict[str, object]] = []

    for first_file, second_file in combinations(samples, 2):
        try:
            score = _compare_voice_files(
                verifier,
                first_file,
                second_file,
            )
        except Exception as error:
            report["message"] = (
                f"Could not compare profile samples: {error}"
            )
            return report

        is_match = score >= VOICE_VERIFICATION_THRESHOLD

        scores.append(score)

        pairs.append(
            {
                "first": first_file.name,
                "second": second_file.name,
                "score": score,
                "match": is_match,
            }
        )

    if not scores:
        report["message"] = "No profile comparisons were created."
        return report

    match_ratio = sum(
        score >= VOICE_VERIFICATION_THRESHOLD
        for score in scores
    ) / len(scores)

    median_score = float(statistics.median(scores))
    average_score = float(statistics.mean(scores))

    healthy = (
        median_score >= VOICE_PROFILE_MINIMUM_MEDIAN_SCORE
        and match_ratio >= VOICE_PROFILE_MINIMUM_MATCH_RATIO
    )

    report.update(
        {
            "healthy": healthy,
            "pairs": pairs,
            "scores": scores,
            "median_score": median_score,
            "average_score": average_score,
            "match_ratio": match_ratio,
            "message": (
                "Voice profile is calibrated and ready."
                if healthy
                else (
                    "Profile recordings do not match reliably enough. "
                    "Enroll again in a quiet room using the same microphone."
                )
            ),
        }
    )

    return report


def format_voice_profile_diagnostics(
    report: dict[str, object],
) -> str:
    lines = [
        "Voice profile diagnostics",
        f"Samples: {report.get('sample_count', 0)}",
    ]

    message = str(report.get("message", "")).strip()

    if message:
        lines.append(f"Status: {message}")

    audio_issues = report.get("audio_issues", {})

    if isinstance(audio_issues, dict):
        for file_name, issues in audio_issues.items():
            joined_issues = "; ".join(str(item) for item in issues)
            lines.append(f"{file_name}: {joined_issues}")

    pairs = report.get("pairs", [])

    if isinstance(pairs, list):
        for pair in pairs:
            if not isinstance(pair, dict):
                continue

            first = pair.get("first", "unknown")
            second = pair.get("second", "unknown")
            score = float(pair.get("score", 0.0))
            match = bool(pair.get("match", False))

            lines.append(
                f"{first} vs {second}: "
                f"score={score:.3f}, match={match}"
            )

    if report.get("scores"):
        lines.append(
            "Summary: "
            f"median={float(report.get('median_score', 0.0)):.3f}, "
            f"average={float(report.get('average_score', 0.0)):.3f}, "
            f"matches={float(report.get('match_ratio', 0.0)) * 100:.0f}%"
        )

    return "\n".join(lines)


def verify_voice_recording(audio_path: str | Path) -> tuple[bool, float, str]:
    candidate_path = Path(audio_path)

    if not candidate_path.is_file():
        return False, 0.0, "Voice recording was not found."

    profile_files = get_voice_profile_files()

    if len(profile_files) < VOICE_AUTH_PROFILE_SAMPLES:
        return (
            False,
            0.0,
            "Voice profile is not enrolled. Run: python enroll_voice.py",
        )

    try:
        candidate_info = inspect_voice_recording(candidate_path)

        candidate_issues = get_voice_quality_issues(
            candidate_info,
            minimum_speech_seconds=VOICE_COMMAND_MINIMUM_SPEECH_SECONDS,
        )
    except Exception as error:
        return (
            False,
            0.0,
            f"Could not inspect voice recording: {error}",
        )

    if candidate_issues:
        return (
            False,
            0.0,
            "Voice check needs a clearer recording. "
            + " ".join(candidate_issues),
        )

    try:
        verifier = _get_voice_verifier()
    except Exception as error:
        return (
            False,
            0.0,
            f"Voice-verification setup could not start: {error}",
        )

    scores: list[float] = []

    for profile_file in profile_files:
        try:
            score = _compare_voice_files(
                verifier,
                profile_file,
                candidate_path,
            )
        except Exception as error:
            print(
                "[Voice comparison skipped: "
                f"{type(error).__name__}: {error}]"
            )
            continue

        is_match = score >= VOICE_VERIFICATION_THRESHOLD

        print(
            f"[Voice comparison: {profile_file.name} | "
            f"score={score:.3f} | match={is_match}]"
        )

        scores.append(score)

    if len(scores) < VOICE_AUTH_REQUIRED_MATCHES:
        return (
            False,
            0.0,
            "Not enough valid voice comparisons were available.",
        )

    best_scores = sorted(scores, reverse=True)[
        :VOICE_AUTH_REQUIRED_MATCHES
    ]

    decision_score = float(statistics.mean(best_scores))

    matches = sum(
        score >= VOICE_VERIFICATION_THRESHOLD
        for score in scores
    )

    if (
        matches >= VOICE_AUTH_REQUIRED_MATCHES
        and decision_score >= VOICE_VERIFICATION_THRESHOLD
    ):
        return (
            True,
            decision_score,
            f"Voice verified with {matches} matching profile samples.",
        )

    return (
        False,
        decision_score,
        f"Voice verification failed. "
        f"Only {matches} profile samples matched.",
    )


def transcribe_verified_audio(audio_path: str | Path) -> str:
    path = Path(audio_path)

    if not path.is_file():
        return ""

    try:
        model = _get_whisper_model()

        segments, _info = model.transcribe(
            str(path),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
        )

        return " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

    except Exception as error:
        print(f"[Speech recognition error: {error}]")
        return ""


def listen_for_verified_text(
    duration_seconds: float,
) -> tuple[str, bool, float, str]:
    audio_path: Path | None = None

    try:
        audio_path = _record_wav(
            duration_seconds=duration_seconds,
            prefix="voice_auth_command",
        )

        is_verified, score, message = verify_voice_recording(audio_path)

        if not is_verified:
            return "", False, score, message

        text = transcribe_verified_audio(audio_path)

        if not text:
            return (
                "",
                True,
                score,
                "Voice verified, but no speech was recognized.",
            )

        return text, True, score, message

    except Exception as error:
        return (
            "",
            False,
            0.0,
            f"Voice authentication error: {error}",
        )

    finally:
        if audio_path and audio_path.is_file():
            try:
                audio_path.unlink()
            except OSError:
                pass


def replace_voice_profile(sample_paths: list[str | Path]) -> None:
    samples = [Path(path) for path in sample_paths]

    if len(samples) < VOICE_AUTH_PROFILE_SAMPLES:
        raise ValueError(
            f"At least {VOICE_AUTH_PROFILE_SAMPLES} samples are required."
        )

    for sample_path in samples:
        if not sample_path.is_file():
            raise FileNotFoundError(
                f"Voice sample was not found: {sample_path}"
            )

    staging_dir = VOICE_AUTH_PROFILE_DIR.parent / (
        f".voice_profile_staging_{uuid4().hex}"
    )

    staging_dir.mkdir(parents=True, exist_ok=False)

    try:
        for index, sample_path in enumerate(
            samples[:VOICE_AUTH_PROFILE_SAMPLES],
            start=1,
        ):
            destination = staging_dir / f"profile_{index:02d}.wav"

            shutil.copy2(sample_path, destination)

        if VOICE_AUTH_PROFILE_DIR.exists():
            shutil.rmtree(VOICE_AUTH_PROFILE_DIR)

        staging_dir.replace(VOICE_AUTH_PROFILE_DIR)

    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


def delete_audio_files(audio_paths: list[str | Path]) -> None:
    for audio_path in audio_paths:
        try:
            path = Path(audio_path)

            if path.is_file():
                path.unlink()

        except OSError:
            pass