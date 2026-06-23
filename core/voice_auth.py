# core/voice_auth.py

from __future__ import annotations

import math
import shutil
import threading
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


VOICE_VERIFICATION_SAMPLE_RATE = 16000
VOICE_VERIFICATION_THRESHOLD = 0.25
VOICE_ACTIVITY_PADDING_SECONDS = 0.25
VOICE_MINIMUM_SPEECH_SECONDS = 1.0

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

    write(
        str(audio_path),
        SAMPLE_RATE,
        audio_data,
    )

    return audio_path


def record_voice_profile_sample(sample_number: int) -> Path:
    return _record_wav(
        duration_seconds=VOICE_AUTH_ENROLLMENT_SECONDS,
        prefix=f"voice_profile_sample_{sample_number}",
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


def _prediction_is_match(prediction) -> bool:
    try:
        value = prediction

        if hasattr(value, "detach"):
            value = value.detach().cpu()

        if hasattr(value, "reshape"):
            value = value.reshape(-1)[0]

        if hasattr(value, "item"):
            value = value.item()

        return float(value) >= 0.5

    except (TypeError, ValueError, IndexError):
        return False


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


def _trim_silence(audio, sample_rate: int):
    import numpy as np

    if audio.size == 0:
        raise ValueError("Audio file is empty.")

    absolute_audio = np.abs(audio)
    peak = float(absolute_audio.max())

    if peak < 0.003:
        raise ValueError(
            "No usable speech was detected. Check your microphone input."
        )

    smoothing_window = max(
        int(sample_rate * 0.02),
        1,
    )

    smoothing_kernel = (
        np.ones(smoothing_window, dtype=np.float32) / smoothing_window
    )

    smoothed_volume = np.convolve(
        absolute_audio,
        smoothing_kernel,
        mode="same",
    )

    noise_floor = float(np.percentile(smoothed_volume, 20))

    activity_threshold = max(
        0.006,
        min(
            peak * 0.20,
            max(noise_floor * 3.0, 0.008),
        ),
    )

    active_indices = np.flatnonzero(
        smoothed_volume >= activity_threshold
    )

    if active_indices.size == 0:
        raise ValueError(
            "No clear speech was detected. Speak closer to the microphone."
        )

    padding = int(
        sample_rate * VOICE_ACTIVITY_PADDING_SECONDS
    )

    start_index = max(
        int(active_indices[0]) - padding,
        0,
    )

    end_index = min(
        int(active_indices[-1]) + padding + 1,
        len(audio),
    )

    trimmed_audio = audio[start_index:end_index]

    minimum_samples = int(
        sample_rate * VOICE_MINIMUM_SPEECH_SECONDS
    )

    if len(trimmed_audio) < minimum_samples:
        raise ValueError(
            "Too little speech was detected. Speak for at least two seconds."
        )

    return trimmed_audio


def _load_wav_for_voice_verification(audio_path: str | Path):
    import numpy as np
    import torch
    from scipy.io import wavfile
    from scipy.signal import resample_poly

    path = Path(audio_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file was not found: {path}")

    sample_rate, audio_data = wavfile.read(str(path))

    if sample_rate <= 0:
        raise ValueError(f"Invalid sample rate in audio file: {path}")

    audio = _wav_to_float32(audio_data)

    if sample_rate != VOICE_VERIFICATION_SAMPLE_RATE:
        divisor = math.gcd(
            int(sample_rate),
            VOICE_VERIFICATION_SAMPLE_RATE,
        )

        audio = resample_poly(
            audio,
            VOICE_VERIFICATION_SAMPLE_RATE // divisor,
            int(sample_rate) // divisor,
        ).astype(np.float32)

        sample_rate = VOICE_VERIFICATION_SAMPLE_RATE

    audio = _trim_silence(
        audio,
        sample_rate,
    )

    return torch.from_numpy(
        np.ascontiguousarray(audio),
    ).unsqueeze(0)


def _compare_voice_files(
    verifier,
    profile_file: Path,
    candidate_waveform,
):
    import torch

    profile_waveform = _load_wav_for_voice_verification(
        profile_file
    )

    with torch.inference_mode():
        return verifier.verify_batch(
            profile_waveform,
            candidate_waveform,
            threshold=VOICE_VERIFICATION_THRESHOLD,
        )


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
        verifier = _get_voice_verifier()

        candidate_waveform = _load_wav_for_voice_verification(
            candidate_path
        )

    except Exception as error:
        return (
            False,
            0.0,
            f"Voice-verification setup could not start: {error}",
        )

    matches = 0
    scores: list[float] = []
    successful_comparisons = 0

    for profile_file in profile_files:
        try:
            score, prediction = _compare_voice_files(
                verifier=verifier,
                profile_file=profile_file,
                candidate_waveform=candidate_waveform,
            )

            score_value = _score_to_float(score)
            is_match = _prediction_is_match(prediction)

            print(
                f"[Voice comparison: {profile_file.name} | "
                f"score={score_value:.3f} | match={is_match}]"
            )

            scores.append(score_value)
            successful_comparisons += 1

            if is_match:
                matches += 1

        except Exception as error:
            print(
                "[Voice comparison skipped: "
                f"{type(error).__name__}: {error}]"
            )

    if successful_comparisons < VOICE_AUTH_REQUIRED_MATCHES:
        return (
            False,
            0.0,
            "Not enough valid voice comparisons were available.",
        )

    average_score = sum(scores) / len(scores) if scores else 0.0

    if matches >= VOICE_AUTH_REQUIRED_MATCHES:
        return (
            True,
            average_score,
            f"Voice verified with {matches} matching profile samples.",
        )

    return (
        False,
        average_score,
        f"Voice verification failed. Only {matches} profile samples matched.",
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

            shutil.copy2(
                sample_path,
                destination,
            )

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