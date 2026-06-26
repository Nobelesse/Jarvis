# core/voice_auth.py

from __future__ import annotations

import itertools
import json
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio
import config
from faster_whisper import WhisperModel


AUDIO_DIR = Path(config.AUDIO_DIR)
MODELS_DIR = Path(config.MODELS_DIR)
SAMPLE_RATE = int(config.SAMPLE_RATE)
WHISPER_MODEL = str(config.WHISPER_MODEL)

VOICE_AUTH_PROFILE_DIR = Path(config.VOICE_AUTH_PROFILE_DIR)
VOICE_AUTH_MODEL_DIR = Path(config.VOICE_AUTH_MODEL_DIR)
VOICE_AUTH_MODEL_SOURCE = str(config.VOICE_AUTH_MODEL_SOURCE)
VOICE_AUTH_PROFILE_SAMPLES = max(3, int(config.VOICE_AUTH_PROFILE_SAMPLES))
VOICE_AUTH_REQUIRED_MATCHES = max(1, int(config.VOICE_AUTH_REQUIRED_MATCHES))
VOICE_AUTH_ENROLLMENT_SECONDS = max(
    3.0,
    float(config.VOICE_AUTH_ENROLLMENT_SECONDS),
)

VOICE_AUTH_ENABLED = bool(getattr(config, "VOICE_AUTH_ENABLED", True))
VOICE_AUTH_SCORE_THRESHOLD = max(
    0.25,
    float(getattr(config, "VOICE_AUTH_SCORE_THRESHOLD", 0.32)),
)
VOICE_AUTH_LIVENESS_SECONDS = max(
    4.5,
    float(getattr(config, "VOICE_AUTH_LIVENESS_SECONDS", 6.5)),
)
VOICE_AUTH_KEEP_DIAGNOSTIC_AUDIO = bool(
    getattr(config, "VOICE_AUTH_KEEP_DIAGNOSTIC_AUDIO", False)
)

AUTH_AUDIO_DIR = AUDIO_DIR / "voice_auth"
CALIBRATION_PATH = VOICE_AUTH_PROFILE_DIR / "voice_auth_calibration.json"

MIN_RMS = 0.006
MIN_ACTIVE_SECONDS = 0.70
MAX_PEAK = 0.995
SILENCE_TRIM_PADDING_SECONDS = 0.15

CHALLENGE_ADJECTIVES = (
    "amber",
    "copper",
    "crimson",
    "gentle",
    "golden",
    "lunar",
    "quiet",
    "silver",
    "violet",
    "winter",
)

CHALLENGE_PLACES = (
    "canyon",
    "forest",
    "garden",
    "harbor",
    "river",
    "valley",
    "village",
)

CHALLENGE_OBJECTS = (
    "comet",
    "falcon",
    "lantern",
    "marble",
    "thunder",
)


_speaker_verifier: object | None = None
_whisper_model: WhisperModel | None = None
_speaker_verifier_lock = threading.Lock()
_whisper_model_lock = threading.Lock()


class VoiceAuthError(RuntimeError):
    """Raised when Jarvis cannot securely complete voice authentication."""


@dataclass(frozen=True)
class AudioQuality:
    duration_seconds: float
    rms: float
    peak: float
    active_seconds: float
    active_ratio: float
    clipped: bool
    passed: bool
    reason: str


@dataclass(frozen=True)
class SpeakerVerificationResult:
    passed: bool
    best_score: float
    matching_samples: int
    required_matches: int
    threshold: float
    scores: tuple[float, ...]
    reason: str


@dataclass(frozen=True)
class VoiceLivenessResult:
    passed: bool
    challenge_phrase: str
    transcript: str
    quality: AudioQuality | None
    speaker: SpeakerVerificationResult | None
    reason: str


def _profile_files() -> list[Path]:
    return sorted(VOICE_AUTH_PROFILE_DIR.glob("profile_*.wav"))


def has_voice_profile() -> bool:
    return len(_profile_files()) >= VOICE_AUTH_PROFILE_SAMPLES


def _normalise_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", str(value).lower())
    return re.sub(r"\s+", " ", text).strip()


def _random_challenge_phrase() -> str:
    random_source = secrets.SystemRandom()

    return " ".join(
        (
            random_source.choice(CHALLENGE_ADJECTIVES),
            random_source.choice(CHALLENGE_PLACES),
            random_source.choice(CHALLENGE_OBJECTS),
        )
    )


def _record_wav(destination: Path, seconds: float) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    try:
        print(f"[Listening for {seconds:.1f} seconds...]")

        recording = sd.rec(
            int(SAMPLE_RATE * seconds),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
        )
        sd.wait()

        sf.write(
            str(destination),
            recording,
            SAMPLE_RATE,
            subtype="PCM_16",
        )
    except Exception as exc:
        raise VoiceAuthError(f"Microphone recording failed: {exc}") from exc

    return destination


def _analyse_audio(audio_path: Path) -> AudioQuality:
    samples, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )
    mono = np.mean(samples, axis=1)

    if len(mono) == 0:
        return AudioQuality(0.0, 0.0, 0.0, 0.0, 0.0, False, False, "Empty audio.")

    duration_seconds = len(mono) / float(sample_rate)
    rms = float(np.sqrt(np.mean(np.square(mono))))
    peak = float(np.max(np.abs(mono)))

    frame_length = max(1, int(sample_rate * 0.03))
    usable_length = (len(mono) // frame_length) * frame_length

    if usable_length == 0:
        return AudioQuality(
            duration_seconds,
            rms,
            peak,
            0.0,
            0.0,
            False,
            False,
            "Audio was too short.",
        )

    frames = mono[:usable_length].reshape(-1, frame_length)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    activity_floor = max(MIN_RMS, rms * 0.35)
    active_ratio = float(np.mean(frame_rms >= activity_floor))
    active_seconds = active_ratio * duration_seconds
    clipped = peak >= MAX_PEAK

    if rms < MIN_RMS:
        reason = "Voice was too quiet."
    elif active_seconds < MIN_ACTIVE_SECONDS:
        reason = "Not enough clear speech was detected."
    elif clipped:
        reason = "Audio was clipped. Move slightly away from the microphone."
    else:
        reason = "Audio quality passed."

    return AudioQuality(
        duration_seconds=duration_seconds,
        rms=rms,
        peak=peak,
        active_seconds=active_seconds,
        active_ratio=active_ratio,
        clipped=clipped,
        passed=reason == "Audio quality passed.",
        reason=reason,
    )


def _get_whisper_model() -> WhisperModel:
    global _whisper_model

    with _whisper_model_lock:
        if _whisper_model is None:
            print("[Loading offline speech-recognition model for liveness...]")

            _whisper_model = WhisperModel(
                WHISPER_MODEL,
                device="cpu",
                compute_type="int8",
                download_root=str(MODELS_DIR),
            )

    return _whisper_model


def _transcribe_audio(audio_path: Path) -> str:
    model = _get_whisper_model()

    try:
        segments, _ = model.transcribe(
            str(audio_path),
            language="en",
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            temperature=0.0,
        )

        return " ".join(segment.text.strip() for segment in segments).strip()
    except Exception as exc:
        raise VoiceAuthError(f"Offline transcription failed: {exc}") from exc


def _speech_only_waveform(audio_path: Path) -> torch.Tensor:
    samples, source_rate = sf.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )
    mono = np.mean(samples, axis=1)

    if len(mono) == 0:
        raise VoiceAuthError("The audio file is empty.")

    rms = float(np.sqrt(np.mean(np.square(mono))))
    speech_floor = max(MIN_RMS, rms * 0.35)
    speech_positions = np.flatnonzero(np.abs(mono) >= speech_floor)

    if len(speech_positions) == 0:
        raise VoiceAuthError("No usable speech was found in the audio.")

    padding = int(source_rate * SILENCE_TRIM_PADDING_SECONDS)
    start = max(0, int(speech_positions[0]) - padding)
    end = min(len(mono), int(speech_positions[-1]) + padding + 1)

    mono = mono[start:end]

    waveform = torch.from_numpy(mono.copy()).to(
        dtype=torch.float32
    ).unsqueeze(0)

    if source_rate != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=source_rate,
            new_freq=SAMPLE_RATE,
        )

    minimum_samples = int(SAMPLE_RATE * 0.50)

    if waveform.shape[1] < minimum_samples:
        raise VoiceAuthError("The spoken part of the audio was too short.")

    return waveform


def _get_speaker_verifier() -> object:
    global _speaker_verifier

    with _speaker_verifier_lock:
        if _speaker_verifier is None:
            try:
                from speechbrain.pretrained import SpeakerRecognition

                print("[Loading offline voice-verification model...]")
                VOICE_AUTH_MODEL_DIR.mkdir(parents=True, exist_ok=True)

                _speaker_verifier = SpeakerRecognition.from_hparams(
                    source=VOICE_AUTH_MODEL_SOURCE,
                    savedir=str(VOICE_AUTH_MODEL_DIR),
                    run_opts={"device": "cpu"},
                )
            except Exception as exc:
                raise VoiceAuthError(
                    "Voice-verification model could not load. "
                    "Run the Phase 14 package reset commands before continuing. "
                    f"Details: {exc}"
                ) from exc

    return _speaker_verifier


def _embedding_from_file(audio_path: Path) -> torch.Tensor:
    verifier = _get_speaker_verifier()
    waveform = _speech_only_waveform(audio_path)

    try:
        with torch.inference_mode():
            embedding = verifier.encode_batch(waveform)
    except Exception as exc:
        raise VoiceAuthError(f"Speaker embedding extraction failed: {exc}") from exc

    embedding = embedding.reshape(1, -1).to(dtype=torch.float32)

    return F.normalize(embedding, p=2, dim=1).squeeze(0).cpu()


def _verify_speaker_recording(audio_path: Path) -> SpeakerVerificationResult:
    profiles = _profile_files()

    if len(profiles) < VOICE_AUTH_PROFILE_SAMPLES:
        return SpeakerVerificationResult(
            passed=False,
            best_score=0.0,
            matching_samples=0,
            required_matches=VOICE_AUTH_REQUIRED_MATCHES,
            threshold=VOICE_AUTH_SCORE_THRESHOLD,
            scores=(),
            reason=(
                "Voice profile is incomplete. "
                "Run enroll_voice.py before using Jarvis."
            ),
        )

    candidate_embedding = _embedding_from_file(audio_path)
    profile_embeddings = [_embedding_from_file(profile) for profile in profiles]

    scores = tuple(
        float(torch.dot(candidate_embedding, profile_embedding).item())
        for profile_embedding in profile_embeddings
    )

    required_matches = min(VOICE_AUTH_REQUIRED_MATCHES, len(scores))
    matching_samples = sum(
        score >= VOICE_AUTH_SCORE_THRESHOLD
        for score in scores
    )
    best_score = max(scores, default=0.0)
    passed = matching_samples >= required_matches

    if passed:
        reason = "Speaker verification passed."
    else:
        reason = (
            f"Only {matching_samples} profile samples matched; "
            f"{required_matches} are required."
        )

    return SpeakerVerificationResult(
        passed=passed,
        best_score=best_score,
        matching_samples=matching_samples,
        required_matches=required_matches,
        threshold=VOICE_AUTH_SCORE_THRESHOLD,
        scores=scores,
        reason=reason,
    )


def _phrase_matches(challenge_phrase: str, transcript: str) -> bool:
    expected_words = _normalise_text(challenge_phrase).split()
    heard_words = set(_normalise_text(transcript).split())

    return bool(expected_words) and all(word in heard_words for word in expected_words)


def run_voice_liveness_check(
    *,
    interactive: bool,
    speak_prompt: bool,
    verify_speaker: bool = True,
) -> VoiceLivenessResult:
    challenge_phrase = _random_challenge_phrase()

    recording_path = AUTH_AUDIO_DIR / (
        f"liveness_{int(time.time() * 1000)}.wav"
    )

    print("\nJarvis Phase 14: Voice Liveness Check")
    print(f'Say exactly: "{challenge_phrase}"')

    if speak_prompt:
        from core.text_to_speech import speak

        speak(f"Security check, Boss. Please say: {challenge_phrase}")
        time.sleep(0.35)

    if interactive:
        input("Press Enter, then speak the phrase clearly when recording begins...")

    try:
        _record_wav(recording_path, VOICE_AUTH_LIVENESS_SECONDS)
        quality = _analyse_audio(recording_path)

        print(
            "Audio quality: "
            f"rms={quality.rms:.3f}, "
            f"peak={quality.peak:.3f}, "
            f"active={quality.active_ratio:.0%}"
        )

        if not quality.passed:
            return VoiceLivenessResult(
                passed=False,
                challenge_phrase=challenge_phrase,
                transcript="",
                quality=quality,
                speaker=None,
                reason=quality.reason,
            )

        transcript = _transcribe_audio(recording_path)

        print(f"[Heard: {transcript or '(nothing)'}]")

        if not _phrase_matches(challenge_phrase, transcript):
            return VoiceLivenessResult(
                passed=False,
                challenge_phrase=challenge_phrase,
                transcript=transcript,
                quality=quality,
                speaker=None,
                reason="The random challenge phrase did not match.",
            )

        if not verify_speaker:
            return VoiceLivenessResult(
                passed=True,
                challenge_phrase=challenge_phrase,
                transcript=transcript,
                quality=quality,
                speaker=None,
                reason="Phrase and audio-quality checks passed.",
            )

        speaker = _verify_speaker_recording(recording_path)

        print(
            "[Voice score: "
            f"{speaker.best_score:.3f}; "
            f"matches: {speaker.matching_samples}/{speaker.required_matches}]"
        )

        return VoiceLivenessResult(
            passed=speaker.passed,
            challenge_phrase=challenge_phrase,
            transcript=transcript,
            quality=quality,
            speaker=speaker,
            reason=speaker.reason,
        )

    except VoiceAuthError as exc:
        return VoiceLivenessResult(
            passed=False,
            challenge_phrase=challenge_phrase,
            transcript="",
            quality=None,
            speaker=None,
            reason=str(exc),
        )

    finally:
        if recording_path.exists() and not VOICE_AUTH_KEEP_DIAGNOSTIC_AUDIO:
            recording_path.unlink(missing_ok=True)


def _save_calibration(profile_scores: list[float]) -> None:
    VOICE_AUTH_PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_count": len(_profile_files()),
        "score_threshold": VOICE_AUTH_SCORE_THRESHOLD,
        "required_matches": VOICE_AUTH_REQUIRED_MATCHES,
        "profile_pair_scores": [round(score, 6) for score in profile_scores],
        "profile_pair_median": (
            round(float(np.median(profile_scores)), 6)
            if profile_scores
            else None
        ),
    }

    CALIBRATION_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def enroll_voice_profile() -> None:
    VOICE_AUTH_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    AUTH_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    for old_profile in _profile_files():
        old_profile.unlink(missing_ok=True)

    CALIBRATION_PATH.unlink(missing_ok=True)

    enrollment_phrases = (
        "Jarvis, this is my secure voice profile sample one.",
        "Jarvis, I am recording my secure voice profile sample two.",
        "Jarvis, only my verified voice may operate this assistant.",
        "Jarvis, this profile helps protect my personal assistant.",
        "Jarvis, I authorize secure access with my natural voice.",
    )

    print("\nJarvis Phase 14: Voice Profile Enrollment")
    print(
        "Use your normal voice and the same microphone you will use daily. "
        "Speak clearly without music or another person nearby."
    )

    for index in range(VOICE_AUTH_PROFILE_SAMPLES):
        phrase = enrollment_phrases[index % len(enrollment_phrases)]

        destination = VOICE_AUTH_PROFILE_DIR / (
            f"profile_{index + 1:02d}.wav"
        )

        print(f"\nSample {index + 1} of {VOICE_AUTH_PROFILE_SAMPLES}")
        print(f'Say naturally: "{phrase}"')
        input("Press Enter when you are ready...")

        _record_wav(destination, VOICE_AUTH_ENROLLMENT_SECONDS)
        quality = _analyse_audio(destination)

        print(
            "Audio quality: "
            f"rms={quality.rms:.3f}, "
            f"peak={quality.peak:.3f}, "
            f"active={quality.active_ratio:.0%}"
        )

        if not quality.passed:
            destination.unlink(missing_ok=True)

            raise VoiceAuthError(
                f"Enrollment stopped at sample {index + 1}: {quality.reason} "
                "Run enroll_voice.py again."
            )

    profile_embeddings = [
        _embedding_from_file(profile)
        for profile in _profile_files()
    ]

    profile_scores = [
        float(torch.dot(left, right).item())
        for left, right in itertools.combinations(profile_embeddings, 2)
    ]

    median_score = float(np.median(profile_scores)) if profile_scores else 0.0

    if median_score < VOICE_AUTH_SCORE_THRESHOLD:
        for profile in _profile_files():
            profile.unlink(missing_ok=True)

        raise VoiceAuthError(
            "Enrollment recordings were not consistent enough for secure "
            f"verification (median score {median_score:.3f}; required "
            f"{VOICE_AUTH_SCORE_THRESHOLD:.3f}). Move closer to the microphone "
            "and run enroll_voice.py again."
        )

    _save_calibration(profile_scores)

    print("\nVoice profile enrolled securely.")
    print(
        f"Profile consistency median: {median_score:.3f} | "
        f"verification threshold: {VOICE_AUTH_SCORE_THRESHOLD:.3f}"
    )


def listen_for_verified_text(seconds: float) -> str | None:
    if not VOICE_AUTH_ENABLED:
        from core.speech_to_text import listen_for_text

        return listen_for_text(seconds)

    if not has_voice_profile():
        from core.text_to_speech import speak

        message = (
            "Your voice profile is not enrolled. "
            "Run python enroll_voice.py first."
        )

        print(f"[Access blocked: {message}]")
        speak(message)

        return None

    result = run_voice_liveness_check(
        interactive=False,
        speak_prompt=True,
        verify_speaker=True,
    )

    if not result.passed:
        from core.text_to_speech import speak

        print(f"[Access blocked: {result.reason}]")
        speak("Access blocked. Voice verification failed.")

        return None

    from core.speech_to_text import listen_for_text
    from core.text_to_speech import speak

    speak("Voice verified. What can I do for you?")

    return listen_for_text(seconds)