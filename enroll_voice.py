# enroll_voice.py

from __future__ import annotations

import sys
from pathlib import Path

from config import VOICE_AUTH_PROFILE_SAMPLES
from core.voice_auth import (
    VOICE_ENROLLMENT_MINIMUM_SPEECH_SECONDS,
    delete_audio_files,
    format_voice_audio_quality,
    format_voice_profile_diagnostics,
    get_voice_profile_diagnostics,
    get_voice_quality_issues,
    inspect_voice_recording,
    record_voice_profile_sample,
    replace_voice_profile,
    warm_up_voice_verifier,
)


ENROLLMENT_SENTENCES = (
    "Jarvis, this is my secure voice profile. Please verify my voice.",
    "Jarvis, I am the authorized user of this assistant.",
    "Jarvis, confirm my voice profile for secure laptop access.",
    "Jarvis, my voice should unlock my personal assistant.",
    "Jarvis, this recording is for my offline voice verification.",
)

MAX_SAMPLE_ATTEMPTS = 3
MAX_ENROLLMENT_SESSIONS = 2


def get_enrollment_sentence(sample_number: int) -> str:
    index = (sample_number - 1) % len(ENROLLMENT_SENTENCES)
    return ENROLLMENT_SENTENCES[index]


def record_valid_sample(sample_number: int) -> Path | None:
    sentence = get_enrollment_sentence(sample_number)

    for attempt in range(1, MAX_SAMPLE_ATTEMPTS + 1):
        print()
        print(
            f"Sample {sample_number} of {VOICE_AUTH_PROFILE_SAMPLES} "
            f"(attempt {attempt} of {MAX_SAMPLE_ATTEMPTS})"
        )
        print("Speak this sentence naturally until recording ends:")
        print(f'"{sentence}"')
        input("Press Enter when you are ready...")

        audio_path = record_voice_profile_sample(sample_number)

        try:
            audio_info = inspect_voice_recording(audio_path)
            print(format_voice_audio_quality(audio_info))

            issues = get_voice_quality_issues(
                audio_info,
                minimum_speech_seconds=(
                    VOICE_ENROLLMENT_MINIMUM_SPEECH_SECONDS
                ),
            )
        except Exception as error:
            print(f"Audio-quality check failed: {error}")
            delete_audio_files([audio_path])
            continue

        if not issues:
            print(f"Saved temporary sample: {audio_path}")
            return audio_path

        print("This sample was not clear enough:")

        for issue in issues:
            print(f"- {issue}")

        delete_audio_files([audio_path])
        print("The sample was discarded. Record it again.")

    return None


def enroll_voice() -> bool:
    print("Jarvis Phase 12: Secure Voice Enrollment")
    print(
        "Use the same microphone and sit in a quiet room. "
        "Do not play music or let Jarvis speak during recording."
    )

    try:
        warm_up_voice_verifier()
    except Exception as error:
        print(f"Voice-verification model could not start: {error}")
        return False

    for session in range(1, MAX_ENROLLMENT_SESSIONS + 1):
        print()
        print(
            f"Enrollment session {session} of "
            f"{MAX_ENROLLMENT_SESSIONS}"
        )

        sample_paths: list[Path] = []

        try:
            for sample_number in range(
                1,
                VOICE_AUTH_PROFILE_SAMPLES + 1,
            ):
                audio_path = record_valid_sample(sample_number)

                if audio_path is None:
                    print(
                        "Enrollment stopped because a clear sample "
                        "was not recorded."
                    )
                    return False

                sample_paths.append(audio_path)

            print()
            print("Checking whether your profile samples match each other...")

            report = get_voice_profile_diagnostics(sample_paths)
            print(format_voice_profile_diagnostics(report))

            if bool(report.get("healthy", False)):
                replace_voice_profile(sample_paths)

                print()
                print("Voice profile enrolled successfully.")
                print(
                    "Your existing profile was replaced only after "
                    "the new profile passed calibration."
                )
                return True

            print()
            print(
                "Enrollment calibration failed. The existing profile "
                "was kept unchanged."
            )
            print(
                "Use the same microphone position for every sample and "
                "speak naturally at a steady volume."
            )

        finally:
            delete_audio_files(sample_paths)

    return False


def main() -> None:
    success = enroll_voice()

    if not success:
        print()
        print("Voice enrollment was not completed.")
        sys.exit(1)


if __name__ == "__main__":
    main()