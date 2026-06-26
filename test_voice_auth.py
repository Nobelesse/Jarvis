# test_voice_auth.py

from __future__ import annotations

import sys

from config import VOICE_AUTH_ENROLLMENT_SECONDS
from core.voice_auth import (
    delete_audio_files,
    format_voice_audio_quality,
    format_voice_profile_diagnostics,
    get_voice_profile_diagnostics,
    has_voice_profile,
    inspect_voice_recording,
    record_voice_verification_sample,
    verify_voice_recording,
)


def main() -> None:
    if not has_voice_profile():
        print("No complete voice profile exists.")
        print("Run: python enroll_voice.py")
        sys.exit(1)

    print("Jarvis Phase 12: Voice Authentication Test")
    print()
    print("First checking the stored voice profile...")

    report = get_voice_profile_diagnostics()
    print(format_voice_profile_diagnostics(report))

    if not bool(report.get("healthy", False)):
        print()
        print("The stored profile is not reliable enough.")
        print("Run: python enroll_voice.py")
        sys.exit(1)

    print()
    print(
        "Speak a normal command for the next "
        f"{VOICE_AUTH_ENROLLMENT_SECONDS:.1f} seconds."
    )
    input("Press Enter when you are ready...")

    audio_path = record_voice_verification_sample(
        VOICE_AUTH_ENROLLMENT_SECONDS
    )

    try:
        audio_info = inspect_voice_recording(audio_path)
        print(format_voice_audio_quality(audio_info))

        is_verified, score, message = verify_voice_recording(audio_path)

        print(f"Verified: {is_verified}")
        print(f"Voice score: {score:.3f}")
        print(f"Message: {message}")

        if not is_verified:
            sys.exit(1)

    finally:
        delete_audio_files([audio_path])


if __name__ == "__main__":
    main()