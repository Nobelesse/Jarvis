# enroll_voice.py

from __future__ import annotations

from config import (
    VOICE_AUTH_ENROLLMENT_SECONDS,
    VOICE_AUTH_PROFILE_SAMPLES,
)
from core.voice_auth import (
    delete_audio_files,
    get_voice_profile_files,
    record_voice_profile_sample,
    replace_voice_profile,
)


ENROLLMENT_PHRASES = [
    "Hello Jarvis. This is my voice profile.",
    "Jarvis, please verify that this is my voice.",
    "I am the authorized user of this laptop.",
    "Jarvis, voice authentication is now being enrolled.",
    "This is my secure Jarvis voice profile.",
]


def main() -> None:
    existing_samples = get_voice_profile_files()

    print("Jarvis Voice Profile Enrollment")
    print("-" * 40)
    print(f"Samples required: {VOICE_AUTH_PROFILE_SAMPLES}")
    print(f"Recording time per sample: {VOICE_AUTH_ENROLLMENT_SECONDS:.1f} seconds")

    if existing_samples:
        print(
            f"\nYour existing profile has {len(existing_samples)} sample(s). "
            "It will be replaced after successful enrollment."
        )

    print("\nUse your normal daily speaking voice.")
    print("Keep the laptop microphone distance similar to normal Jarvis use.")
    print("Choose a quiet room and avoid music, fan noise, or other voices.")

    captured_samples = []

    try:
        for sample_number in range(1, VOICE_AUTH_PROFILE_SAMPLES + 1):
            phrase = ENROLLMENT_PHRASES[
                (sample_number - 1) % len(ENROLLMENT_PHRASES)
            ]

            print(f"\nSample {sample_number} of {VOICE_AUTH_PROFILE_SAMPLES}")
            print(f'Say clearly: "{phrase}"')

            input("Press Enter when you are ready...")

            sample_path = record_voice_profile_sample(sample_number)
            captured_samples.append(sample_path)

            print(f"[Sample {sample_number} captured]")

        replace_voice_profile(captured_samples)

        saved_profiles = get_voice_profile_files()

        print("\nVoice profile enrolled successfully.")
        print(
            f"Saved samples: {len(saved_profiles)} "
            f"of {VOICE_AUTH_PROFILE_SAMPLES}"
        )

    except KeyboardInterrupt:
        print("\nEnrollment cancelled. Your existing profile was not changed.")

    except Exception as error:
        print(f"\nVoice enrollment failed: {error}")

    finally:
        delete_audio_files(captured_samples)


if __name__ == "__main__":
    main()