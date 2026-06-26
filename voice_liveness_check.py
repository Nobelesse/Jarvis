from __future__ import annotations

import argparse
import re
import secrets
import sys
from collections.abc import Sequence

from core.voice_auth import has_voice_profile, listen_for_verified_text


CHALLENGE_PHRASES = (
    "amber falcon river",
    "cobalt meadow lantern",
    "crimson valley orchid",
    "silver comet garden",
    "velvet harbor maple",
    "golden canyon sparrow",
    "indigo forest beacon",
    "marble sunrise willow",
)


def normalize_words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def contains_words_in_order(
    heard_words: Sequence[str],
    expected_words: Sequence[str],
) -> bool:
    expected_index = 0

    for word in heard_words:
        if expected_index >= len(expected_words):
            break

        if word == expected_words[expected_index]:
            expected_index += 1

    return expected_index == len(expected_words)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a one-time random voice-liveness test using Jarvis's "
            "existing enrolled voice profile."
        )
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=7.0,
        help="Microphone recording duration in seconds. Default: 7.0",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.seconds < 3.0:
        print("Use at least 3 seconds for a reliable liveness recording.")
        return 2

    if not has_voice_profile():
        print(
            "No complete voice profile was found. "
            "Run: .\\.venv\\Scripts\\python.exe enroll_voice.py"
        )
        return 2

    challenge = secrets.choice(CHALLENGE_PHRASES)
    expected_words = normalize_words(challenge)

    print("\nJarvis Phase 13: Voice Liveness Check")
    print("This phrase is random for this one test.")
    print(f'Say exactly: "{challenge}"')
    input("Press Enter, then speak the phrase clearly when recording begins...")

    text, is_verified, score, message = listen_for_verified_text(
        duration_seconds=args.seconds,
    )

    print(f"\nVoice verification: {'PASSED' if is_verified else 'FAILED'}")
    print(f"Voice score: {score:.3f}")
    print(f"Details: {message}")

    if not is_verified:
        print(
            "\nLiveness result: FAILED. This attempt was rejected before "
            "the phrase check."
        )
        return 1

    heard_words = normalize_words(text)
    liveness_passed = contains_words_in_order(
        heard_words=heard_words,
        expected_words=expected_words,
    )

    print(f"Recognized phrase: {text or '[nothing recognized]'}")
    print(
        "Liveness result: "
        f"{'PASSED' if liveness_passed else 'FAILED'}"
    )

    if liveness_passed:
        print(
            "\nSecure test passed. Your live voice matched the enrolled "
            "profile and repeated the newly generated phrase."
        )
        return 0

    print(
        "\nVoice matched, but the spoken phrase did not match the random "
        "challenge. Speak the phrase again in the same order and rerun "
        "the test."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())