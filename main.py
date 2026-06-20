# main.py

from __future__ import annotations

from automation.local_commands import handle_local_command
from config import (
    COMMAND_WINDOW_SECONDS,
    VOICE_AUTH_ENABLED,
)
from core.ai_engine import get_ai_response, get_last_backend
from core.speech_to_text import listen_for_text
from core.text_to_speech import speak
from core.voice_auth import (
    has_voice_profile,
    listen_for_verified_text,
)
from core.wake_word import wait_for_wake_word


EXIT_COMMANDS = {
    "exit",
    "quit",
    "stop jarvis",
    "shutdown jarvis",
    "close jarvis",
    "goodbye jarvis",
}

SLEEP_COMMANDS = {
    "sleep",
    "go to sleep",
    "sleep jarvis",
    "go back to sleep",
}


def clean_text(text):
    if text is None:
        return ""

    return str(text).strip()


def create_jarvis_prompt(question):
    return f"""
You are Jarvis, a helpful personal laptop voice assistant.

Answer naturally and clearly.
Keep the response concise because it will be spoken aloud.
Do not use markdown symbols, bullet points, hashtags, or long introductions.
Call the user Boss when it sounds natural.

User question:
{question}
""".strip()


def get_jarvis_response(question):
    try:
        prompt = create_jarvis_prompt(question)
        response = get_ai_response(prompt)

        response = clean_text(response)

        if not response:
            return "Sorry Boss, I could not generate a response."

        return response

    except Exception as error:
        print(f"AI Response Error: {error}")
        return "Sorry Boss, I am having trouble connecting to my AI system."


def main():
    print("Jarvis Phase 7 is running.")
    print("Say 'Jarvis' to activate me.")
    print("Say 'stop Jarvis' after activation to close the program.")

    if VOICE_AUTH_ENABLED:
        print("Voice profile verification: enabled")

        if not has_voice_profile():
            print("Voice profile not found.")
            print("Run this once before using Jarvis:")
            print("python enroll_voice.py")

    else:
        print("Voice profile verification: disabled")

    while True:
        try:
            wake_result = wait_for_wake_word()

            if wake_result is False:
                continue

            if VOICE_AUTH_ENABLED and not has_voice_profile():
                print(
                    "[Access blocked: Enroll your voice profile before "
                    "using Jarvis.]"
                )
                continue

            speak("Yes Boss, what can I do for you?")

            if VOICE_AUTH_ENABLED:
                question, is_verified, score, verification_message = (
                    listen_for_verified_text(COMMAND_WINDOW_SECONDS)
                )

                print(f"[Voice score: {score:.3f}]")
                print(
                    f"[Voice verification: "
                    f"{verification_message}]"
                )

                if not is_verified:
                    print(
                        "[Command rejected. Returning to sleep mode.]"
                    )
                    continue

            else:
                question = listen_for_text()

            question = clean_text(question)

            print(f"\nYou: {question}")

            if not question:
                speak("I did not hear anything clearly, Boss.")
                continue

            normalized_question = question.lower()

            if normalized_question in EXIT_COMMANDS:
                speak("Goodbye Boss.")
                print("Jarvis stopped.")
                break

            if normalized_question in SLEEP_COMMANDS:
                speak("Going back to sleep, Boss.")
                print("Jarvis is back in sleep mode.")
                continue

            was_handled, local_response = handle_local_command(question)

            if was_handled:
                print(f"Jarvis: {local_response}\n")
                speak(local_response)

                print("Returning to sleep mode.")
                print("Say 'Jarvis' to activate again.\n")
                continue

            print("Jarvis is thinking...")

            response = get_jarvis_response(question)
            backend = clean_text(get_last_backend())

            if backend:
                print(f"AI Backend: {backend}")

            print(f"Jarvis: {response}\n")

            speak(response)

            print("Returning to sleep mode.")
            print("Say 'Jarvis' to activate again.\n")

        except KeyboardInterrupt:
            print("\nJarvis stopped from keyboard.")
            break

        except Exception as error:
            print(f"Jarvis Runtime Error: {error}")
            speak(
                "Sorry Boss, something went wrong. "
                "I am returning to sleep mode."
            )


if __name__ == "__main__":
    main()