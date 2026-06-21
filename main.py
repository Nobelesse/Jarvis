# main.py

from __future__ import annotations

from config import (
    COMMAND_WINDOW_SECONDS,
    VOICE_AUTH_ENABLED,
)

from automation.app_launcher import handle_app_command
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


def clean_text(text) -> str:
    if text is None:
        return ""

    return str(text).strip()


def create_jarvis_prompt(question: str) -> str:
    return f"""
You are Jarvis, a helpful personal laptop voice assistant.

Answer naturally and clearly.
Keep the response concise because it will be spoken aloud.
Do not use markdown symbols, bullet points, hashtags, or long introductions.
Call the user Boss when it sounds natural.

User question:
{question}
""".strip()


def get_jarvis_response(question: str) -> str:
    try:
        prompt = create_jarvis_prompt(question)
        response = clean_text(get_ai_response(prompt))

        backend = clean_text(get_last_backend())

        if backend:
            print(f"[AI backend: {backend}]")

        if response:
            return response

        return "I could not generate a response right now, Boss."

    except Exception as error:
        print(f"Jarvis AI Error: {error}")
        return "I ran into a problem while processing that, Boss."


def unwrap_listener_result(result) -> str:
    if isinstance(result, str):
        return clean_text(result)

    if isinstance(result, (tuple, list)):
        for item in result:
            if isinstance(item, str) and item.strip():
                return clean_text(item)

    return ""


def call_listener(listener) -> str:
    attempts = [
        lambda: listener(timeout=COMMAND_WINDOW_SECONDS),
        lambda: listener(phrase_time_limit=COMMAND_WINDOW_SECONDS),
        lambda: listener(COMMAND_WINDOW_SECONDS),
        lambda: listener(),
    ]

    for attempt in attempts:
        try:
            result = attempt()
            return unwrap_listener_result(result)

        except TypeError:
            continue

        except Exception as error:
            print(f"Listening Error: {error}")
            return ""

    return ""


def listen_for_verified_command() -> str:
    try:
        result = listen_for_verified_text(COMMAND_WINDOW_SECONDS)

    except Exception as error:
        print(f"Voice Authentication Error: {error}")
        speak("I could not verify your voice, Boss.")
        return ""

    if not isinstance(result, tuple) or len(result) != 4:
        print("[Access blocked: Voice authentication returned an invalid result.]")
        speak("I could not verify your voice, Boss.")
        return ""

    command, is_verified, score, message = result

    command = clean_text(command)
    message = clean_text(message)

    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0

    if not is_verified:
        print(f"[Access blocked: {message}]")
        print(f"[Voice score: {score_value:.3f}]")
        speak("I could not verify your voice, Boss.")
        return ""

    print(f"[Voice verified: {message}]")
    print(f"[Voice score: {score_value:.3f}]")

    return command


def listen_for_command() -> str:
    if VOICE_AUTH_ENABLED:
        try:
            profile_exists = has_voice_profile()
        except Exception as error:
            print(f"Voice Profile Check Error: {error}")
            profile_exists = False

        if not profile_exists:
            message = (
                "Your voice profile is not enrolled. "
                "Run python enroll_voice.py first."
            )
            print(f"[Access blocked: {message}]")
            speak(message)
            return ""

        return listen_for_verified_command()

    return call_listener(listen_for_text)


def main() -> None:
    print("Jarvis Phase 8 is running.")
    print("Say 'Jarvis' to activate me.")
    print("Voice authentication remains enabled before commands are accepted.")

    while True:
        wait_for_wake_word()

        speak("Hey Boss, what can I do for you?")

        command = clean_text(listen_for_command())

        if not command:
            continue

        print(f"You: {command}")

        normalized_command = command.lower()

        if normalized_command in EXIT_COMMANDS:
            speak("Goodbye Boss.")
            break

        if normalized_command in SLEEP_COMMANDS:
            speak("Going back to sleep, Boss.")
            continue

        app_response = handle_app_command(command)

        if app_response:
            print(f"Jarvis: {app_response}")
            speak(app_response)
            continue

        response = get_jarvis_response(command)

        print(f"Jarvis: {response}")
        speak(response)


if __name__ == "__main__":
    main()