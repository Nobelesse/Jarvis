from datetime import datetime

from config import COMMAND_WINDOW_SECONDS
from core.speech_to_text import listen_for_text
from core.text_to_speech import speak
from core.wake_word import wait_for_wake_word


def normalize_command(command: str) -> str:
    return " ".join(
        command.lower().strip().split()
    )


def should_exit(command: str) -> bool:
    normalized = normalize_command(command)

    exit_commands = {
        "exit jarvis",
        "quit jarvis",
        "close jarvis",
        "goodbye jarvis"
    }

    return normalized in exit_commands


def build_phase_one_reply(command: str) -> str:
    """
    Handles only safe local test commands for Phase 1.
    """

    normalized = normalize_command(command)

    if not normalized:
        return "I did not catch that. Returning to sleep."

    if "your name" in normalized or "who are you" in normalized:
        return "I am Jarvis, your local laptop assistant."

    if "time" in normalized:
        current_time = datetime.now().strftime("%I:%M %p")
        return f"It is {current_time}."

    if "date" in normalized or "day is it" in normalized:
        current_date = datetime.now().strftime("%A, %d %B %Y")
        return f"Today is {current_date}."

    if "hello" in normalized or "hi" in normalized:
        return "Hello Boss. Jarvis is online and ready."

    return (
        f"I heard: {command}. "
        "Command intelligence and laptop actions will be added in the next phases."
    )


def main() -> None:
    print("=" * 60)
    print("JARVIS - PHASE 1: LOCAL VOICE FOUNDATION")
    print("=" * 60)
    print("Jarvis is running silently.")
    print("Press Ctrl + C in this terminal to stop testing.\n")

    try:
        while True:
            wait_for_wake_word()

            speak("Hey Boss, what can I do for you?")

            print("[Jarvis is active. Speak your command now.]")
            command = listen_for_text(COMMAND_WINDOW_SECONDS)

            if command:
                print(f"[Command heard: {command}]")

            if should_exit(command):
                speak("Goodbye Boss. Jarvis is shutting down.")
                break

            response = build_phase_one_reply(command)
            speak(response)

    except KeyboardInterrupt:
        print("\n[Jarvis stopped by user.]")

    except Exception as error:
        print(f"\n[Jarvis stopped because of an error: {error}]")


if __name__ == "__main__":
    main()