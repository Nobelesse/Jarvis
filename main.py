from core.ai_engine import get_ai_response
from core.speech_to_text import listen_for_text
from core.text_to_speech import speak


STOP_COMMANDS = {
    "stop",
    "stop jarvis",
    "jarvis stop",
    "exit",
    "exit jarvis",
    "jarvis exit",
    "quit",
    "quit jarvis",
    "jarvis quit",
}


def should_stop_jarvis(text):
    cleaned_text = str(text).lower().strip()
    return cleaned_text in STOP_COMMANDS


def main():
    print("Jarvis Phase 3 is running.")
    print("Speak a question. Say 'stop Jarvis' to close the program.")

    try:
        while True:
            user_text = listen_for_text()

            if not user_text:
                continue

            print(f"\nYou: {user_text}")

            if should_stop_jarvis(user_text):
                goodbye_message = "Goodbye Boss. Jarvis is shutting down."

                print(f"Jarvis: {goodbye_message}")
                speak(goodbye_message)
                break

            print("Jarvis: Thinking...")

            response = get_ai_response(user_text)

            print(f"Jarvis: {response}\n")
            speak(response)

    except KeyboardInterrupt:
        print("\nJarvis stopped from the keyboard.")


if __name__ == "__main__":
    main()