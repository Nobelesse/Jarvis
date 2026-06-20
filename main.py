from core.ai_engine import get_ai_response
from core.speech_to_text import listen_for_text
from core.text_to_speech import speak


EXIT_COMMANDS = {
    "exit",
    "quit",
    "shutdown jarvis",
    "stop jarvis",
    "goodbye jarvis",
}


def is_exit_command(command):
    return command.lower().strip() in EXIT_COMMANDS


def main():
    print("Jarvis Phase 2 is running.")
    print("Speak a question. Say 'stop Jarvis' to close the program.")

    while True:
        command = listen_for_text()

        if not command:
            continue

        command = command.strip()

        print(f"You said: {command}")

        if is_exit_command(command):
            goodbye_message = "Goodbye Boss. Jarvis is shutting down."
            print(f"Jarvis: {goodbye_message}")
            speak(goodbye_message)
            break

        print("Jarvis is thinking...")

        response = get_ai_response(command)

        print(f"Jarvis: {response}")

        speak(response)


if __name__ == "__main__":
    main()