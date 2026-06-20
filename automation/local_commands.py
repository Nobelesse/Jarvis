from __future__ import annotations

import ctypes
import os
import re
import subprocess
import webbrowser
from datetime import datetime
from typing import Callable
from urllib.parse import quote_plus


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF


def _clean_command(text: str) -> str:
    command = str(text).lower().strip()
    command = command.replace("jarvis", " ")
    command = re.sub(r"[^a-z0-9\s']", " ", command)
    command = re.sub(r"\s+", " ", command)
    return command.strip()


def _start_process(command: list[str]) -> None:
    subprocess.Popen(
        command,
        creationflags=CREATE_NO_WINDOW,
    )


def _send_media_key(key_code: int, presses: int = 1) -> None:
    user32 = ctypes.windll.user32

    for _ in range(presses):
        user32.keybd_event(key_code, 0, 0, 0)
        user32.keybd_event(key_code, 0, 0x0002, 0)


def _open_notepad() -> str:
    _start_process(["notepad.exe"])
    return "Opening Notepad, Boss."


def _open_calculator() -> str:
    _start_process(["calc.exe"])
    return "Opening Calculator, Boss."


def _open_paint() -> str:
    _start_process(["mspaint.exe"])
    return "Opening Paint, Boss."


def _open_file_explorer() -> str:
    _start_process(["explorer.exe"])
    return "Opening File Explorer, Boss."


def _open_command_prompt() -> str:
    _start_process(["cmd.exe"])
    return "Opening Command Prompt, Boss."


def _open_task_manager() -> str:
    _start_process(["taskmgr.exe"])
    return "Opening Task Manager, Boss."


def _open_settings() -> str:
    os.startfile("ms-settings:")
    return "Opening Windows Settings, Boss."


def _open_google() -> str:
    webbrowser.open_new_tab("https://www.google.com")
    return "Opening Google, Boss."


def _open_youtube() -> str:
    webbrowser.open_new_tab("https://www.youtube.com")
    return "Opening YouTube, Boss."


def _open_github() -> str:
    webbrowser.open_new_tab("https://github.com")
    return "Opening GitHub, Boss."


def _open_browser() -> str:
    webbrowser.open_new_tab("about:blank")
    return "Opening your browser, Boss."


APP_ACTIONS: dict[str, Callable[[], str]] = {
    "notepad": _open_notepad,
    "calculator": _open_calculator,
    "paint": _open_paint,
    "file explorer": _open_file_explorer,
    "files": _open_file_explorer,
    "explorer": _open_file_explorer,
    "command prompt": _open_command_prompt,
    "cmd": _open_command_prompt,
    "task manager": _open_task_manager,
    "settings": _open_settings,
    "google": _open_google,
    "youtube": _open_youtube,
    "github": _open_github,
    "browser": _open_browser,
}


def _get_battery_response() -> str:
    powershell_command = (
        "$battery = Get-CimInstance Win32_Battery | Select-Object -First 1; "
        "if ($null -ne $battery) { $battery.EstimatedChargeRemaining }"
    )

    try:
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                powershell_command,
            ],
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=CREATE_NO_WINDOW,
        )

        percentage = result.stdout.strip()

        if percentage.isdigit():
            return f"Your battery is at {percentage} percent, Boss."

        return "I could not detect a battery. Your laptop may be plugged in or Windows did not return battery information."

    except Exception:
        return "Sorry Boss, I could not check the battery right now."


def _google_search(query: str) -> str:
    clean_query = query.strip()

    if not clean_query:
        return "Please tell me what you want to search for, Boss."

    search_url = f"https://www.google.com/search?q={quote_plus(clean_query)}"
    webbrowser.open_new_tab(search_url)

    return f"Searching Google for {clean_query}, Boss."


def _is_open_command(command: str) -> bool:
    return command.startswith(("open ", "launch ", "start "))


def handle_local_command(text: str) -> tuple[bool, str]:
    """
    Returns:
        (True, response)  -> Local command was handled.
        (False, "")       -> Send the request to the AI engine.
    """
    command = _clean_command(text)

    if not command:
        return True, "I did not hear a command, Boss."

    if command.startswith("search for "):
        return True, _google_search(command.removeprefix("search for "))

    if command.startswith("search "):
        return True, _google_search(command.removeprefix("search "))

    if command.startswith("google "):
        return True, _google_search(command.removeprefix("google "))

    if command in {
        "what time is it",
        "tell me the time",
        "tell me time",
        "current time",
        "time",
    }:
        current_time = datetime.now().strftime("%I:%M %p")
        return True, f"It is {current_time}, Boss."

    if command in {
        "what is today's date",
        "what is the date",
        "tell me the date",
        "today's date",
        "date",
        "what day is it",
    }:
        current_date = datetime.now().strftime("%A, %d %B %Y")
        return True, f"Today is {current_date}, Boss."

    if any(
        phrase in command
        for phrase in (
            "battery percentage",
            "battery percent",
            "battery level",
            "battery status",
            "how much battery",
            "how much charge",
        )
    ):
        return True, _get_battery_response()

    if any(
        phrase in command
        for phrase in (
            "volume up",
            "increase volume",
            "raise volume",
            "make it louder",
        )
    ):
        _send_media_key(VK_VOLUME_UP, presses=4)
        return True, "Volume increased, Boss."

    if any(
        phrase in command
        for phrase in (
            "volume down",
            "decrease volume",
            "lower volume",
            "make it quieter",
        )
    ):
        _send_media_key(VK_VOLUME_DOWN, presses=4)
        return True, "Volume decreased, Boss."

    if any(
        phrase in command
        for phrase in (
            "mute volume",
            "mute sound",
            "mute",
            "unmute",
        )
    ):
        _send_media_key(VK_VOLUME_MUTE)
        return True, "Volume toggled, Boss."

    if _is_open_command(command):
        for app_name, action in APP_ACTIONS.items():
            if app_name in command:
                try:
                    return True, action()
                except Exception:
                    return True, f"Sorry Boss, I could not open {app_name}."

    return False, ""