# automation/app_launcher.py

from __future__ import annotations

import os
import re
import shutil
import subprocess
import webbrowser
from pathlib import Path


CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


WEBSITES = {
    "google": "https://www.google.com",
    "youtube": "https://www.youtube.com",
    "gmail": "https://mail.google.com",
    "github": "https://github.com",
}


ALIASES = {
    "notepad": "notepad",
    "calculator": "calculator",
    "calc": "calculator",
    "file explorer": "file_explorer",
    "explorer": "file_explorer",
    "files": "file_explorer",
    "this pc": "file_explorer",
    "downloads": "downloads",
    "downloads folder": "downloads",
    "folder downloads": "downloads",
    "documents": "documents",
    "documents folder": "documents",
    "folder documents": "documents",
    "desktop": "desktop",
    "desktop folder": "desktop",
    "folder desktop": "desktop",
    "settings": "settings",
    "windows settings": "settings",
    "task manager": "task_manager",
    "control panel": "control_panel",
    "terminal": "terminal",
    "command prompt": "terminal",
    "cmd": "terminal",
    "powershell": "terminal",
    "visual studio code": "vs_code",
    "vs code": "vs_code",
    "vscode": "vs_code",
    "chrome": "chrome",
    "google chrome": "chrome",
    "browser": "browser",
    "google": "google",
    "youtube": "youtube",
    "gmail": "gmail",
    "github": "github",
}


HELP_COMMANDS = {
    "what apps can you open",
    "what can you open",
    "which apps can you open",
    "show launcher commands",
    "launcher commands",
}


def clean_command(text: str) -> str:
    text = str(text or "").lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_target(command: str) -> str | None:
    command = clean_command(command)

    for prefix in ("open ", "launch ", "start ", "run "):
        if command.startswith(prefix):
            target = command[len(prefix):].strip()
            return target or None

    return None


def start_process(command: list[str]) -> bool:
    try:
        subprocess.Popen(
            command,
            creationflags=CREATE_NO_WINDOW,
        )
        return True

    except OSError:
        return False


def launch_first_available(candidates: list[str]) -> bool:
    for candidate in candidates:
        path = Path(candidate)

        if path.is_file():
            if start_process([str(path)]):
                return True
            continue

        if shutil.which(candidate):
            if start_process([candidate]):
                return True

    return False


def get_program_files_paths() -> tuple[str, str, str]:
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    program_files = os.environ.get("PROGRAMFILES", "")
    program_files_x86 = os.environ.get("PROGRAMFILES(X86)", "")

    return local_app_data, program_files, program_files_x86


def open_folder(folder: Path, label: str) -> str:
    if not folder.exists():
        return f"I could not find your {label} folder, Boss."

    if start_process(["explorer.exe", str(folder)]):
        return f"Opening your {label} folder, Boss."

    return f"I could not open your {label} folder, Boss."


def open_settings() -> str:
    try:
        os.startfile("ms-settings:")
        return "Opening Windows Settings, Boss."
    except OSError:
        return "I could not open Windows Settings, Boss."


def open_website(name: str) -> str:
    url = WEBSITES[name]
    opened = webbrowser.open_new_tab(url)

    if opened:
        return f"Opening {name.title()}, Boss."

    return f"I could not open {name.title()}, Boss."


def open_notepad() -> str:
    if launch_first_available(["notepad.exe"]):
        return "Opening Notepad, Boss."

    return "I could not open Notepad, Boss."


def open_calculator() -> str:
    if launch_first_available(["calc.exe"]):
        return "Opening Calculator, Boss."

    return "I could not open Calculator, Boss."


def open_file_explorer() -> str:
    if launch_first_available(["explorer.exe"]):
        return "Opening File Explorer, Boss."

    return "I could not open File Explorer, Boss."


def open_task_manager() -> str:
    if launch_first_available(["taskmgr.exe"]):
        return "Opening Task Manager, Boss."

    return "I could not open Task Manager, Boss."


def open_control_panel() -> str:
    if launch_first_available(["control.exe"]):
        return "Opening Control Panel, Boss."

    return "I could not open Control Panel, Boss."


def open_terminal() -> str:
    if launch_first_available(["wt.exe", "powershell.exe", "cmd.exe"]):
        return "Opening the terminal, Boss."

    return "I could not open a terminal, Boss."


def open_vs_code() -> str:
    local_app_data, program_files, program_files_x86 = get_program_files_paths()

    candidates = [
        "code",
        str(Path(local_app_data) / "Programs" / "Microsoft VS Code" / "Code.exe"),
        str(Path(program_files) / "Microsoft VS Code" / "Code.exe"),
        str(Path(program_files_x86) / "Microsoft VS Code" / "Code.exe"),
    ]

    if launch_first_available(candidates):
        return "Opening Visual Studio Code, Boss."

    return "I could not find Visual Studio Code. Please make sure VS Code is installed, Boss."


def open_chrome() -> str:
    local_app_data, program_files, program_files_x86 = get_program_files_paths()

    candidates = [
        "chrome.exe",
        str(Path(program_files) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(program_files_x86) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(local_app_data) / "Google" / "Chrome" / "Application" / "chrome.exe"),
    ]

    if launch_first_available(candidates):
        return "Opening Google Chrome, Boss."

    return "I could not find Google Chrome, Boss."


def open_default_browser() -> str:
    opened = webbrowser.open_new_tab("https://www.google.com")

    if opened:
        return "Opening your default browser, Boss."

    return "I could not open your default browser, Boss."


def launcher_help() -> str:
    return (
        "You can ask me to open Notepad, Calculator, File Explorer, Downloads, "
        "Documents, Desktop, Settings, Task Manager, Control Panel, Terminal, "
        "VS Code, Chrome, Google, YouTube, Gmail, or GitHub, Boss."
    )


def handle_app_command(command: str) -> str | None:
    normalized_command = clean_command(command)

    if normalized_command in HELP_COMMANDS:
        return launcher_help()

    target = extract_target(normalized_command)

    if not target:
        return None

    target = ALIASES.get(target)

    if not target:
        return None

    home_folder = Path.home()

    if target in WEBSITES:
        return open_website(target)

    if target == "notepad":
        return open_notepad()

    if target == "calculator":
        return open_calculator()

    if target == "file_explorer":
        return open_file_explorer()

    if target == "downloads":
        return open_folder(home_folder / "Downloads", "Downloads")

    if target == "documents":
        return open_folder(home_folder / "Documents", "Documents")

    if target == "desktop":
        return open_folder(home_folder / "Desktop", "Desktop")

    if target == "settings":
        return open_settings()

    if target == "task_manager":
        return open_task_manager()

    if target == "control_panel":
        return open_control_panel()

    if target == "terminal":
        return open_terminal()

    if target == "vs_code":
        return open_vs_code()

    if target == "chrome":
        return open_chrome()

    if target == "browser":
        return open_default_browser()

    return None