# process_manager.py

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import psutil


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = DATA_DIR / "logs"

MAIN_FILE = PROJECT_DIR / "main.py"
TRAY_FILE = PROJECT_DIR / "tray_app.py"
PID_FILE = DATA_DIR / "jarvis.pid"

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def ensure_data_folders() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def save_pid() -> None:
    ensure_data_folders()
    PID_FILE.write_text(str(__import__("os").getpid()), encoding="utf-8")


def clear_pid() -> None:
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _normalise_path(path: Path) -> str:
    return str(path.resolve()).replace("/", "\\").casefold()


def _find_processes_for_file(file_path: Path) -> list[psutil.Process]:
    target_path = _normalise_path(file_path)
    found_processes: list[psutil.Process] = []

    for process in psutil.process_iter(["pid", "cmdline"]):
        try:
            command_line = " ".join(process.info.get("cmdline") or [])
            normalised_command = command_line.replace("/", "\\").casefold()

            if target_path in normalised_command:
                found_processes.append(process)

        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            continue

    return found_processes


def find_jarvis_processes() -> list[psutil.Process]:
    return _find_processes_for_file(MAIN_FILE)


def find_tray_processes() -> list[psutil.Process]:
    return _find_processes_for_file(TRAY_FILE)


def is_jarvis_running() -> bool:
    return bool(find_jarvis_processes())


def is_tray_running() -> bool:
    return bool(find_tray_processes())


def _pythonw_path() -> Path:
    virtual_environment_pythonw = (
        PROJECT_DIR / ".venv" / "Scripts" / "pythonw.exe"
    )

    if virtual_environment_pythonw.exists():
        return virtual_environment_pythonw

    current_pythonw = Path(sys.executable).with_name("pythonw.exe")

    if current_pythonw.exists():
        return current_pythonw

    return Path(sys.executable)


def _stop_processes(
    processes: list[psutil.Process],
    label: str,
) -> tuple[bool, str]:
    if not processes:
        return False, f"No {label} process is running."

    active_processes: list[psutil.Process] = []

    for process in processes:
        try:
            if process.is_running():
                process.terminate()
                active_processes.append(process)
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            continue

    _, still_running = psutil.wait_procs(active_processes, timeout=5)

    for process in still_running:
        try:
            process.kill()
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            continue

    if label == "Jarvis":
        clear_pid()

    return True, f"{label} stopped."


def stop_jarvis() -> tuple[bool, str]:
    return _stop_processes(find_jarvis_processes(), "Jarvis")


def stop_tray() -> tuple[bool, str]:
    return _stop_processes(find_tray_processes(), "Tray")


def _launch_hidden(script_file: Path, log_name: str) -> tuple[bool, str]:
    if not script_file.exists():
        return False, f"Missing required file: {script_file.name}"

    ensure_data_folders()

    log_file = LOG_DIR / log_name

    try:
        with log_file.open("a", encoding="utf-8") as log_stream:
            log_stream.write(
                f"\n\n--- Started {script_file.name} ---\n"
            )

            subprocess.Popen(
                [str(_pythonw_path()), str(script_file)],
                cwd=str(PROJECT_DIR),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                creationflags=CREATE_NO_WINDOW,
            )

        time.sleep(0.5)

        return True, f"{script_file.stem} started in the background."

    except OSError as error:
        return False, f"Could not start {script_file.name}: {error}"


def start_jarvis_hidden() -> tuple[bool, str]:
    if is_jarvis_running():
        return False, "Jarvis is already running."

    return _launch_hidden(MAIN_FILE, "jarvis-runtime.log")


def start_tray_hidden() -> tuple[bool, str]:
    if is_tray_running():
        return False, "Jarvis tray is already running."

    return _launch_hidden(TRAY_FILE, "tray-runtime.log")


def restart_jarvis() -> tuple[bool, str]:
    if is_jarvis_running():
        stop_jarvis()
        time.sleep(0.5)

    return start_jarvis_hidden()


def get_status_message() -> str:
    jarvis_processes = find_jarvis_processes()
    tray_processes = find_tray_processes()

    jarvis_status = "running" if jarvis_processes else "not running"
    tray_status = "running" if tray_processes else "not running"

    return (
        f"Jarvis is {jarvis_status}. "
        f"Tray controller is {tray_status}."
    )


def stop_all() -> tuple[bool, str]:
    stop_jarvis()
    stop_tray()

    return True, "Jarvis and its tray controller were stopped."


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Manage the Jarvis background processes."
    )

    parser.add_argument(
        "command",
        choices=[
            "start",
            "stop",
            "restart",
            "status",
            "start-tray",
            "stop-tray",
            "stop-all",
        ],
    )

    arguments = parser.parse_args()

    if arguments.command == "start":
        _, message = start_jarvis_hidden()

    elif arguments.command == "stop":
        _, message = stop_jarvis()

    elif arguments.command == "restart":
        _, message = restart_jarvis()

    elif arguments.command == "status":
        message = get_status_message()

    elif arguments.command == "start-tray":
        _, message = start_tray_hidden()

    elif arguments.command == "stop-tray":
        _, message = stop_tray()

    else:
        _, message = stop_all()

    print(message)


if __name__ == "__main__":
    main()