# tray_app.py

from __future__ import annotations

import ctypes
import sys
from typing import Any

import pystray
from PIL import Image, ImageDraw, ImageFont

from process_manager import (
    get_status_message,
    restart_jarvis,
    start_jarvis_hidden,
    stop_jarvis,
)


ERROR_ALREADY_EXISTS = 183


def prevent_duplicate_tray() -> None:
    mutex = ctypes.windll.kernel32.CreateMutexW(
        None,
        False,
        "Global\\JarvisTraySingleInstanceLock",
    )

    if not mutex:
        sys.exit(1)

    if ctypes.windll.kernel32.GetLastError() == ERROR_ALREADY_EXISTS:
        sys.exit(0)


def create_icon() -> Image.Image:
    image = Image.new("RGBA", (64, 64), (17, 24, 39, 255))
    draw = ImageDraw.Draw(image)

    draw.ellipse(
        (8, 8, 56, 56),
        fill=(14, 165, 233, 255),
        outline=(255, 255, 255, 255),
        width=2,
    )

    font = ImageFont.load_default()
    text = "J"

    text_box = draw.textbbox((0, 0), text, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    draw.text(
        (
            (64 - text_width) / 2,
            (64 - text_height) / 2 - 1,
        ),
        text,
        fill=(255, 255, 255, 255),
        font=font,
    )

    return image


def notify(icon: Any, message: str) -> None:
    try:
        icon.notify(message, "Jarvis")
    except Exception:
        pass


def start_jarvis_action(
    icon: Any,
    _item: Any,
) -> None:
    _, message = start_jarvis_hidden()
    notify(icon, message)


def stop_jarvis_action(
    icon: Any,
    _item: Any,
) -> None:
    _, message = stop_jarvis()
    notify(icon, message)


def restart_jarvis_action(
    icon: Any,
    _item: Any,
) -> None:
    _, message = restart_jarvis()
    notify(icon, message)


def status_action(
    icon: Any,
    _item: Any,
) -> None:
    notify(icon, get_status_message())


def close_tray_only(
    icon: Any,
    _item: Any,
) -> None:
    notify(icon, "Tray controller closed. Jarvis keeps running.")
    icon.stop()


def exit_everything(
    icon: Any,
    _item: Any,
) -> None:
    stop_jarvis()
    icon.stop()


def main() -> None:
    prevent_duplicate_tray()

    start_jarvis_hidden()

    icon = pystray.Icon(
        "Jarvis",
        create_icon(),
        "Jarvis Assistant",
        menu=pystray.Menu(
            pystray.MenuItem(
                "Start Jarvis",
                start_jarvis_action,
            ),
            pystray.MenuItem(
                "Stop Jarvis",
                stop_jarvis_action,
            ),
            pystray.MenuItem(
                "Restart Jarvis",
                restart_jarvis_action,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Show Status",
                status_action,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem(
                "Close Tray Only",
                close_tray_only,
            ),
            pystray.MenuItem(
                "Exit Jarvis Completely",
                exit_everything,
            ),
        ),
    )

    icon.run()


if __name__ == "__main__":
    main()