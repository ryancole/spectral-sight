"""Where a window's client area sits inside its captured frame.

Graphics Capture delivers the window's visible bounds -- what
`DWMWA_EXTENDED_FRAME_BOUNDS` reports, title bar and the 1px border included --
and the game is drawn only in the client area inside them. On the kilrogg
receiver at 96 DPI that is a 2117x1354 frame around a 2115x1322 client area at
(1, 31). The offset is the client area's screen origin minus the frame's.

Both rectangles are read on a thread that is per-monitor DPI aware for the
duration, whatever the process is: `DWMWA_EXTENDED_FRAME_BOUNDS` is always in
physical pixels, but `GetClientRect` and `ClientToScreen` answer a DPI-unaware
caller in scaled ones, and mixing the two puts the game area wherever the
scale factor says rather than where the game is.

The maths is `client_area`, kept free of Win32 so it can be tested anywhere.
"""

from __future__ import annotations

import sys

from spectral_sight.types import GameArea


def client_area(
    frame_origin: tuple[int, int],
    client_origin: tuple[int, int],
    client_size: tuple[int, int],
    frame_size: tuple[int, int],
) -> GameArea:
    """The client rectangle in the pixels of the captured frame.

    `frame_origin` is the captured bounds' top-left on the screen,
    `client_origin` the client area's (`ClientToScreen` of (0, 0)),
    `client_size` the client area's size and `frame_size` the captured
    image's. The result is clipped to the image: a client area reaching past
    the captured bounds cannot be drawn into the frame anyway.
    """
    area = GameArea(
        client_origin[0] - frame_origin[0],
        client_origin[1] - frame_origin[1],
        client_size[0],
        client_size[1],
    )
    return area.clipped(*frame_size)


if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _dwmapi = ctypes.WinDLL("dwmapi")

    _DWMWA_EXTENDED_FRAME_BOUNDS = 9
    _PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

    _user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    _user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    _user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.RECT)]
    _user32.ClientToScreen.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.POINT)]
    _user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    _user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    _user32.IsWindowVisible.argtypes = [wintypes.HWND]
    _dwmapi.DwmGetWindowAttribute.argtypes = [
        wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
    ]
    _ENUM_PROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)


def find_window(title: str) -> int | None:
    """The first visible top-level window whose title contains `title`, in
    Z order, or None. Resolved here so that the window captured and the
    window measured are the same one."""
    if sys.platform != "win32":
        return None
    found: list[int] = []

    def visit(hwnd, _lparam):  # noqa: ANN001
        if not _user32.IsWindowVisible(hwnd):
            return True
        length = _user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        _user32.GetWindowTextW(hwnd, buffer, length + 1)
        if title in buffer.value:
            found.append(hwnd)
            return False
        return True

    _user32.EnumWindows(_ENUM_PROC(visit), 0)
    return found[0] if found else None


def window_game_area(hwnd: int, frame_size: tuple[int, int]) -> GameArea | None:
    """The client area of `hwnd` inside a captured frame of `frame_size`, or
    None if Windows would not say (the window went away, or no Win32)."""
    if sys.platform != "win32":
        return None
    previous = _user32.SetThreadDpiAwarenessContext(_PER_MONITOR_AWARE_V2)
    try:
        bounds = wintypes.RECT()
        if _dwmapi.DwmGetWindowAttribute(
            hwnd, _DWMWA_EXTENDED_FRAME_BOUNDS,
            ctypes.byref(bounds), ctypes.sizeof(bounds),
        ) != 0:
            return None
        client = wintypes.RECT()
        if not _user32.GetClientRect(hwnd, ctypes.byref(client)):
            return None
        origin = wintypes.POINT(0, 0)
        if not _user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return None
    finally:
        if previous:
            _user32.SetThreadDpiAwarenessContext(previous)
    area = client_area(
        (bounds.left, bounds.top),
        (origin.x, origin.y),
        (client.right - client.left, client.bottom - client.top),
        frame_size,
    )
    return area if area.width > 0 and area.height > 0 else None
