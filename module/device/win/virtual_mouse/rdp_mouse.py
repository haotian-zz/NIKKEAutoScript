"""Session-bound RDP mouse input through an existing Interception installation.

Protocol: https://github.com/oblitum/Interception/blob/master/library/interception.c
Only TS_INPT\\TS_MOU devices receive reports. A zero-motion RAWINPUT marker
identifies the receiving session before any button or wheel operation is allowed.
"""

import ctypes
import os
import secrets
import struct
import time
from ctypes import wintypes

import win32api
import win32gui
import win32ts
import pywintypes

from module.device.win.virtual_mouse.driver_mouse import VirtualMouse
from module.logger import logger

_kernel = ctypes.WinDLL('kernel32', use_last_error=True)
_user = ctypes.WinDLL('user32', use_last_error=True)
_kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                              wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
_kernel.CreateFileW.restype = wintypes.HANDLE
_kernel.CreateEventW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.BOOL, wintypes.LPCWSTR]
_kernel.CreateEventW.restype = wintypes.HANDLE
_kernel.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                  ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
_kernel.DeviceIoControl.restype = wintypes.BOOL
_user.GetRawInputData.argtypes = [wintypes.HANDLE, wintypes.UINT, ctypes.c_void_p,
                                ctypes.POINTER(wintypes.UINT), wintypes.UINT]
_user.GetRawInputData.restype = wintypes.UINT

RDP_MOUSE_ID = 'TS_INPT\\TS_MOU'
IOCTL_EVENT = 0x222040
IOCTL_WRITE = 0x222080
IOCTL_HARDWARE_ID = 0x222200


class SessionMouseError(OSError):
    pass


class MousePacket(ctypes.Structure):
    _fields_ = [('unit', wintypes.WORD), ('flags', wintypes.WORD), ('buttons', wintypes.WORD),
                ('wheel', wintypes.WORD), ('raw', wintypes.DWORD), ('x', wintypes.LONG),
                ('y', wintypes.LONG), ('extra', wintypes.DWORD)]


class RawInputDevice(ctypes.Structure):
    _fields_ = [('page', wintypes.WORD), ('usage', wintypes.WORD), ('flags', wintypes.DWORD),
                ('hwnd', wintypes.HWND)]


_user.RegisterRawInputDevices.argtypes = [ctypes.POINTER(RawInputDevice), wintypes.UINT, wintypes.UINT]
_user.RegisterRawInputDevices.restype = wintypes.BOOL


def remote_session_id():
    value = wintypes.DWORD()
    if not _kernel.ProcessIdToSessionId(os.getpid(), ctypes.byref(value)):
        raise ctypes.WinError(ctypes.get_last_error())
    if _user.GetSystemMetrics(0x1000):
        return value.value
    if value.value == _kernel.WTSGetActiveConsoleSessionId():
        return None
    try:
        protocol = win32ts.WTSQuerySessionInformation(0, value.value, win32ts.WTSClientProtocolType)
    except pywintypes.error as exc:
        raise SessionMouseError('Cannot identify the input session; console input will not be used') from exc
    if protocol != 2:
        raise SessionMouseError('Input requires the active console or a verified RDP session')
    return value.value


def button_edges(previous, current):
    flags = 0
    for mask, down, up in ((1, 1, 2), (2, 4, 8), (4, 16, 32)):
        if bool(previous & mask) != bool(current & mask):
            flags |= down if current & mask else up
    return flags


def absolute_point(x, y, bounds):
    left, top, width, height = bounds
    if width <= 0 or height <= 0 or not left <= x < left + width or not top <= y < top + height:
        raise SessionMouseError(f'RDP mouse position outside desktop: {(x, y)}, bounds={bounds}')
    return round((x - left) * 65535 / width), round((y - top) * 65535 / height)


def select_session_device(received, candidates):
    matches = set(received).intersection(candidates)
    if len(matches) != 1:
        raise SessionMouseError(f'Expected one RDP mouse in this session; observed {sorted(matches)}')
    return matches.pop()


class _Channel:
    def __init__(self, index):
        self.index, self.handle, self.event = index, None, None
        handle = _kernel.CreateFileW(f'\\\\.\\interception{index:02d}', 0x80000000, 0, None, 3, 0, None)
        if handle == wintypes.HANDLE(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self.handle = handle

    def ioctl(self, code, data=None, result=None):
        size = wintypes.DWORD()
        ok = _kernel.DeviceIoControl(
            self.handle, code, ctypes.byref(data) if data is not None else None,
            ctypes.sizeof(data) if data is not None else 0,
            ctypes.byref(result) if result is not None else None,
            ctypes.sizeof(result) if result is not None else 0, ctypes.byref(size), None,
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return size.value

    def hardware_id(self):
        value = ctypes.create_unicode_buffer(1024)
        self.ioctl(IOCTL_HARDWARE_ID, result=value)
        return value.value

    def prepare(self):
        self.event = _kernel.CreateEventW(None, True, False, None)
        if not self.event:
            raise ctypes.WinError(ctypes.get_last_error())
        self.ioctl(IOCTL_EVENT, (wintypes.HANDLE * 2)(self.event, None))

    def write(self, packet):
        if self.ioctl(IOCTL_WRITE, packet) != ctypes.sizeof(packet):
            raise SessionMouseError('RDP mouse report was not fully accepted')

    def close(self):
        if self.handle is not None:
            _kernel.CloseHandle(self.handle)
            self.handle = None
        if self.event is not None:
            _kernel.CloseHandle(self.event)
            self.event = None


def _identify_session(channels):
    """The receiving desktop supplies the association; device numbers are never guessed."""
    received, callback_errors = set(), []
    marker = secrets.randbits(24) << 8
    header_size = 8 + 2 * ctypes.sizeof(ctypes.c_void_p)

    def procedure(hwnd, message, wparam, lparam):
        if message == 0xFF:
            try:
                size = wintypes.UINT()
                _user.GetRawInputData(lparam, 0x10000003, None, ctypes.byref(size), header_size)
                if header_size + 24 <= size.value <= 4096:
                    buffer = ctypes.create_string_buffer(size.value)
                    count = _user.GetRawInputData(lparam, 0x10000003, buffer, ctypes.byref(size), header_size)
                    if count == size.value and struct.unpack_from('<I', buffer.raw)[0] == 0:
                        extra = struct.unpack_from('<I', buffer.raw, header_size + 20)[0]
                        if extra & 0xFFFFFF00 == marker:
                            received.add(extra & 0xFF)
            except (OSError, pywintypes.error, ValueError, struct.error) as exc:
                callback_errors.append(str(exc))
        return win32gui.DefWindowProc(hwnd, message, wparam, lparam)

    cls = win32gui.WNDCLASS()
    cls.lpszClassName = f'NKAS_RdpMouse_{os.getpid()}_{marker:x}'
    cls.lpfnWndProc = procedure
    cls.hInstance = win32api.GetModuleHandle(None)
    win32gui.RegisterClass(cls)
    hwnd = None
    try:
        hwnd = win32gui.CreateWindow(cls.lpszClassName, '', 0, 0, 0, 0, 0, -3, 0, cls.hInstance, None)
        registration = RawInputDevice(1, 2, 0x100, hwnd)
        if not _user.RegisterRawInputDevices(ctypes.byref(registration), 1, ctypes.sizeof(registration)):
            raise ctypes.WinError(ctypes.get_last_error())
        for index, channel in channels.items():
            channel.write(MousePacket(extra=marker | index))
        deadline = time.monotonic() + 0.7
        while time.monotonic() < deadline:
            win32gui.PumpWaitingMessages()
            time.sleep(0.01)
        if callback_errors:
            raise SessionMouseError('; '.join(callback_errors))
        return select_session_device(received, channels)
    finally:
        if hwnd is not None:
            removal = RawInputDevice(1, 2, 1, None)
            _user.RegisterRawInputDevices(ctypes.byref(removal), 1, ctypes.sizeof(removal))
            win32gui.DestroyWindow(hwnd)
        win32gui.UnregisterClass(cls.lpszClassName, cls.hInstance)


class RdpMouseDevice:
    def __init__(self):
        self.channel = None
        self.session_id = None
        self.buttons = 0
        self.position = None

    @property
    def device_path(self):
        return f'RDP session {self.session_id}, Interception {self.channel.index}' if self.channel else None

    def open(self):
        if self.channel is not None:
            return True
        channels = {}
        try:
            self.session_id = remote_session_id()
            if self.session_id is None:
                raise SessionMouseError('RDP input cannot operate on the console desktop')
            for index in range(10, 20):
                candidate = None
                try:
                    candidate = _Channel(index)
                    if candidate.hardware_id() == RDP_MOUSE_ID:
                        candidate.prepare()
                        channels[index] = candidate
                        candidate = None
                except OSError:
                    pass
                finally:
                    if candidate is not None:
                        candidate.close()
            if not channels:
                raise SessionMouseError('No Interception RDP mouse found; console HID input will not be used')
            index = _identify_session(channels)
            self.channel = channels.pop(index)
            logger.info(f'RDP mouse bound without pointer movement: {self.device_path}')
            return True
        except (OSError, pywintypes.error) as exc:
            logger.error(f'RDP mouse binding failed: {exc}')
            return False
        finally:
            for channel in channels.values():
                channel.close()

    def close(self):
        if self.channel is not None:
            try:
                if self.buttons:
                    self.send(buttons=0)
            finally:
                self.channel.close()
                self.channel = None
                self.buttons, self.position = 0, None

    def _write(self, x, y, buttons, wheel=0):
        if self.channel is None and not self.open():
            return False
        try:
            if remote_session_id() != self.session_id:
                raise SessionMouseError('Input session changed; refusing to use the previous device')
            bounds = tuple(_user.GetSystemMetrics(index) for index in (76, 77, 78, 79))
            px, py = absolute_point(x, y, bounds)
            flags = button_edges(self.buttons, buttons)
            if wheel:
                flags |= 0x400
            packet = MousePacket(flags=3, buttons=flags, wheel=(int(wheel) * 120) & 0xFFFF, x=px, y=py)
            self.channel.write(packet)
            self.buttons, self.position = buttons, (x, y)
            return True
        except OSError as exc:
            logger.error(f'RDP mouse report failed: {exc}')
            return False

    def move_absolute(self, x, y):
        return self._write(int(x), int(y), self.buttons)

    def send(self, buttons=0, dx=0, dy=0, wheel=0):
        x, y = self.position if self.position is not None else win32api.GetCursorPos()
        return self._write(x + dx, y + dy, buttons, wheel)


class RdpMouse(VirtualMouse):
    session_local = True

    def __init__(self):
        super().__init__(driver=RdpMouseDevice())

    def set_cursor(self, x, y):
        if not self.driver.move_absolute(x, y):
            return False
        deadline = time.monotonic() + 0.1
        while time.monotonic() < deadline:
            actual = self.cursor()
            if abs(actual[0] - x) <= 2 and abs(actual[1] - y) <= 2:
                return True
            time.sleep(0.005)
        return False

    def move_to(self, x, y, buttons=0, tolerance=None):
        return self.set_cursor(x, y)
