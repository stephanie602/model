#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows 控制台和进程内存的几件杂事。

原来的代码是给 macOS / Linux 写的,这三处在 Windows 上都没有对应实现:

  · ANSI 转义序列 —— 终端里的灰色临时结果、\r\033[2K 原地刷新全靠它。
    Windows 10 1511 之后的 conhost 支持,但默认关着,要自己 SetConsoleMode 打开。
    Windows Terminal / PowerShell 7 默认是开的,重复开一次也无害。
  · 中文输出 —— 控制台默认代码页是 936(GBK),Python 的 stdout 跟着走 GBK,
    遇到 emoji(🤖 前缀)直接 UnicodeEncodeError 把线程打死。统一改成 UTF-8。
  · 进程内存 —— Linux 读 /proc/self/statm,macOS 用 resource.getrusage;
    Windows 两个都没有(resource 模块整个不存在),走 psapi 的 GetProcessMemoryInfo。
"""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes

STD_OUTPUT_HANDLE = -11
STD_ERROR_HANDLE = -12
ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004


class _PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _mem_counters() -> _PROCESS_MEMORY_COUNTERS | None:
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        # K32GetProcessMemoryInfo 是 Win7+ 直接搬进 kernel32 的版本,
        # 免得再去 LoadLibrary 一个 psapi.dll
        fn = k32.K32GetProcessMemoryInfo

        # argtypes/restype 必须显式声明。不声明的话 ctypes 按 C int(32 位)处理,
        # 而 GetCurrentProcess() 返回的伪句柄是 (HANDLE)-1 —— 64 位下被截成 32 位
        # 再传进去就不是同一个值了,函数直接返回失败。表现是内存永远读出 0,
        # 不抛异常、不报错,只是数字不对。
        k32.GetCurrentProcess.argtypes = []
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        fn.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(_PROCESS_MEMORY_COUNTERS),
            wintypes.DWORD,
        ]
        fn.restype = wintypes.BOOL

        counters = _PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not fn(k32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return None
        return counters
    except (AttributeError, OSError):
        return None


def rss_now() -> int:
    """当前进程的物理内存占用(工作集)。拿不到就返回 0,调用方只是少打一行日志。"""
    c = _mem_counters()
    return int(c.WorkingSetSize) if c else 0


def rss_peak() -> int:
    """进程物理内存峰值。"""
    c = _mem_counters()
    return int(c.PeakWorkingSetSize) if c else 0


def setup_console() -> None:
    """打开 ANSI 转义 + 把 stdout/stderr 切成 UTF-8。程序一进 main 就调。"""
    try:
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        for handle_id in (STD_OUTPUT_HANDLE, STD_ERROR_HANDLE):
            h = k32.GetStdHandle(handle_id)
            mode = wintypes.DWORD()
            if k32.GetConsoleMode(h, ctypes.byref(mode)):
                k32.SetConsoleMode(
                    h, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                )
    except (AttributeError, OSError):
        pass  # 输出被重定向到文件时没有控制台句柄,不是错误

    for stream in (sys.stdout, sys.stderr):
        try:
            # errors="replace":宁可打个问号,也别让一个生僻字把打印线程炸掉
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
