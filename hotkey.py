#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在终端里不按回车就收单个按键。演示用它模拟"设备来了一条报警"。

为什么要自己写:input() 要等回车,而且会把整个线程堵死 —— 语音链路是常驻循环,
堵住就没法一边听麦克风一边等按键。这里做成一个后台线程,轮询式取键,
主循环完全不受影响。

Windows 和 POSIX 两条路:
  · Windows  msvcrt.kbhit()/getwch(),控制台天生就是单键模式,不用改任何状态
  · POSIX    终端默认是行缓冲(cooked),得先切到 cbreak 才收得到单个字符;
             切了就必须还原,否则程序退出后用户的 shell 不回显、不换行 ——
             所以退出路径全部走 finally

拿不到终端(输出被重定向、跑在没有 tty 的环境里)时不报错,只是静默不启动:
热键是演示用的辅助功能,不该因为它把主程序拦下来。
"""

from __future__ import annotations

import sys
import threading

POLL_SECONDS = 0.05  # 轮询间隔。按键响应要"跟手",50 ms 感觉不到延迟,CPU 也可忽略


def available() -> bool:
    """当前环境能不能收单键。"""
    try:
        if sys.platform == "win32":
            import msvcrt  # noqa: F401

            return True
        return sys.stdin.isatty()
    except Exception:
        return False


def _loop_windows(stop_event: threading.Event, on_key) -> None:
    import msvcrt

    while not stop_event.is_set():
        if not msvcrt.kbhit():
            stop_event.wait(POLL_SECONDS)
            continue
        ch = msvcrt.getwch()
        # 方向键之类是两次 getwch:先来一个 \x00 或 \xe0,再来扫描码。
        # 不把第二个字节吃掉的话,它会被当成一个普通字母触发误动作。
        if ch in ("\x00", "\xe0"):
            msvcrt.getwch()
            continue
        if ch == "\x03":  # Ctrl-C:控制台已经把信号送给主线程了,这里跟着退出
            break
        on_key(ch)


def _loop_posix(stop_event: threading.Event, on_key) -> None:
    import select
    import termios
    import tty

    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    try:
        # cbreak 而不是 raw:保留 ISIG,Ctrl-C 还能正常中断程序
        tty.setcbreak(fd)
        while not stop_event.is_set():
            if not select.select([sys.stdin], [], [], POLL_SECONDS)[0]:
                continue
            ch = sys.stdin.read(1)
            if not ch:  # stdin 到头了(被重定向的空输入)
                break
            on_key(ch)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def start(stop_event: threading.Event, on_key) -> threading.Thread | None:
    """起一个后台线程收键,每个键回调一次 on_key(ch)。收不到键就返回 None。

    线程是 daemon 的:主程序 Ctrl-C 退出时不等它 —— POSIX 那边它可能正堵在
    select 上,而终端状态的还原在 finally 里,进程退出时内核会把 tty 复位。
    """
    if not available():
        return None

    def run() -> None:
        try:
            if sys.platform == "win32":
                _loop_windows(stop_event, on_key)
            else:
                _loop_posix(stop_event, on_key)
        except Exception:
            # 收键失败不该带崩语音链路,静默退出就是了(功能降级已在启动时提示过)
            pass

    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t
