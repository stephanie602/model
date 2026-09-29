#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
麦克风 → AGC → 降噪 → VAD 分句 → ASR → 终端实时文字(可选:接 LLM 对话 + TTS 朗读)

Windows 版。整条链路:

    sounddevice(WASAPI) → AGC 增益 →┬→ [唤醒词闸] KWS
                                     └→ GTCRN 降噪 → Silero VAD → SenseVoice
                                                                      ↓
                           扬声器 ← Kokoro TTS ← llama-server(HTTP)

唤醒词吃的是未降噪的音频 —— GTCRN 会把弱语音当噪声削掉,对只有 3.3 M 参数的
KWS 是致命的;而 VAD 那一路恰恰离不开降噪。两边要的东西相反,所以在 AGC 之后分叉。

唤醒词(--wake,见 wakeword.py)是一道闸:没听到"你好小智"之前,后面
整条链路都不动,只有 3.3 MB 的 KWS 模型在跑,待机 CPU 能降一个数量级。

唤醒状态机的规矩(安全相关,改之前先看这里):
    未唤醒   KWS 只认唤醒词。"处理好了""全部解除"这些命令词一律当环境音,
             不进命令缓冲 —— 否则报警刚响时旁边一句闲聊就能把报警撤掉
    唤醒后   才开命令词 KWS / ASR 命令匹配;唤醒词之前说的话不算
    优先级   唤醒词 > 命令词。醒着时再喊一声唤醒词 = 重新开始,之前半句作废
    报警     不自动打开闸门(--alert-wake 可以恢复旧行为)

识别模型不是逐采样点输出的流式模型,所以这里做的是"低延迟分块识别":
用 VAD 把麦克风音频切成一句一句,说话中每隔 ~1.2 秒出一次临时结果(灰色,
原地刷新),检测到停顿后再对整句重新识别一次并定稿(白色,换行留下)。

语言:SenseVoice 覆盖中/英/日/韩/粤五种。方言只有粤语,四川话、上海话这些会按
普通话猜,错字会明显变多。

用法(PowerShell,先跑过 setup_windows.ps1):
    .venv\\Scripts\\python.exe realtime_asr.py                      # 实时麦克风识别
    .venv\\Scripts\\python.exe realtime_asr.py --list-devices       # 列出可用输入设备
    .venv\\Scripts\\python.exe realtime_asr.py --device 1           # 指定输入设备
    .venv\\Scripts\\python.exe realtime_asr.py --simulate test.wav  # 没有麦克风时用音频文件模拟
    .venv\\Scripts\\python.exe realtime_asr.py --no-partial         # 只要定稿结果,省一半算力
    .venv\\Scripts\\python.exe realtime_asr.py --denoise            # 噪声环境:GTCRN 降噪后再识别
    .venv\\Scripts\\python.exe realtime_asr.py --chat               # 识别 + LLM 回答(语音助手)
    .venv\\Scripts\\python.exe realtime_asr.py --no-wake            # 关掉唤醒词(不推荐:命令词会一直开着)
    .venv\\Scripts\\python.exe realtime_asr.py --chat --speak              # 待机省 CPU 的语音助手
    .venv\\Scripts\\python.exe realtime_asr.py --denoise --chat --speak    # 全语音对话
    .venv\\Scripts\\python.exe realtime_asr.py --speak --voice zm_yunxi    # 不开对话,直接朗读识别结果
    .venv\\Scripts\\python.exe realtime_asr.py --chat-url http://192.168.1.10:8080/v1 --chat
    .venv\\Scripts\\python.exe realtime_asr.py --no-machine        # 关掉包装机场景(报警 + 状态问答)

--chat 之前要先在另一个窗口起好 LLM server:.\\start_llm.ps1

包装机场景(默认开,见 machine.py):跑起来以后
    1 / 2   放第几条报警(缺原材料 / 传送带异物堵塞)
    3 / 4   解除对应的那一条,只解这一条
    空格键  下一条报警,按顺序循环(一路演下去用这个)
    R 键    全部解除
    直接问  "现在温度和速度是多少""今天完成了多少个包装" —— 用真实状态回答,不经 LLM

唤醒词默认是**开**的:先喊"你好小智"(或者按回车)再说命令。报警响了也一样要先叫醒 ——
这正是为了挡住旁边人一句"处理好了"把报警误解除。--no-wake 可以关掉,但命令词会一直开着。

Ctrl-C 退出。
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from collections import deque
from math import gcd
from pathlib import Path

import numpy as np

import gain
import hotkey
import intent as intent_mod
import machine as machine_mod
import manual as manual_mod
import vad as vad_mod
import wakeword
import winutil

MODEL_SAMPLE_RATE = 16000  # 识别模型和 GTCRN 都固定 16 kHz,整条链路一个采样率走到底
BLOCK_SECONDS = 0.1  # 麦克风回调粒度

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"  # setup_windows.ps1 把模型都放这儿
# 语音起点前多留一段音频再送去识别,避免吃掉第一个字。
# 0.5 而不是 0.3:VAD 判"开始说话"本身有延迟(Silero 一窗 32 ms,再加上
# 0.1 秒的块粒度),而中文第一个字的声母往往就那么几十毫秒。留短了的表现是
# "打开电灯"识别成"开电灯" —— 命令词匹配那边能捞回来,但不如一开始就别丢。
PRE_ROLL_SECONDS = 0.5
MIN_DECODE_SECONDS = 1.0
# 反问"都处理好了吗"之后,多久没等到应答就当他没回答(秒)
CONFIRM_TIMEOUT = 30.0  # 模型对 <1s 的音频不稳定,不足则补零

# 触发阈值 = 底噪 x 3,但夹在这两个值之间:
# 下限防止安静房间里一点风扇声就误触发;上限保证正常音量的说话一定能触发
# (否则一开口就有声音时,底噪估计会被说话声本身抬高,阈值跟着涨到永远触发不了)。
THRESHOLD_MIN = 0.006
THRESHOLD_MAX = 0.05

# 唤醒词文案和应答词,给提示语和 text 模式那条路用。main() 按参数填进去
WAKE_HINT = ["你好小智"]
WAKE_REPLY = ["你好，我在"]

CLEAR_LINE = "\r\033[2K"
DIM = "\033[2m"
RESET = "\033[0m"


def degrade(what: str, why: str) -> None:
    """某个模块起不来时打一行,然后继续跑。

    所有模块现在默认开启,硬失败就意味着"少一个模型,整个程序都跑不起来"——
    对一个语音助手来说太脆了。缺什么就少什么,但必须说清楚少了什么、为什么,
    否则用户会以为它在正常工作。
    """
    print(f"{DIM}[跳过] {what}: {why}{RESET}")


def load_paths_env() -> dict[str, str]:
    """读 setup_windows.ps1 写的 models\\paths.env,拿模型路径当默认值。

    没有这个文件也能跑(所有值都能用命令行参数覆盖),只是要自己写全路径。
    """
    env: dict[str, str] = {}
    f = MODELS_DIR / "paths.env"
    if not f.is_file():
        return env
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if v:
            env[k.strip()] = v
    return env


# --------------------------------------------------------------------------- #
# 内存统计
# --------------------------------------------------------------------------- #
def gb(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def print_mem(label: str) -> None:
    print(f"{DIM}[内存] {label}: 工作集当前 {gb(winutil.rss_now())}"
          f" / 峰值 {gb(winutil.rss_peak())}{RESET}")


# --------------------------------------------------------------------------- #
# 音频输入
# --------------------------------------------------------------------------- #
def pick_input_samplerate(device, prefer: int = MODEL_SAMPLE_RATE) -> int:
    """优先按 16 kHz 采集(模型要的采样率,省一次重采样);设备不支持就退回它的默认值。

    Windows 上很多 USB 麦克风和板载声卡通过 MME/WASAPI 报出来的是 44.1 或 48 kHz,
    拿不到 16 kHz 是常态 —— 那就采完再重采样,只有 --denoise 那条路会被卡住
    (GTCRN 固定 16 kHz,见下面 main() 里的检查)。
    """
    import sounddevice as sd

    info = sd.query_devices(device, "input")
    candidates = [prefer, MODEL_SAMPLE_RATE, int(info["default_samplerate"])]
    for sr in candidates:
        try:
            sd.check_input_settings(
                device=device, samplerate=sr, channels=1, dtype="float32"
            )
            return sr
        except Exception:
            continue
    raise RuntimeError(
        f"输入设备 {info['name']} 不支持 {candidates} 里任何一个采样率的单声道采集"
    )


def mic_blocks(device, samplerate: int, stop_event: threading.Event,
               overflow: list[int] | None = None):
    """从麦克风持续产出 0.1 秒的单声道 float32 音频块(设备原始采样率)。

    overflow 给一个单元素列表的话,采集丢块的次数会累加到 overflow[0]。
    """
    import sounddevice as sd

    overflow = overflow if overflow is not None else [0]

    audio_q: "queue.Queue[np.ndarray]" = queue.Queue()
    blocksize = int(BLOCK_SECONDS * samplerate)

    def callback(indata, frames, time_info, status):
        # input overflow = 驱动那边的缓冲被写满了,这一段音频是真的丢了(不是延迟)。
        # 原来这里完全不看 status,丢了也一声不吭 —— 表现出来就是"话说了一半没识别到",
        # 和说话太轻、离得太远的症状一模一样,根本分不出是哪一种。所以现在记一笔。
        if status and status.input_overflow:
            overflow[0] += 1
        audio_q.put(indata[:, 0].copy())

    with sd.InputStream(
        device=device,
        samplerate=samplerate,
        channels=1,
        dtype="float32",
        blocksize=blocksize,
        callback=callback,
    ):
        while not stop_event.is_set():
            try:
                yield audio_q.get(timeout=0.5)
            except queue.Empty:
                continue


def mic_check(device, sr: int, seconds: float, wake_model, keywords: list[str],
              save: str | None = None) -> int:
    """录一段,把采集质量一次量完:电平、底噪、信噪比、削顶、丢块、唤醒词命中。

    存在的理由:识别不准的原因有好几种,症状却长得一模一样 —— 说话太轻、离得太远、
    噪声太大、驱动丢块,在终端上都表现为"话说了没反应"。一条条试要花掉一下午。
    这个命令一次把四个量都测出来,调完麦克风位置再跑一遍就知道有没有变好。
    """
    import sounddevice as sd

    print(f"录 {seconds:.0f} 秒。请用平时说话的音量、平时的距离,")
    print(f"反复说「{keywords[0]}」,中间正常停顿。\n")
    for i in (3, 2, 1):
        print(f"  {i} ...", end="\r", flush=True)
        time.sleep(1)
    print("  开始说话!        ")

    overflow = [0]
    stop = threading.Event()
    agc = gain.AutoGain()
    ww = None
    if wake_model:
        try:
            ww = wakeword.WakeWord(wake_model, keywords=keywords)
        except Exception as exc:
            print(f"{DIM}(唤醒词模型加载失败,这次只测电平: {exc}){RESET}")

    raw_blocks: list[np.ndarray] = []
    hits = 0
    t0 = time.time()
    for block in mic_blocks(device, sr, stop, overflow):
        raw_blocks.append(block.copy())
        gained = agc(resample_to_model_rate(block, sr))
        if ww is not None and ww.accept(gained):
            hits += 1
            print(f"  [{time.time() - t0:4.1f}s] 检出「{keywords[0]}」x{hits}")
        if time.time() - t0 >= seconds:
            stop.set()
            break
    stop.set()

    raw = np.concatenate(raw_blocks) if raw_blocks else np.zeros(1, np.float32)
    raw16 = resample_to_model_rate(raw, sr)
    if save:
        import soundfile as sf

        sf.write(save, raw16, MODEL_SAMPLE_RATE, subtype="PCM_16")
        print(f"\n{DIM}原始录音已存到 {save}(没加增益,可以直接听){RESET}")

    peak = float(np.abs(raw).max())
    snr = agc.snr_db
    print("\n===== 麦克风体检 =====")
    print(f"输入峰值   {peak:.3f}      (正常 0.1-0.4;<0.06 偏低,AGC 补得回来但不如硬件调好)")
    print(f"底噪 RMS   {agc.noise:.4f}")
    print(f"信噪比     {snr:.0f} dB" if snr is not None else "信噪比     没听到人说话")
    print(f"AGC 补到   {agc.gain:.1f}x" + (f"   削顶 {agc.clipped_blocks} 块" if agc.clipped_blocks else ""))
    print(f"采集丢块   {overflow[0]} 次" + ("" if not overflow[0] else "   ← 音频真的丢了,不是延迟"))
    if ww is not None:
        print(f"唤醒词     {seconds:.0f} 秒里检出 {hits} 次")

    print("\n结论:")
    bad = False
    if peak < 0.06:
        bad = True
        print(f"  [!] 电平低。系统 → 声音 → 输入,音量拉到 80-100;"
              f"设备属性 → 级别里把「麦克风加强」开到 +20 dB。")
    if agc.clipped_blocks > len(raw_blocks) * 0.02:
        bad = True
        print(f"  [!] 削顶了。系统输入音量或者麦克风加强调低一档,离麦克风远一点。")
    if overflow[0]:
        bad = True
        print(f"  [!] 采集丢块。CPU 被抢光了 —— LLM 的 -Threads 调小,"
              f"或者加 --no-partial / --num-threads 2。")
    hint = gain.snr_hint(snr)
    if hint:
        bad = True
        print(f"  [!] {hint}")
    if ww is not None and hits == 0:
        bad = True
        print(f"  [!] 一次都没检出。先确认上面几项;都正常的话把录音存下来自己听一遍:"
              f"\n      --mic-check --save-check check.wav")
    if not bad:
        print("  各项正常。识别还是不准的话,把录音存下来听一遍是最快的办法:"
              "\n      --mic-check --save-check check.wav")
    return 0


def load_audio_file(path: str, sr: int) -> np.ndarray:
    """读音频文件成单声道 float32,重采样到 sr。

    走 soundfile —— 只认 wav/flac/ogg 这类,mp3/m4a 得先自己转。
    --simulate 只是个验证工具,不值得为它再拖一个 ffmpeg 依赖。
    """
    import soundfile as sf

    audio, file_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio[:, 0])  # 取第一个声道
    if file_sr != sr:
        from scipy.signal import resample_poly

        g = gcd(int(file_sr), int(sr))
        audio = resample_poly(audio, sr // g, int(file_sr) // g).astype(np.float32)
    return audio


def file_blocks(
    path: str,
    stop_event: threading.Event,
    sr: int = MODEL_SAMPLE_RATE,
    realtime: bool = True,
):
    """把音频文件按 0.1 秒切块产出,可按真实时间节奏喂入,用来在没有麦克风时验证整条链路。"""
    audio = load_audio_file(path, sr)
    blocksize = int(BLOCK_SECONDS * sr)
    # 末尾补 1.5 秒静音,好让最后一句触发定稿
    audio = np.concatenate([audio, np.zeros(int(1.5 * sr), np.float32)])

    for start in range(0, len(audio), blocksize):
        if stop_event.is_set():
            return
        block = audio[start : start + blocksize]
        if len(block) < blocksize:
            block = np.pad(block, (0, blocksize - len(block)))
        if realtime:
            time.sleep(BLOCK_SECONDS)
        yield block


def resample_to_model_rate(audio: np.ndarray, sr: int) -> np.ndarray:
    """整段一次性重采样到 16 kHz —— 不在块边界上做,避免每 0.1 秒一次滤波器瞬态。"""
    if sr == MODEL_SAMPLE_RATE:
        return audio
    from scipy.signal import resample_poly

    g = gcd(sr, MODEL_SAMPLE_RATE)
    return resample_poly(audio, MODEL_SAMPLE_RATE // g, sr // g).astype(np.float32)


# --------------------------------------------------------------------------- #
# 解码任务调度
# --------------------------------------------------------------------------- #
class JobBoard:
    """
    单槽 partial + FIFO final。

    临时结果只保留最新的一个:解码比说话慢时,过时的 partial 直接被覆盖丢弃,
    这样延迟不会越积越多;定稿任务一个都不能丢,所以单独排队。
    """

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._partial: tuple[int, np.ndarray] | None = None
        self._finals: deque[tuple[int, np.ndarray, list[str]]] = deque()
        self._closed = False

    def put_partial(self, seg_id: int, audio: np.ndarray) -> None:
        with self._cv:
            self._partial = (seg_id, audio)
            self._cv.notify()

    def put_final(self, seg_id: int, audio: np.ndarray,
                  commands: list[str] | None = None) -> None:
        """commands 是这一段里 KWS 直接听出来的命令词(见 --command-kws)。
        有它的话下游就不靠 ASR 的文字去认命令了 —— 噪声大时 ASR 那条路先垮。"""
        with self._cv:
            self._finals.append((seg_id, audio, commands or []))
            self._cv.notify()

    def close(self) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()

    def get(self) -> tuple[str, int, np.ndarray, list[str]] | None:
        """定稿优先。返回 None 表示收工。"""
        with self._cv:
            while not self._finals and self._partial is None and not self._closed:
                self._cv.wait()
            if self._finals:
                seg_id, audio, cmds = self._finals.popleft()
                return ("final", seg_id, audio, cmds)
            if self._partial is not None:
                seg_id, audio = self._partial
                self._partial = None
                return ("partial", seg_id, audio, [])
            return None


# --------------------------------------------------------------------------- #
# 识别
# --------------------------------------------------------------------------- #
def save_segment(save_dir: Path, seg_id: int, audio: np.ndarray, text: str) -> str:
    """把一句定稿音频存成 wav。文件名带识别结果,好在文件管理器里直接对照哪句错了。"""
    import re

    import soundfile as sf

    # Windows 文件名不能有 \ / : * ? " < > |,识别结果里出现标点很正常
    safe = re.sub(r'[\\/:*?"<>|]', "", text)[:24] or "空"
    path = save_dir / f"{seg_id:03d}_{safe}.wav"
    sf.write(str(path), audio, MODEL_SAMPLE_RATE, subtype="PCM_16")
    return path.name


def decoder_loop(
    board: JobBoard,
    rec,
    transcript: list[str],
    show_mem: bool = False,
    chat=None,
    print_lock: threading.Lock | None = None,
    speaker=None,
    timing: bool = False,
    save_dir: Path | None = None,
    text_gate=None,
    matcher=None,
    machine=None,
    wake=None,
    wake_once: bool = False,
    manual=None,
    wake_text=None,
) -> None:
    lock = print_lock or threading.Lock()

    def emit(s: str) -> None:
        # 回答是另一个线程在流式打印,这里必须拿同一把锁,否则两边的字会交错
        with lock:
            sys.stdout.write(s)
            sys.stdout.flush()

    # 反问"都处理好了吗"之后,等工人那一声应答。存成一元列表是为了在闭包里改它
    awaiting_confirm = [0.0]

    def sleep_gates() -> None:
        """一句唤醒对应一条指令:这条处理完就睡回去,不等 --wake-timeout。

        车间里这样更稳 —— 醒着的每一秒,旁边的说话都可能被当成命令;
        而且"下一句还是不是说给助手听的"只有用户自己知道,不该由超时来猜。
        """
        if not wake_once:
            return
        slept = False
        for gate in (wake, text_gate):
            if gate is not None and gate.awake:
                gate.sleep()
                slept = True
        if slept:
            emit(f"{DIM}[睡眠] 这条处理完了。再叫一声「{WAKE_HINT[0]}」或者按回车{RESET}\n")

    last_final_seg = -1
    while True:
        job = board.get()
        if job is None:
            break
        kind, seg_id, audio, kws_cmds = job
        if kind == "partial" and seg_id <= last_final_seg:
            continue  # 这句已经定稿了,迟到的临时结果直接扔掉
        # 太短的音频模型不稳定,补零到 1 秒
        if len(audio) < int(MIN_DECODE_SECONDS * MODEL_SAMPLE_RATE):
            audio = np.pad(
                audio, (0, int(MIN_DECODE_SECONDS * MODEL_SAMPLE_RATE) - len(audio))
            )
        t_asr = time.time()
        try:
            text = rec.transcribe(audio, partial=(kind == "partial"))
        except Exception as exc:  # 单句失败不该让整个程序退出
            emit(f"{CLEAR_LINE}[识别失败] {exc}\n")
            continue
        asr_seconds = time.time() - t_asr
        if kind == "final":
            last_final_seg = max(last_final_seg, seg_id)
            if text:
                transcript.append(text)
                emit(f"{CLEAR_LINE}{text}\n")
            else:
                emit(CLEAR_LINE)
            if timing:
                # RTF < 1 才跟得上实时。这个数接近或超过 1 说明 ASR 本身就是瓶颈,
                # 得加 --num-threads 或者关掉临时结果腾 CPU。
                dur = len(audio) / MODEL_SAMPLE_RATE
                emit(
                    f"{DIM}[计时] ASR {asr_seconds:.2f}s / 音频 {dur:.1f}s"
                    f" · RTF {asr_seconds / max(dur, 1e-9):.2f}{RESET}\n"
                )
            if save_dir is not None:
                try:
                    name = save_segment(save_dir, seg_id, audio, text)
                    emit(f"{DIM}[存盘] {name}{RESET}\n")
                except Exception as exc:  # 存盘失败不该影响识别本身
                    emit(f"{DIM}[存盘失败] {exc}{RESET}\n")
            # 先过唤醒闸门,再认命令 —— 顺序不能反。
            #
            # 原来是先把整句替换成命令词、再拿去找唤醒词:"你好小智,处理好了"被
            # 换成"处理好了"之后唤醒词就没了,text 模式根本叫不醒;而"处理好了,
            # 你好小智"这种唤醒词**前面**的命令反倒跟着执行了。
            # 现在的规矩:没醒 → 整句只当环境音;唤醒词之前的话一律不算,
            # 命令只从唤醒词后面那部分里找。
            command = text
            passed = True
            if text_gate is not None and text:
                was_awake = text_gate.awake
                passed, command = text_gate.feed_text(text)
                if not passed:
                    # 没唤醒:这句只当环境音,识别结果照样打出来,但不惊动 LLM 和 TTS
                    emit(f"{DIM}(未唤醒,忽略){RESET}\n")
                elif not was_awake:
                    emit(f"{DIM}[唤醒] {WAKE_REPLY[0]},请说{RESET}\n")
                    if speaker is not None and not command:
                        speaker.interrupt()
                        speaker.say(WAKE_REPLY[0])
                if not passed or not was_awake:
                    # 这一句是睡着时开始说的,KWS 听到的命令分不清在唤醒词前还是后,
                    # 宁可不要(segment_loop 那边睡着时本来就不喂命令 KWS,这里是兜底)
                    kws_cmds = []
            elif text_gate is not None:
                passed = text_gate.awake  # ASR 没出字,只剩 KWS 的命令:没醒就不认

            # 醒着的时候又喊了一遍唤醒词:照样应答,别把它当成一条命令。
            #
            # kws 模式下 segment_loop 醒着时也在听唤醒词,听到就把那半句扔了;
            # 这里接的是它漏掉、但 ASR 认出来的那几次。唤醒词优先:
            # 它前面的话(包括 KWS 听到的命令)作废,只留后面的。
            if passed and wake_text is not None and command:
                hit, rest = wake_text.detector.match(command)
                if hit:
                    kws_cmds = []
                    for gate in (wake, text_gate):
                        # 又叫了一声说明人还在,把超时重新计时,别马上睡回去
                        if gate is not None:
                            gate.awake = True
                            if hasattr(gate, "idle"):
                                gate.idle = 0.0
                            if hasattr(gate, "_last"):
                                gate._last = time.time()
                    if not rest.strip():
                        emit(f"{CLEAR_LINE}{DIM}[唤醒] {WAKE_REPLY[0]},请说{RESET}\n")
                        if speaker is not None and WAKE_REPLY[0]:
                            speaker.interrupt()
                            speaker.say(WAKE_REPLY[0])
                        continue
                    command = rest

            if not passed:
                command = ""
            elif kws_cmds or (matcher is not None and command):
                # 两条路各有各的强项,合并而不是二选一:
                #   ASR + 拼音匹配   噪声小时更全 —— 它看得到整句,一句里说了
                #                    几条命令都能切出来
                #   KWS 直接听音频   噪声大时唯一还站着的 —— 实测 5 dB 下
                #                    ASR 那条 0/2、这条 2/2
                # 早期版本是"有 KWS 就不看 ASR",结果在噪声不大的段落里反而丢命令:
                # ASR 明明听全了两条,却被只听到一条的 KWS 覆盖掉。
                heard = command  # 唤醒词后面那部分;没装闸门就是整句
                asr_cmds = [h.command.text for h in matcher.match_all(heard)] if (
                    matcher is not None and heard) else []
                merged = list(asr_cmds)
                for c in kws_cmds:  # KWS 多听到的补进去,顺序放在后面
                    if c not in merged:
                        merged.append(c)
                if merged:
                    command = "，".join(merged)
                    if command != heard:
                        src = []
                        if asr_cmds:
                            src.append(f"ASR「{heard}」")
                        if kws_cmds:
                            src.append(f"KWS{kws_cmds}")
                        emit(f"{DIM}[命令] {command}   ← {' + '.join(src)}{RESET}\n")
                    if transcript:
                        transcript[-1] = command
                    else:
                        transcript.append(command)

            # 报警挂着的时候,先看这句是不是"故障处理好了"。工人手上正忙着
            # (在上料、在掏传送带),回来按键不现实,说一句就该把报警撤掉。
            # 排在状态问答前面:"料加好了"里也有"料"字,别被当成问产量
            res = machine_mod.voice_clear(
                command, machine, confirming=bool(awaiting_confirm[0])
            ) if (machine is not None and command) else None
            if res is not None and res.ask:
                # 挂着好几条,工人只说了句笼统的"处理好了" —— 先问清楚。
                # 直接全清太危险(可能他只修了一条),逼他一条条念又太烦
                q = machine_mod.confirm_question(res.ask)
                awaiting_confirm[0] = time.time()
                emit(f"{CLEAR_LINE}[确认] {q}   {DIM}← 语音「{command}」{RESET}\n")
                if speaker is not None:
                    speaker.interrupt()
                    speaker.say(q)
                if chat is not None:
                    chat.note(command, q)
                continue
            if res is not None and res.cleared:
                awaiting_confirm[0] = 0.0
                speech = machine_mod.cleared_speech(res.cleared)
                names = "、".join(a.title for a in res.cleared)
                emit(f"{CLEAR_LINE}[解除] {names}   {DIM}← 语音「{command}」{RESET}\n")
                emit(f"{DIM}       {machine.status_text()}{RESET}\n")
                if speaker is not None:
                    speaker.interrupt()
                    speaker.say(speech)
                if chat is not None:
                    chat.note(command, speech)
                sleep_gates()
                continue
            # 没接住这句就把"等确认"撤掉 —— 工人已经说别的了
            if awaiting_confirm[0] and time.time() - awaiting_confirm[0] > CONFIRM_TIMEOUT:
                awaiting_confirm[0] = 0.0

            # 问设备状态的先在本地答掉,不进 LLM:答案是温度/产量这些真实数字,
            # 让模型转述只会多一道出错的机会,而且省掉一整轮生成(快一两秒)。
            # 对不上的问题(“这个温度正常吗”)返回 None,照旧交给 LLM ——
            # 它的系统提示里已经带了同一份状态,见 machine.system_context()。
            parts = machine_mod.answer_parts(command, machine) if (
                machine is not None and command) else None
            status = "".join(parts) if parts else None
            if status:
                emit(f"{CLEAR_LINE}[状态] {status}\n")
                if speaker is not None:
                    speaker.interrupt()
                    # 分段念:第一段是固定前缀("当前封口温度"),开机就预合成好了,
                    # 立刻出声;它在播的时候,带数字的下一段正好在后面合成。
                    # 整句一次合成的话要 5 秒多才响第一声
                    for part in parts:
                        speaker.say(part)
                if chat is not None:
                    # 让 LLM 知道这一轮问过什么、答了什么,后面才接得上
                    # “那正常吗”“还能撑多久”这类追问
                    chat.note(command, status)
            elif chat is not None and command:
                # 手册问答:先去手册里找。找到了就把那几段连同页码交给 LLM,
                # 并且约束它"只依据手册、要报页码、没写就说没写" ——
                # 设备手册答错比答不出来危险得多,工人真会照着做。
                # 找不到(BM25 分数够不着)就当普通问题,该聊什么聊什么。
                ctx, pages = manual.context(command) if manual is not None else ("", [])
                if ctx:
                    emit(f"{DIM}[手册] 命中第 {'、'.join(str(p) for p in pages)} 页,"
                         f"按手册内容回答{RESET}\n")
                    # 200 token 够说 40 个字加页码;闲聊那条路仍然是 --chat-max-tokens
                    chat.submit(command, manual_mod.system_prompt(ctx), max_tokens=200)
                else:
                    chat.submit(command)
            elif speaker is not None and command:
                # 没开 --chat 时 --speak 就是复读机:用来单独验证 TTS 和回声闸门
                speaker.interrupt()
                speaker.say(command)
            sleep_gates()
            if show_mem:
                head = f"第 {seg_id} 句 {len(audio) / MODEL_SAMPLE_RATE:.1f}s"
                body = f"工作集 {gb(winutil.rss_now())} / 峰值 {gb(winutil.rss_peak())}"
                emit(f"{DIM}[内存] {head} → {body}{RESET}\n")
        else:
            emit(f"{CLEAR_LINE}{DIM}{text}{RESET}")


# --------------------------------------------------------------------------- #
# 能量 VAD 分句
# --------------------------------------------------------------------------- #
def segment_loop(
    blocks,
    board: JobBoard,
    sr: int,
    args,
    speaker=None,
    wake=None,
    text_gate=None,
    print_lock: threading.Lock | None = None,
    denoiser=None,
    vad=None,
    cmd_kws=None,
) -> None:
    block_dur = BLOCK_SECONDS
    lock = print_lock or threading.Lock()

    # 降噪只接在 VAD/ASR 这一路上,唤醒词吃的是没降噪的音频。
    # 实测 GTCRN 会把唤醒词吃掉:三段确认说了"你好小智"的录音,不降噪 3/3 检出,
    # 降噪后 0/3;电平健康的那段 60 秒录音也从 15 次掉到 14、7 次(看和 AGC 的先后)。
    # 原因是 GTCRN 按"压噪声"训练,遇到弱语音会连语音一起削 —— 对只有 3.3 M 参数、
    # 本来就靠那点声学证据的 KWS 是致命的。
    # 而 VAD 那一路恰恰离不开它:噪声环境下底噪 RMS 0.02,阈值被上限钳死,不降噪根本切不出句子。
    # 两边要的东西相反,那就别共用一条流。GTCRN 每块照跑(RTF 0.02,可以忽略),
    # 只是唤醒词不吃它的输出 —— 流式模型的状态必须连续,不能只在醒着的时候喂。
    dn_stream = denoiser.stream(sr) if denoiser is not None else None

    def emit(s: str) -> None:
        with lock:
            sys.stdout.write(s)
            sys.stdout.flush()
    noise: float | None = None
    announced = False

    pre_roll_len = getattr(args, "pre_roll", PRE_ROLL_SECONDS)
    pre_roll: deque[np.ndarray] = deque(maxlen=max(1, int(pre_roll_len / block_dur)))
    speech: list[np.ndarray] = []
    in_speech = False
    seg_id = 0
    speech_dur = 0.0
    silence_dur = 0.0
    last_partial_dur = 0.0

    seg_cmds: list[str] = []

    def gate_open() -> bool:
        """现在能不能接命令。没装唤醒闸门就一直能;装了就看醒没醒。"""
        if wake is not None:
            return wake.awake
        if text_gate is not None:
            return text_gate.awake
        return True

    was_open = gate_open()

    def drop_pending() -> None:
        """闸门一开一关,都把这之前攒的东西全扔掉:半句话、命令 KWS 听到的词、
        命令 KWS 解了一半的解码状态。

        睡着时听到的任何一个字都不许带进醒来之后 —— 原来就是这里漏的:
        报警刚响,旁边有人说"处理好了",命令 KWS 记下了;随后有人喊"你好小智",
        醒来后第一句的定稿把那条"处理好了"一起捎上,报警被没人确认过的话解除了。
        """
        nonlocal speech, in_speech, speech_dur, silence_dur, last_partial_dur
        pre_roll.clear()
        speech = []
        in_speech = False
        speech_dur = silence_dur = last_partial_dur = 0.0
        seg_cmds.clear()
        if cmd_kws is not None:
            cmd_kws.reset()

    def flush() -> None:
        nonlocal seg_id, in_speech, speech, speech_dur, silence_dur, last_partial_dur
        audio = resample_to_model_rate(np.concatenate(speech), sr)
        board.put_final(seg_id, audio, list(seg_cmds))
        seg_cmds.clear()
        seg_id += 1
        in_speech = False
        speech = []
        speech_dur = 0.0
        silence_dur = 0.0
        last_partial_dur = 0.0

    for raw_block in blocks:
        # 降噪流的状态必须连续,所以每块都推 —— 哪怕这块马上要被闸掉
        block = dn_stream.push(raw_block) if dn_stream is not None else raw_block

        if speaker is not None and speaker.blocking_mic():
            # 喇叭正在响,麦克风这会儿录到的基本是助手自己的声音。不闸掉的话它会被
            # 识别成用户说的话、再触发一次回答,自己跟自己没完没了地聊。
            # 半句状态也一并清掉,免得播放前后的两截音频被拼成一句。
            pre_roll.clear()
            speech = []
            in_speech = False
            speech_dur = silence_dur = last_partial_dur = 0.0
            continue

        # 闸门可能被别的线程开关:回车 / 报警(wake_now)、一条指令处理完睡回去
        # (sleep_gates)。不管谁动的,状态一变就清场,见 drop_pending
        now_open = gate_open()
        if now_open != was_open:
            drop_pending()
            was_open = now_open

        # 有没有人在说话。silero 看频谱结构,energy 比能量 —— 见 vad.py。
        # 睡眠期间照样判:energy 那条路要靠它维护底噪估计,不然醒来第一块拿到的是
        # 唤醒词的说话声,阈值当场被顶到上限,命令的开头几个字会被啃掉。
        if hasattr(vad, "in_speech"):
            vad.in_speech = in_speech
        voiced = vad(block)
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        noise = getattr(vad, "noise", None)
        threshold = getattr(vad, "threshold", 0.0)

        if wake is not None:
            # 唤醒词 KWS 睡着醒着都听,而且排在命令词前面:唤醒词优先级最高。
            #   睡着  只有 3.3 MB 的 KWS 模型在跑,SenseVoice 一次都不解,命令 KWS 也不喂
            #   醒着  又喊了一声 = 重新开始,之前半句话和听到的命令全作废
            #         ("处理好了……不对,你好小智,把温度报一下"只执行后半句)
            # 醒着多喂这一路 RTF 才 0.012,换来的是唤醒词永远压得住命令词。
            was_awake = wake.awake
            hit = wake.feed(raw_block)  # 未降噪的那一份,见上面 dn_stream 的注释
            if not hit:
                if not was_awake:
                    continue
            else:
                tag = "唤醒" if not was_awake else "重新唤醒"
                emit(f"{CLEAR_LINE}{DIM}[{tag}]「{hit}」{WAKE_REPLY[0]},请说{RESET}\n")
                # 预滚里装的是唤醒词本身,留着会被识别成命令的一部分;
                # 命令 KWS 从这一刻起才开始听,别让它带着唤醒词之前的解码状态
                drop_pending()
                was_open = True
                if speaker is not None and args.wake_reply:
                    speaker.interrupt()
                    speaker.say(args.wake_reply)
                continue

        # 命令词也交给 KWS 直接从音频听 —— 不经过 ASR。实测同一段混音:
        # 信噪比 5 dB 时 SenseVoice 只剩"小打开关闭",两条命令一条都对不上;
        # KWS 两条全中。ASR 要在几千个字里解出"你说了什么",KWS 只要回答
        # "有没有出现这几个词",搜索空间小几个数量级,噪声里剩下那点证据就够用。
        # 喂未降噪的音频,理由和唤醒词一样(见上面 dn_stream 的注释)。
        #
        # 只在醒着时听。睡着时"处理好了""全部解除"就是普通的环境说话,
        # 不进命令缓冲 —— 否则旁边一句闲聊就能把报警撤掉。
        # text 模式下这也意味着:喊醒它的那一句里的命令词 KWS 不算,
        # 要等下一句(那一句里 ASR 切出来的唤醒词之后的部分照样算)。
        if cmd_kws is not None and gate_open():
            hit = cmd_kws.accept(raw_block)
            if hit:
                seg_cmds.append(hit)
                emit(f"{CLEAR_LINE}{DIM}[命令·KWS] {hit}{RESET}\n")

        if wake is not None and wake.tick(voiced or in_speech):
            # 醒着但一直没人说话:睡回去,别让 ASR 白白挂在那儿等
            if in_speech and speech:
                flush()
            emit(f"{CLEAR_LINE}{DIM}[休眠] {args.wake_timeout:.0f}s 没人说话,"
                 f"再叫一声「{args.wake_word[0]}」{RESET}\n")
            continue

        if not announced:
            announced = True
            if noise is not None:
                print(
                    f"底噪估计 RMS≈{noise:.5f} → 触发阈值≈{threshold:.5f}"
                    f"(运行中会自动跟随环境噪声,也可以用 --threshold 固定)"
                )
            print(f"{DIM}开始说话吧(Ctrl-C 退出){RESET}\n")

        if not in_speech:
            pre_roll.append(block)
            if voiced:
                in_speech = True
                # 有人开口了:让 TTS 停下后台备货,把 CPU 让给识别,
                # 也免得等下要出声时排在一条备货后面(实测能差 1.8 秒)
                if speaker is not None:
                    speaker.note_activity()
                speech = list(pre_roll)
                pre_roll.clear()
                speech_dur = len(speech) * block_dur
                silence_dur = 0.0
                last_partial_dur = 0.0
            continue

        speech.append(block)
        speech_dur += block_dur
        silence_dur = 0.0 if voiced else silence_dur + block_dur

        if silence_dur >= args.silence:
            flush()
        elif speech_dur >= args.max_segment:
            # 一直不停顿(念稿、朗读)时强制切句,避免单次解码越来越慢
            flush()
        elif (
            args.partial_interval > 0
            # text 模式下还没被唤醒:临时结果没人看,解了纯属浪费(定稿那一次
            # 还是要解的 —— 唤醒词本身就在里面)
            and (text_gate is None or text_gate.awake)
            and speech_dur >= MIN_DECODE_SECONDS
            and speech_dur - last_partial_dur >= args.partial_interval
        ):
            board.put_partial(seg_id, resample_to_model_rate(np.concatenate(speech), sr))
            last_partial_dur = speech_dur

    if in_speech and speech:
        flush()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def list_devices() -> int:
    import sounddevice as sd

    devices = sd.query_devices()
    inputs = [(i, d) for i, d in enumerate(devices) if d["max_input_channels"] > 0]
    print(devices)
    print()
    if not inputs:
        print("没有找到任何音频输入设备。")
        return 1
    print("可用输入设备:")
    for i, d in inputs:
        print(f"  --device {i}   {d['name']}  ({int(d['default_samplerate'])} Hz)")
    print(
        f"\n{DIM}同一个物理麦克风通常会出现好几次(MME / Windows DirectSound / WASAPI)。"
        f"优先挑 WASAPI 那一条:延迟最低,而且更容易协商到 16 kHz。{RESET}"
    )
    return 0


NO_MIC_HELP = """
找不到音频输入设备,无法采集麦克风。

先确认系统层面认得到(这一步跟 Python 无关):
  1. 设置 → 系统 → 声音 → 输入,确认设备在列,对着说话能看到音量条在动
  2. 设置 → 隐私和安全性 → 麦克风,把"麦克风访问"和"让桌面应用访问麦克风"
     都打开 —— 关着的时候 sounddevice 看得到设备,录出来却是一整条静音,
     不报任何错,最难查的就是这种
  3. 设备管理器里确认没有黄色感叹号(常见于免驱 USB 声卡插在 USB3 口上)

如果系统里能录音但这里看不到:
  .venv\\Scripts\\python.exe realtime_asr.py --list-devices
  用 --device <编号> 显式指定,优先选名字里带 WASAPI 的那条。

想先验证识别链路本身,可以用音频文件模拟实时输入:
  .venv\\Scripts\\python.exe realtime_asr.py --simulate test.wav
"""


def main() -> int:
    winutil.setup_console()

    from assistant import DEFAULT_CHAT_URL

    env = load_paths_env()

    p = argparse.ArgumentParser(
        description="麦克风 → 降噪 → VAD → ASR → 终端实时文字(可接 LLM 对话 + TTS)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--asr-model",
        default=None,
        help="SenseVoice 模型目录,默认读 models\\paths.env",
    )
    p.add_argument(
        "--asr-provider",
        default=None,
        help="onnxruntime 执行提供器,默认 cpu。自己编了 DirectML/CUDA 版才需要改",
    )
    p.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="给 ASR 用几个线程。LLM 也在抢 CPU,别开满",
    )
    p.add_argument(
        "--language",
        default="Chinese",
        help="识别语言(Chinese / Cantonese / English / Japanese / Korean);传 auto 让模型自己判断。"
        "仅 SenseVoice 有效,zipformer 那个模型是纯中文的",
    )
    p.add_argument(
        "--hotwords",
        default=None,
        metavar="FILE",
        help="热词文件,解码时给里面的词加分(仅 transducer 模型)。"
        "固定命令词场景务必用上 —— 见 hotwords.txt",
    )
    p.add_argument(
        "--hotwords-score",
        type=float,
        default=2.0,
        help="热词加分。不够就加不动,过头会凭空冒词(底噪被硬解成命令)",
    )
    p.add_argument("--device", default=None, help="输入设备编号或名称")
    p.add_argument("--list-devices", action="store_true", help="列出音频设备后退出")
    p.add_argument(
        "--mic-check",
        nargs="?",
        type=float,
        const=10.0,
        default=None,
        metavar="秒",
        help="麦克风体检:录一段(默认 10 秒),一次量完电平/底噪/信噪比/削顶/丢块/唤醒命中。"
        "识别不准时先跑这个 —— 这几种原因症状一样,一条条试要花掉一下午",
    )
    p.add_argument(
        "--save-check",
        default=None,
        metavar="WAV",
        help="把 --mic-check 录的原始音频存下来(不加增益),自己听一遍最直接",
    )
    p.add_argument("--simulate", metavar="WAV", default=None, help="用音频文件模拟实时麦克风输入")
    p.add_argument(
        "--silence",
        type=float,
        default=0.7,
        help="多长的停顿算一句话结束(秒)。这一段是纯等待,直接算进「说完到出声」,"
        "所以很想调小 —— 但试过 0.55,噪声环境下识别明显变差:噪声本来就让 VAD "
        "判断发飘,再把容忍的停顿缩短,一句话被从中间切成两半,"
        "两个半句各自送去识别,出来的字自然对不上。这 0.15 秒不值得省,别再动它",
    )
    p.add_argument("--max-segment", type=float, default=15.0, help="单句最长时长,超过就强制切(秒)")
    p.add_argument(
        "--partial-interval",
        type=float,
        default=1.2,
        help="说话过程中出临时结果的间隔(秒),0 表示不出临时结果",
    )
    p.add_argument("--no-partial", action="store_true", help="关掉临时结果,只输出定稿(更省算力)")
    p.add_argument(
        "--denoise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="先降噪再送 VAD 和识别(GTCRN)。噪声环境下主要是救分句"
        "(能量 VAD 在底噪高时会被钳死);对识别文字本身提升有限,详见 denoise.py",
    )
    p.add_argument(
        "--denoise-model",
        default=None,
        help="GTCRN 模型文件,默认读 models\\paths.env",
    )
    p.add_argument(
        "--chat",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="每句识别定稿后交给 LLM 生成回答。要先起好 server(.\\start_llm.ps1)",
    )
    p.add_argument(
        "--chat-model",
        default=None,
        help="传给 server 的模型名,默认 local(llama-server 不校验这个字段)",
    )
    p.add_argument(
        "--chat-url",
        default=None,
        help=f"OpenAI 兼容 server 的地址,默认 {DEFAULT_CHAT_URL}。LLM 放别的机器上就改这里",
    )
    p.add_argument(
        "--chat-api-key",
        default=None,
        help="server 需要鉴权时给。llama-server 默认不用",
    )
    p.add_argument(
        "--chat-max-tokens",
        type=int,
        default=60,
        help="单次回答的最大 token 数。合成比实时慢,回答长一倍就要多听一倍 —— "
        "这个上限是兜底,真正管长度的是系统提示里那句「不超过 15 个字」",
    )
    p.add_argument(
        "--chat-system",
        default=None,
        help="覆盖系统提示。默认告诉模型输入来自语音识别、没标点、可能有同音错字,回答要短",
    )
    p.add_argument(
        "--speak",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="把回答念出来(Kokoro-82M)。没开 --chat 时改成朗读识别结果",
    )
    p.add_argument(
        "--tts-model",
        default=None,
        help="Kokoro 模型目录,默认读 models\\paths.env",
    )
    p.add_argument(
        "--voice",
        default=None,
        help="音色,默认 zf_xiaoxiao(女声);男声用 zm_yunxi。全部音色见 tts.py --list-voices",
    )
    p.add_argument(
        "--speak-speed",
        type=float,
        default=1.15,
        help="朗读语速倍率。1.15 听着还自然,但音频短了 13%%,合成也跟着快 13%% —— "
        "合成慢的时候这是白捡的",
    )
    p.add_argument(
        "--tts-threads",
        type=int,
        default=4,
        help="Kokoro 合成用几个线程。合成慢的表现是"
        "「文字早打完了、声音还在后面挤」,那就把这个调大;开 --timing 看 TTS 的 RTF",
    )
    p.add_argument(
        "--tts-min-clause",
        type=int,
        default=None,
        metavar="N",
        help="回答攒够 N 个字才在逗号处切一句去合成(默认 12)。直接影响"
        "「说完到出声」的延迟:调到 6 出声明显更快,代价是句子偏碎、韵律断一点",
    )
    p.add_argument(
        "--sync-text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回答的文字跟着声音打,而不是跟着 LLM 打。合成比实时慢时,"
        "边生成边打会出现「屏幕上早写完了、喇叭还在念第一句」;"
        "开着它每句等真出声了才打出来。关掉就是老行为(文字先行)",
    )
    p.add_argument(
        "--chat-filler",
        nargs="*",
        default=[],
        metavar="词",
        help="回答之前先念的应答词(“好的”“嗯”),轮着用,默认不用 —— "
        "它能让 0 秒就有回应,但那是拿废话填时间,正文反而更晚。"
        "要的话:--chat-filler 好的， 我看一下，",
    )
    p.add_argument(
        "--tts-first-chunk",
        type=int,
        default=None,
        metavar="N",
        help="回答的第一小段攒够 N 个字就先去合成,不等标点(默认 8)。"
        "优先在逗号处切,模型一口气写到底时才按字数硬切。"
        "合成时间对长度有断崖,第一段短能让声音早好几秒出来;"
        "代价是这一刀可能切在词中间。设 0 关掉",
    )
    p.add_argument("--output-device", default=None, help="扬声器设备编号或名称")
    p.add_argument(
        "--tail-guard",
        type=float,
        default=None,
        metavar="秒",
        help="助手念完之后再多静音几秒(默认 0.6)。它自己的话被识别成你的命令就调大;"
        "念完要等太久才能接话就调小",
    )
    p.add_argument(
        "--no-mic-gate",
        action="store_true",
        help="朗读时不静音麦克风。默认是静音的 —— 否则喇叭的声音会被录回去,"
        "识别成用户说话,助手自己跟自己聊起来。开了这个就得戴耳机",
    )
    p.add_argument(
        "--save-segments",
        default=None,
        metavar="DIR",
        help="把每句定稿的音频存成 wav 到这个目录(16 kHz 单声道),文件名带识别结果。"
        "识别不准时用它抓现场:拿真实录音去离线复现和调参,比对着麦克风反复试有效得多",
    )
    p.add_argument(
        "--gain",
        default="auto",
        help="麦克风增益:auto=自动(AGC,默认) / off=不处理 / 一个数字=固定倍数。"
        "板载麦克风电平常常低到 VAD 和唤醒词都够不着,见 gain.py",
    )
    p.add_argument(
        "--bare",
        action="store_true",
        help="只留最基本的识别:唤醒词/降噪/命令词/对话/朗读全关。"
        "排查问题时用 —— 先确认识别本身是好的,再一层层加回来",
    )
    p.add_argument(
        "--wake",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="开唤醒词:说「你好小智」之前不跑识别。省 CPU 的主力开关 —— "
        "睡眠时只有 3.3 MB 的 KWS 模型在跑,SenseVoice 完全停着。"
        "车间里一直有人说话时尤其要开,否则旁边的闲聊也会进识别。"
        "嫌叫醒麻烦就 --no-wake,或者按回车代替喊唤醒词",
    )
    p.add_argument(
        "--wake-word",
        nargs="+",
        default=[wakeword.DEFAULT_KEYWORD],
        metavar="词",
        help="唤醒词,可以给多个。挑四个字、发音别太常见的,两个字的误唤醒会很多",
    )
    p.add_argument(
        "--wake-mode",
        choices=("kws", "text"),
        default="kws",
        help="kws=专用唤醒模型(省 CPU);text=拿识别结果匹配文字(不用下模型,但省不了 ASR)",
    )
    p.add_argument(
        "--wake-model",
        default=None,
        help="KWS 模型目录,默认在 models\\ 里找 sherpa-onnx-kws-*",
    )
    p.add_argument(
        "--wake-keywords-file",
        default=None,
        help="自己写好的 keywords 文件(格式照抄 KWS 模型目录里的 keywords.txt),"
        "给了就不再从 --wake-word 转",
    )
    p.add_argument(
        "--wake-threshold",
        type=float,
        default=wakeword.DEFAULT_THRESHOLD,
        help="唤醒检出阈值。误唤醒多就往 0.4 调,叫不醒就往 0.15 调",
    )
    p.add_argument(
        "--wake-score",
        type=float,
        default=wakeword.DEFAULT_SCORE,
        help="唤醒词解码加分,和 ASR 热词一个意思",
    )
    p.add_argument(
        "--wake-once",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="叫醒一次只接一条指令,处理完立刻睡回去。"
        "关掉(--no-wake-once)就是连续对话:醒着的这段时间里可以一直说,"
        "--wake-timeout 秒没人说话才睡",
    )
    p.add_argument(
        "--wake-timeout",
        type=float,
        default=15.0,
        help="唤醒后多久没人说话就睡回去(秒)。一问一答之间不用重复叫醒",
    )
    p.add_argument(
        "--wake-reply",
        default="你好，我在",
        help="被唤醒时念一句应答(要 --speak)。设成空串就不出声",
    )
    p.add_argument(
        "--alert-wake",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="设备报警时自动打开唤醒闸门,不用喊唤醒词就能接着说。默认关:"
        "报警刚响时旁边一句「处理好了」就会把报警误解除",
    )
    p.add_argument(
        "--wake-energy-gate",
        action="store_true",
        help="开唤醒词前面那层能量闸:安静的块连特征都不提,KWS 那 1.2%% 的 CPU 还能再省七成。"
        "默认关 —— 跳过块会打乱流的对齐,实测同一段录音开着会漏检",
    )
    p.add_argument("--mem", action="store_true", help="每句定稿后打印一次内存占用(结尾总是会打印峰值)")
    p.add_argument(
        "--timing",
        action="store_true",
        help="打印每一环的耗时(ASR / LLM 首字 / 首句交给 TTS)。嫌延迟大时先开这个看瓶颈在哪,"
        "别凭感觉调参数",
    )
    p.add_argument(
        "--pre-roll",
        type=float,
        default=PRE_ROLL_SECONDS,
        metavar="秒",
        help="语音起点前多送多少音频给识别(默认 0.5)。识别结果老是少开头第一个字就调大",
    )
    p.add_argument(
        "--commands",
        nargs="?",
        const="hotwords.txt",
        default="hotwords.txt",
        metavar="FILE",
        help="命令表(默认 hotwords.txt)。识别定稿后按拼音对到表上,"
        "把同音错字捞回来 —— 噪声环境下的固定命令词场景必开。见 intent.py",
    )
    p.add_argument(
        "--no-commands",
        dest="commands",
        action="store_const",
        const=None,
        help="关掉命令词匹配",
    )
    p.add_argument(
        "--command-kws",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="命令词直接用 KWS 从音频里听,不经过 ASR。噪声大时抗噪多 10-15 dB:"
        "实测信噪比 5 dB 下 ASR 那条路 0/2、KWS 2/2。要 --commands 提供命令表",
    )
    p.add_argument(
        "--command-kws-threshold",
        type=float,
        default=wakeword.DEFAULT_COMMAND_THRESHOLD,
        help="命令 KWS 的检出阈值。比唤醒词保守 —— 命令一命中就真去执行,"
        "误触发代价大。实测 0.2-0.3 零误报,再高会开始漏",
    )
    p.add_argument(
        "--command-threshold",
        type=float,
        default=intent_mod.DEFAULT_THRESHOLD,
        help="命令词匹配阈值(归一化编辑距离)。调大更容易命中但会误匹配",
    )
    p.add_argument(
        "--manual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="设备手册问答:问到手册里写着的事(横封不牢、报警含义、参数参考值),"
        "就检索手册内容让 LLM 照着答,并报出页码。"
        "要先入库一次:manual.py --build。见 manual.py",
    )
    p.add_argument(
        "--machine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="包装机场景:设备状态问答(温度/速度/今日产量,直接用真实数字回答,不经 LLM)"
        "+ 空格键模拟设备主动报警。见 machine.py",
    )
    p.add_argument(
        "--vad",
        choices=("silero", "energy"),
        default="silero",
        help="用什么判断有没有人在说话。silero=神经网络(640 KB,噪声环境下必须用这个);"
        "energy=纯能量阈值(不用模型,安静环境够用,噪声一大就把句子切碎)。见 vad.py",
    )
    p.add_argument(
        "--vad-threshold",
        type=float,
        default=None,
        help="silero 的判定阈值 0-1(默认 0.5)。咬掉句子开头就调低,把噪声当人声就调高",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="语音能量阈值(RMS),默认用开头 1 秒的环境噪声自动校准",
    )
    args = p.parse_args()

    if args.bare:
        # 每一层都可能是"识别不准"的嫌疑人,先全关掉,确认底座是好的
        args.wake = args.denoise = args.chat = args.speak = args.command_kws = False
        args.commands = None
        args.machine = False

    if args.no_partial:
        args.partial_interval = 0.0
    language = None if args.language.lower() in ("auto", "none", "") else args.language

    if args.list_devices:
        return list_devices()

    if args.mic_check is not None:
        import sounddevice as sd

        dev = args.device
        if dev is not None and str(dev).lstrip("-").isdigit():
            dev = int(dev)
        try:
            check_sr = pick_input_samplerate(dev, MODEL_SAMPLE_RATE)
            print(f"输入设备: {sd.query_devices(dev, 'input')['name']}  采集 {check_sr} Hz")
        except Exception as exc:
            print(f"打开输入设备失败: {exc}")
            print(NO_MIC_HELP)
            return 1
        kws_dir = args.wake_model or env.get("SHERPA_KWS_DIR")
        if kws_dir and not Path(kws_dir).is_absolute():
            kws_dir = str(MODELS_DIR / kws_dir)
        kws_dir = kws_dir or wakeword.find_kws_model(MODELS_DIR)
        return mic_check(dev, check_sr, args.mic_check, kws_dir,
                         args.wake_word, args.save_check)

    # 整条链路都走 16 kHz:GTCRN 和 SenseVoice 都是这个采样率,连重采样都省了
    prefer_sr = MODEL_SAMPLE_RATE

    # 先确认音频输入可用,再去加载模型 —— 别让用户等完模型才发现没麦克风
    sr = prefer_sr
    if args.simulate is None:
        try:
            import sounddevice as sd

            device = args.device
            if device is not None and str(device).lstrip("-").isdigit():
                device = int(device)
            if device is None and sd.default.device[0] < 0:
                print(NO_MIC_HELP)
                return 1
            sr = pick_input_samplerate(device, prefer_sr)
            info = sd.query_devices(device, "input")
            print(f"输入设备: {info['name']}  采集采样率: {sr} Hz")
        except Exception as exc:
            print(f"打开输入设备失败: {exc}")
            print(NO_MIC_HELP)
            return 1
    else:
        device = None
        print(f"模拟模式: 按真实时间节奏读入 {args.simulate}({sr} Hz)")

    if args.denoise and sr != MODEL_SAMPLE_RATE:
        # GTCRN 只吃 16 kHz,而流式那条路是逐块处理的,没法在中间插重采样(块长会错位)
        degrade("降噪", f"设备只能给 {sr} Hz,GTCRN 固定 16 kHz。"
                        f"用 --list-devices 换一条 16 kHz 的输入(WASAPI 那条通常可以)")
        args.denoise = False

    try:
        agc = gain.parse_gain(args.gain)
    except ValueError as exc:
        print(exc)
        return 1

    denoiser = None
    if args.denoise:
        t0 = time.time()
        from denoise import SherpaDenoiser

        model_path = args.denoise_model or str(
            MODELS_DIR / env.get("SHERPA_DENOISE_MODEL", "gtcrn_simple.onnx")
        )
        try:
            print(f"加载 GTCRN {model_path} ...")
            denoiser = SherpaDenoiser(model_path, num_threads=1)
            print(
                f"降噪就绪,用时 {time.time() - t0:.1f}s"
                f"({denoiser.sr} Hz,帧移 {denoiser.frame_shift} 采样点)"
            )
        except Exception as exc:
            degrade("降噪", f"{exc}")
            denoiser = None

    from asr_sherpa import SherpaRecognizer

    asr_dir = args.asr_model or env.get("SHERPA_ASR_DIR")
    if not asr_dir:
        print(
            "没指定 ASR 模型。跑一下 setup_windows.ps1 下载,或者用 "
            "--asr-model 指定 SenseVoice 模型目录。"
        )
        return 1
    if not Path(asr_dir).is_absolute():
        asr_dir = str(MODELS_DIR / asr_dir)
    print(f"加载模型 {asr_dir} ...")
    t0 = time.time()
    # 热词文件给相对路径时按脚本目录找,不按当前工作目录 —— 从别处调用时才不会踩空
    hotwords = args.hotwords
    if hotwords and not Path(hotwords).is_absolute():
        local = HERE / hotwords
        if local.is_file():
            hotwords = str(local)

    rec = SherpaRecognizer(
        asr_dir,
        language=language,
        num_threads=args.num_threads,
        provider=args.asr_provider,
        hotwords_file=hotwords,
        hotwords_score=args.hotwords_score,
    )
    print(
        f"模型就绪,用时 {time.time() - t0:.1f}s"
        f"(provider={rec.provider},{args.num_threads} 线程)"
    )
    print(f"  {rec.describe()}")
    print_mem("权重加载后")

    # 先跑一次假音频,把 onnx 图预热的一次性开销放在这里,别让第一句话背锅
    print("预热 ...")
    t0 = time.time()
    warmup = (np.arange(MODEL_SAMPLE_RATE, dtype=np.float32) % 7 - 3) * 1e-4
    rec.transcribe(warmup)
    print(f"预热完成,用时 {time.time() - t0:.1f}s")
    print_mem("预热后")

    head = (
        f"语言: 中文(zipformer)"
        if rec.kind == "transducer"
        else f"语言: {rec.language or '自动检测'}(SenseVoice)"
    )

    matcher = None
    if args.commands:
        cmd_file = args.commands
        if not Path(cmd_file).is_absolute():
            local = HERE / cmd_file
            if local.is_file():
                cmd_file = str(local)
        try:
            matcher = intent_mod.Matcher.from_file(cmd_file, args.command_threshold)
            print(f"命令词: {matcher.describe()}")
        except (FileNotFoundError, RuntimeError) as exc:
            degrade("命令词匹配", str(exc).splitlines()[0])
            matcher = None

    try:
        vad = vad_mod.make_vad(
            args.vad, MODELS_DIR,
            args.vad_threshold if args.vad == "silero" else args.threshold,
        )
    except FileNotFoundError as exc:
        # 退回能量 VAD 而不是退出。安静环境它够用,噪声环境会把句子切碎 ——
        # 所以这条提示要写清楚,别让用户以为一切正常
        degrade("Silero VAD", f"{str(exc).splitlines()[0]} → 退回能量 VAD(噪声环境下会切碎句子)")
        vad = vad_mod.make_vad("energy", MODELS_DIR, args.threshold)
    print(f"VAD: {vad.describe()}")

    # 唤醒词:kws 模式插在 VAD 前面(睡着时 ASR 一次都不解),
    # text 模式插在 ASR 后面(识别照跑,只是不唤醒就不往 LLM/TTS 送)
    wake = None
    text_gate = None
    wake_label = "关"
    if args.wake and args.wake_mode == "kws":
        model_dir = args.wake_model or env.get("SHERPA_KWS_DIR")
        if model_dir and not Path(model_dir).is_absolute():
            model_dir = str(MODELS_DIR / model_dir)
        # paths.env 里没写(老版本 setup 装的)就自己在 models\ 底下扫一遍
        model_dir = model_dir or wakeword.find_kws_model(MODELS_DIR)
        if sr != MODEL_SAMPLE_RATE:
            # KWS 是流式的,逐块处理,中间插重采样会让块长错位 —— 和 --denoise 同一个理由
            degrade("唤醒词", f"设备只能给 {sr} Hz,KWS 固定 16 kHz。"
                              f"换 16 kHz 输入,或者用 --wake-mode text")
        elif not model_dir:
            degrade("唤醒词", "没找到 KWS 模型,跑一下 setup_windows.ps1")
        else:
            t0 = time.time()
            print(f"加载唤醒词模型 {Path(model_dir).name} ...")
            try:
                detector = wakeword.WakeWord(
                    model_dir,
                    keywords=args.wake_word,
                    threshold=args.wake_threshold,
                    score=args.wake_score,
                    num_threads=1,  # 这一路是常驻的,一个线程足够(RTF≈0.012)
                    energy_gate=args.wake_energy_gate,
                    block_seconds=BLOCK_SECONDS,
                    keywords_file=args.wake_keywords_file,
                )
                wake = wakeword.WakeGate(
                    detector, timeout=args.wake_timeout, block_seconds=BLOCK_SECONDS
                )
                print(f"唤醒词就绪,用时 {time.time() - t0:.1f}s")
                print(f"  {detector.describe()}")
                wake_label = f"{'、'.join(args.wake_word)}(kws)"
            except (ValueError, RuntimeError, FileNotFoundError) as exc:
                degrade("唤醒词", str(exc).splitlines()[0])
    elif args.wake:
        try:
            detector = wakeword.TextWakeWord(args.wake_word)
            text_gate = wakeword.TextWakeGate(detector, timeout=args.wake_timeout)
            print(f"唤醒词(text 模式): {detector.describe()}")
            wake_label = f"{'、'.join(args.wake_word)}(text)"
        except RuntimeError as exc:
            degrade("唤醒词", str(exc).splitlines()[0])

    if args.wake and wake is None and text_gate is None and args.wake_mode == "kws":
        # KWS 唤醒没起来,不能就这么变成"没有闸门":那样命令词一直开着,
        # 睡眠状态禁止执行命令的规矩整个失效。退到 text 模式,至少闸门还在
        try:
            text_gate = wakeword.TextWakeGate(
                wakeword.TextWakeWord(args.wake_word), timeout=args.wake_timeout
            )
            print(f"唤醒词退回 text 模式: {text_gate.detector.describe()}")
            wake_label = f"{'、'.join(args.wake_word)}(text,KWS 没起来)"
        except RuntimeError as exc:
            degrade("唤醒词(text 兜底)", str(exc).splitlines()[0])
    if not args.wake or (wake is None and text_gate is None):
        print(f"{DIM}[!] 没有唤醒闸门:命令词一直在听,旁边的闲聊也可能被当成命令执行{RESET}")

    # 醒着时再喊唤醒词也要能认出来 —— 这一路只看文字,不占 CPU
    wake_text = text_gate
    if wake_text is None and (args.wake or text_gate is not None):
        try:
            wake_text = wakeword.TextWakeGate(
                wakeword.TextWakeWord(args.wake_word), timeout=args.wake_timeout
            )
        except RuntimeError:
            wake_text = None  # 没装 pypinyin 就算了,KWS 那一路照常

    cmd_kws = None
    if args.command_kws:
        kws_dir = args.wake_model or env.get("SHERPA_KWS_DIR")
        if kws_dir and not Path(kws_dir).is_absolute():
            kws_dir = str(MODELS_DIR / kws_dir)
        kws_dir = kws_dir or wakeword.find_kws_model(MODELS_DIR)
        if matcher is None:
            degrade("命令 KWS", "没有命令表(--commands)")
            kws_dir = None
        elif sr != MODEL_SAMPLE_RATE:
            degrade("命令 KWS", f"设备只能给 {sr} Hz,KWS 固定 16 kHz")
            kws_dir = None
        elif not kws_dir:
            degrade("命令 KWS", "没找到 KWS 模型,跑一下 setup_windows.ps1")
        t0 = time.time()
        try:
            cmd_kws = None if not kws_dir else wakeword.WakeWord(
                kws_dir,
                keywords=[c.text for c in matcher.commands],
                threshold=args.command_kws_threshold,
                num_threads=1,
                block_seconds=BLOCK_SECONDS,
            )
        except (ValueError, RuntimeError, FileNotFoundError) as exc:
            degrade("命令 KWS", str(exc).splitlines()[0])
            cmd_kws = None
        if cmd_kws is not None:
            print(f"命令 KWS 就绪,用时 {time.time() - t0:.1f}s")
            print(f"  {cmd_kws.describe()}")

    manual = None
    if args.manual:
        try:
            manual = manual_mod.Manual()
            print(f"{manual.describe()}")
        except (FileNotFoundError, KeyError, ValueError) as exc:
            # 手册没入库不该拦住整条语音链路 —— 其余功能一样能用
            degrade("手册问答", str(exc).splitlines()[0])

    machine = machine_mod.Machine() if args.machine else None
    if machine is not None:
        print(f"包装机: {machine.status_text()}")

    chat = None
    chat_label = "关"
    assistant = None
    if args.chat:
        from assistant import DEFAULT_SYSTEM, ChatWorker, HttpAssistant

        t0 = time.time()
        url = args.chat_url or DEFAULT_CHAT_URL
        print(f"连接 LLM server {url} ...")
        assistant = HttpAssistant(
            url,
            # llama-server 不校验这个字段,填什么都行;真名以 probe() 问到的为准
            model=args.chat_model or "local",
            system=args.chat_system or DEFAULT_SYSTEM,
            max_tokens=args.chat_max_tokens,
            api_key=args.chat_api_key,
            # 设备状态每次请求现取:模型手里有真数字,才不会去编温度和产量
            context=(lambda: machine_mod.system_context(machine)) if machine else None,
        )
        # 现在就探一下,别等用户说完第一句才发现 server 没起
        try:
            served = assistant.probe()
        except RuntimeError as exc:
            # server 没起就退出的话,"只想试试识别"这个最常见的用法就废了。
            # 语音那半条链路和 LLM 完全解耦,少了它照样能出文字。
            degrade("LLM 对话", f"连不上 {url} —— 另开一个窗口跑 .\\start_llm.ps1 再来")
            args.chat = False
            assistant = None
            served = None
        # 用户显式指定了就听他的(比如一个 server 挂了多个模型)
        if assistant is not None and not args.chat_model:
            assistant.model = served
        if assistant is not None:
            chat_label = assistant.model
            print(f"LLM server 就绪,用时 {time.time() - t0:.1f}s(模型: {served})")

    board = JobBoard()
    transcript: list[str] = []
    stop_event = threading.Event()
    print_lock = threading.Lock()

    speaker = None
    if args.speak:
        from tts import DEFAULT_VOICE, Speaker

        voice = args.voice or DEFAULT_VOICE
        tts_model = args.tts_model or str(
            MODELS_DIR / env.get("SHERPA_TTS_DIR", "matcha-icefall-zh-baker")
        )

        out_device = args.output_device
        if out_device is not None and str(out_device).lstrip("-").isdigit():
            out_device = int(out_device)

        def tts_error(msg: str) -> None:
            with print_lock:
                sys.stdout.write(f"{CLEAR_LINE}{msg}\n")
                sys.stdout.flush()

        print(f"加载 TTS {tts_model} ...")
        t0 = time.time()
        try:
            speaker = Speaker(
                model_path=tts_model,
                voice=voice,
                speed=args.speak_speed,
                device=out_device,
                on_error=tts_error,
                num_threads=args.tts_threads,
                timing=args.timing,
            )
            if args.tail_guard is not None:
                speaker.tail_guard = args.tail_guard
            speaker.start()
            # 和 ASR 一样先空跑一次:词典载入 / onnx 图预热要一两秒,
            # 不预热的话第一句回答会卡在那里
            speaker.warmup()
            print(f"TTS 就绪({voice}),用时 {time.time() - t0:.1f}s")
            print_mem("TTS 加载后")
        except Exception as exc:
            # 没有扬声器、模型没下全 —— 都不该让识别跟着一起趴下
            degrade("朗读", str(exc).splitlines()[0])
            speaker = None
            args.speak = False

    denoise_label = denoiser.model_name if denoiser is not None else "关"
    if denoiser is not None and wake is not None:
        denoise_label += "(只接 VAD/ASR,唤醒词走未降噪那一路)"
    gain_label = agc.describe() if agc is not None else "关"
    print(f"{head} | VAD: {vad.name} | 停顿定稿: {args.silence}s | "
          f"临时结果间隔: {args.partial_interval or '关'} | "
          f"降噪: {denoise_label} | "
          f"增益: {gain_label} | "
          f"唤醒: {wake_label} | "
          f"对话: {chat_label} | "
          f"朗读: {speaker.voice if speaker else '关'}\n")
    WAKE_HINT[0] = args.wake_word[0]
    WAKE_REPLY[0] = args.wake_reply
    if wake is not None or text_gate is not None:
        once = "叫醒一次接一条指令,答完自动睡回去" if args.wake_once else \
               f"叫醒后 {args.wake_timeout:.0f}s 内可以连续说"
        print(f"{DIM}现在是睡眠状态,识别不工作。说一声「{args.wake_word[0]}」"
              f"或者按回车叫醒 —— {once}。{RESET}")

    # 模拟模式下音频是从文件喂的,闸掉也没用(文件不会等你),只在真麦克风上做
    mic_gate = speaker if (speaker and args.simulate is None and not args.no_mic_gate) else None
    if speaker is not None and mic_gate is None and args.simulate is None:
        print(f"{DIM}提示: 麦克风闸门已关,不戴耳机的话助手会听见自己说话{RESET}")

    if args.chat:
        chat = ChatWorker(
            assistant, print_lock, speaker=speaker,
            timing=args.timing, min_clause_chars=args.tts_min_clause,
            sync_text=args.sync_text,
            first_chunk_chars=args.tts_first_chunk,
            fillers=args.chat_filler,
        )
        chat.start()

    save_dir = None
    if args.save_segments:
        save_dir = Path(args.save_segments)
        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"{DIM}每句定稿音频会存到 {save_dir.resolve()}{RESET}")

    worker = threading.Thread(
        target=decoder_loop,
        args=(
            board,
            rec,
            transcript,
            args.mem,
            chat,
            print_lock,
            speaker,
            args.timing,
            save_dir,
            text_gate,
            matcher,
            machine,
            wake,
            args.wake_once and args.wake,
            manual,
            wake_text,
        ),
        daemon=True,
    )
    worker.start()

    if speaker is not None and args.wake_reply:
        # 应答词是写死的一句,必须叫一声就立刻答 —— 现合要等,预合成不用
        speaker.precache([args.wake_reply])

    if speaker is not None and args.chat and args.chat_filler:
        # 应答词必须命中缓存才有意义:现合成的话它自己就要等五秒
        speaker.precache(args.chat_filler)

    raise_alert = clear_alert = None  # machine 关掉时这两个键就没有动作

    def wake_now(reply: bool = True) -> bool:
        """把唤醒闸门打开,等同于喊一声唤醒词。回车键和报警都走这里。

        返回是否真的把它从睡眠里叫醒了(本来就醒着返回 False)。
        """
        woke = False
        for gate in (wake, text_gate):
            if gate is None or gate.awake:
                continue
            gate.awake = True
            # 两种闸门的超时基准不一样:kws 按音频块累加(idle),
            # text 按墙上时钟(_last)。哪个有就清哪个
            if hasattr(gate, "idle"):
                gate.idle = 0.0
            if hasattr(gate, "_last"):
                gate._last = time.time()
            woke = True
        if woke and reply:
            with print_lock:
                sys.stdout.write(f"{CLEAR_LINE}{DIM}[唤醒] 回车,{args.wake_reply},请说{RESET}\n")
                sys.stdout.flush()
            if speaker is not None and args.wake_reply:
                speaker.interrupt()
                speaker.say(args.wake_reply)
        return woke

    if machine is not None and speaker is not None:
        # 报警词和解除词是写死的句子,提前合成好放着 —— Kokoro 在这台机器上
        # 合成比实时还慢(RTF 2 左右),一句报警现合要七八秒,而报警恰恰是最
        # 不能等的那一类。启动后队列空着的时候后台慢慢备,用的时候直接播。
        n = speaker.precache(
            [a.speech for a in machine_mod.ALERTS]
            + [a.cleared for a in machine_mod.ALERTS]
            # 状态答句拆成"固定帧 + 数字",固定帧和取值有限的那几个数全预合成;
            # 只剩产量要现合,而它前面那句"今天已经完成"正在播,盖得住
            + machine_mod.PRECACHE_TEXTS
        )
        if n:
            print(f"{DIM}预合成 {n} 条固定话术(报警 / 解除 / 状态答句),"
                  f"后台进行,不挡启动{RESET}")

    if machine is not None:
        # 报警是**设备**发起的,不经过麦克风,也不经过 LLM:话术固定、必须立刻出声。
        # 这里用空格键代替真实的 PLC 报警信号 —— 换成真机时,把 raise_alert
        # 挂到 PLC 的报警回调上就行,下面这段一行都不用改。
        def raise_alert(index: int | None = None) -> None:
            alert = machine.next_alert() if index is None else machine.raise_alert(index)
            with print_lock:
                sys.stdout.write(f"{CLEAR_LINE}\a[报警] {alert.title}: {alert.speech}\n")
                if alert.detail:
                    sys.stdout.write(f"{DIM}       {alert.detail}{RESET}\n")
                sys.stdout.flush()
            if speaker is not None:
                # 报警优先级高于正在念的任何东西,先掐掉再说
                speaker.interrupt()
                speaker.say(alert.speech)
            if chat is not None:
                chat.note("设备报警", alert.speech)
            # 报警**不**自动打开闸门。原来是开的(省得工人再喊一遍唤醒词),
            # 可报警刚响的那几秒,正是旁边人最可能说"处理好了""没事了"的时候 ——
            # 闸门一开,这句闲聊就直接把报警撤了。唤醒词多喊一声的代价远小于误解除。
            # 演示时嫌麻烦可以 --alert-wake 恢复旧行为
            if args.alert_wake:
                wake_now(reply=False)

        def clear_alert(index: int | None = None) -> None:
            # 解除也要出声:报警响过而没有收尾,操作工不知道该不该继续等。
            # R 键是全部解除 —— 挂着几条就逐条点名,只说"全部解除"工人不知道
            # 系统认的是不是他理解的那几条
            gone = machine.clear_all() if index is None else \
                   ([a] if (a := machine.clear(index)) else [])
            alert = gone[0] if gone else None
            speech = machine_mod.cleared_speech(gone)
            with print_lock:
                if gone:
                    names = "、".join(a.title for a in gone)
                    sys.stdout.write(f"{CLEAR_LINE}[解除] {names}: {speech}\n")
                    sys.stdout.write(f"{DIM}       {machine.status_text()}{RESET}\n")
                else:
                    # 没响的报警按了解除:只提示一下,不出声 ——
                    # 念一句"已恢复正常"会让人以为刚才真报过警
                    name = "任何报警" if index is None else machine_mod.ALERTS[index].title
                    sys.stdout.write(f"{CLEAR_LINE}{DIM}[解除] 当前没有{name},忽略{RESET}\n")
                sys.stdout.flush()
            if not gone:
                return
            if speaker is not None:
                speaker.interrupt()
                speaker.say(speech)
            if chat is not None:
                chat.note("故障已排除", speech)

    # 信号键:手册里那 17 条(11 条停机报警 + 6 条质量故障)各占一个键。
    #
    # 原来是"1/2 触发、3/4 解除"两段分配,17 条之后不够分了 —— 数字只有 9 个。
    # 现在按键只管**触发**:1-9 放前九条,a-h 放后八条;解除交给语音
    # ("换膜了""横封调好了",每条信号自带说法)和 R 键(全部解除)。
    # 现场本来也是这个分工:报警是设备发的,解除是人做完事之后说的。
    SIGNAL_KEYS = "123456789abcdefgh"
    n_alerts = len(machine_mod.ALERTS) if machine is not None else 0

    def on_key(ch: str) -> None:
        if ch in ("\r", "\n"):
            # 回车 = 喊一声唤醒词。噪声大、或者不想出声的时候用它,
            # 走的和 KWS 检出完全同一条路
            wake_now()
        elif machine is None:
            return  # 关了包装机场景,就只剩回车这一个键
        elif ch == " ":
            raise_alert()  # 按顺序循环,一路演下去
        elif ch.lower() in SIGNAL_KEYS[:n_alerts]:
            raise_alert(SIGNAL_KEYS.index(ch.lower()))  # 直选某一条,不用一路按过去
        elif ch in ("r", "R"):
            clear_alert()  # 全部解除

    if hotkey.start(stop_event, on_key) is not None:
        head = []
        if wake is not None or text_gate is not None:
            head.append("回车 = 唤醒")
        if machine is not None:
            head.append("空格 = 下一条")
            head.append("R = 全部解除")
            head.append("解除也可以直接说(「换膜了」「横封调好了」)")
        if head:
            print(f"{DIM}热键: {' · '.join(head)}{RESET}")
        if machine is not None:
            # 17 条一行一条太长,三条一行排开;停机的标「停」,不停机的标「质」
            cells = [
                f"{SIGNAL_KEYS[i]} {'停' if a.stops else '质'} {a.title}"
                for i, a in enumerate(machine_mod.ALERTS[:len(SIGNAL_KEYS)])
            ]
            for row in range(0, len(cells), 3):
                print(f"{DIM}   " + "".join(f"{c:<22}" for c in cells[row:row + 3]) + RESET)
    else:
        degrade("热键", "当前环境收不到按键(输出被重定向?),语音那条路不受影响")

    if args.simulate is None:
        blocks = mic_blocks(device, sr, stop_event)
    else:
        blocks = file_blocks(args.simulate, stop_event, sr)

    if agc is not None:
        # AGC 放在最前面,降噪之后才分叉(降噪在 segment_loop 里做)。
        # 原来是反过来的 —— 先降噪再 AGC,理由是"降噪压完底噪,AGC 看到的峰值才是人声"。
        # 实测这个理由在低电平麦克风上不成立:GTCRN 是拿正常电平的语料训的,
        # 喂它峰值 0.05 的音频等于超出分布,它会把语音当噪声一起削掉
        # (同一段录音先降噪再放大,唤醒词 0/3;先放大再降噪 1/1 都在)。
        # 先把电平拉到正常再降噪,GTCRN 才工作在它熟悉的区间里。
        blocks = gain.gained_blocks(blocks, agc)

    try:
        segment_loop(
            blocks, board, sr, args, mic_gate, wake, text_gate, print_lock,
            denoiser, vad, cmd_kws,
        )
    except KeyboardInterrupt:
        if speaker is not None:
            speaker.shutup()  # Ctrl-C 就该立刻安静,别还念完最后一句
    finally:
        stop_event.set()
        board.close()
        worker.join(timeout=60)
        if chat is not None:
            chat.close()  # 等最后一条回答生成完再收尾
        if speaker is not None:
            speaker.close()  # 再等它把最后一句念完

    sys.stdout.write(CLEAR_LINE)
    print("\n===== 本次识别全文 =====")
    print("".join(transcript) if transcript else "(没有识别到内容)")

    if agc is not None:
        print(f"\n{DIM}{agc.report()}{RESET}")
        for hint in (gain.level_hint(agc.peak_in, agc.gain), gain.snr_hint(agc.snr_db)):
            if hint:
                print(f"[!] {hint}")

    if wake is not None:
        print(f"\n{DIM}{wake.detector.stats()}{RESET}")

    print("\n===== 内存峰值 =====")
    print(f"进程工作集峰值: {gb(winutil.rss_peak())}   (含 Python / numpy / 模型文件缓存)")
    if args.chat:
        print(f"{DIM}(LLM 是单独的 server 进程,它的内存不算在这里){RESET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
