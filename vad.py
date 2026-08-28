#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语音活动检测(VAD):判断"这一块里有没有人在说话"。

两种实现,接口一样,realtime_asr.py 用 --vad 选:

    silero (默认)   Silero VAD,640 KB 的神经网络。判的是"这是不是人声",
                    噪声再大也不会被当成说话。
    energy          纯能量阈值,不用模型。安静环境够用,噪声一大就废。

为什么默认换成 silero —— 实测数据(一段 60 秒真人录音,人声清楚但背景噪声持续):

    能量 VAD    600 块里只判出 6 块是人声   ← 等于瘫痪,句子根本切不出来
    Silero      625 块里判出 303 块          ← 正常

能量 VAD 失效的机制是写死的两个常数打架:触发阈值 = 底噪 x 3,但夹在
0.006 ~ 0.05 之间。噪声环境下底噪 RMS 能到 0.02,x3 就是 0.06,被上限钳到 0.05;
而那段录音里语音块的 RMS 只有 0.042-0.059 —— 和阈值重叠。结果是一句话里只有
最响的那个音节能过线,VAD 从半句话中间开始切,SenseVoice 只看到残片,
输出就是"嗯""对""是"这种单字。

上限 0.05 不能简单调高:调高了正常音量的说话也过不了线。真正的出路是别再用能量
判人声 —— 噪声和人声的能量本来就可以一样大,这个信息量不够。Silero 看的是频谱
结构,一句话在它眼里和一台风扇完全不同,不管两者谁更响。

代价:640 KB 模型,1 线程 RTF≈0.01(和唤醒词一个量级,可以忽略)。

单独测:
    .venv\\Scripts\\python.exe vad.py test.wav              # 画出人声区间
    .venv\\Scripts\\python.exe vad.py test.wav --vad energy # 和能量 VAD 对比
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODEL_SAMPLE_RATE = 16000

# Silero 的判定阈值(0-1,越大越保守)。0.5 是官方默认值。
# 调低会把气声、远处说话也收进来;调高会咬掉句子开头的轻声字。
DEFAULT_THRESHOLD = 0.5

# 能量 VAD 的两个夹逼常数,和 realtime_asr 里保持一致
THRESHOLD_MIN = 0.006
THRESHOLD_MAX = 0.05


def find_model(models_dir: Path) -> Path | None:
    p = models_dir / "silero_vad.onnx"
    return p if p.is_file() else None


DOWNLOAD_HELP = """\
没找到 Silero VAD 模型(640 KB),下一个放到 models\\ 底下:

  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

或者重新跑一遍 setup_windows.ps1。不想下的话用 --vad energy 退回纯能量判定
—— 安静环境够用,噪声环境下句子会被切碎。"""


class SileroVad:
    """神经网络 VAD。每来一块 16 kHz 音频调一次 __call__,返回这块里有没有人声。

    Silero 固定吃 512 个样本一窗(32 ms),而我们的块是 1600 个样本(100 ms),
    除不尽。所以这里自己缓冲:攒够一窗算一次,块尾剩下的留到下一块。
    一块里只要有一窗判成人声,这一块就算有人声 —— 宁可多算,后面还有
    --silence 那道停顿判定兜着,漏判反而会把句子从中间切开。
    """

    name = "silero"

    def __init__(
        self,
        model: str | Path,
        threshold: float = DEFAULT_THRESHOLD,
        num_threads: int = 1,
        provider: str = "cpu",
    ) -> None:
        import sherpa_onnx

        p = Path(model).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Silero VAD 模型不存在: {p}\n\n{DOWNLOAD_HELP}")

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(p)
        cfg.silero_vad.threshold = threshold
        # 这两个是 Silero 自己的平滑参数,设小是故意的:分句该由上层的
        # --silence / --max-segment 说了算,这里只回答"当前这一窗是不是人声"。
        # 留着大值会和上层的判定叠加,停顿多久算一句就说不清了。
        cfg.silero_vad.min_silence_duration = 0.1
        cfg.silero_vad.min_speech_duration = 0.1
        cfg.sample_rate = MODEL_SAMPLE_RATE
        cfg.num_threads = num_threads
        cfg.provider = provider

        self.model_path = str(p)
        self.threshold = threshold
        self.vad = sherpa_onnx.VadModel.create(cfg)
        self.window = int(self.vad.window_size())
        self._buf = np.zeros(0, dtype=np.float32)

    def describe(self) -> str:
        return f"Silero({Path(self.model_path).name},阈值 {self.threshold})"

    def __call__(self, block: np.ndarray) -> bool:
        self._buf = np.concatenate([self._buf, np.asarray(block, dtype=np.float32)])
        voiced = False
        while len(self._buf) >= self.window:
            win, self._buf = self._buf[: self.window], self._buf[self.window :]
            if self.vad.is_speech(win):
                voiced = True
        return voiced

    def reset(self) -> None:
        self.vad.reset()
        self._buf = self._buf[:0]


class EnergyVad:
    """原来那套:块 RMS 超过"底噪 x 3"就算有人说话。

    留着有两个用处:没下模型时的兜底,以及 --vad energy 做对照实验
    (想确认"是不是 VAD 的锅",换过来跑一遍最快)。
    """

    name = "energy"

    def __init__(self, fixed_threshold: float | None = None) -> None:
        self.fixed = fixed_threshold
        self.noise: float | None = None
        self.threshold = fixed_threshold or THRESHOLD_MIN
        self.in_speech = False  # 上层每块告诉它,底噪只在没人说话时更新

    def describe(self) -> str:
        if self.fixed is not None:
            return f"能量(固定阈值 {self.fixed})"
        return f"能量(阈值 {self.threshold:.4f},跟随底噪)"

    def __call__(self, block: np.ndarray) -> bool:
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
        if self.noise is None:
            self.noise = rms
        if self.fixed is not None:
            self.threshold = self.fixed
        else:
            self.threshold = min(max(self.noise * 3.0, THRESHOLD_MIN), THRESHOLD_MAX)
        voiced = rms > self.threshold
        # 底噪只在"确定没人说话"的块上更新:降得快、升得慢
        if not voiced and not self.in_speech:
            self.noise = (
                0.9 * self.noise + 0.1 * rms if rms < self.noise
                else 0.99 * self.noise + 0.01 * rms
            )
        return voiced

    def reset(self) -> None:
        pass


def make_vad(kind: str, models_dir: Path, threshold: float | None = None,
             num_threads: int = 1):
    """按名字造一个 VAD。silero 但模型不在时抛异常 —— 不静默降级,
    否则用户以为自己在用神经 VAD,实际还是那套会被噪声钳死的能量判定。"""
    if kind == "energy":
        return EnergyVad(threshold)
    model = find_model(models_dir)
    if model is None:
        raise FileNotFoundError(DOWNLOAD_HELP)
    return SileroVad(model, threshold if threshold is not None else DEFAULT_THRESHOLD,
                     num_threads=num_threads)


def main() -> int:
    import argparse
    import time
    from math import gcd

    import soundfile as sf

    import winutil

    winutil.setup_console()

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="VAD 自检:看一段录音里哪些地方被判成人声")
    p.add_argument("wav")
    p.add_argument("--vad", choices=("silero", "energy", "both"), default="both")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(audio[:, 0])
    if sr != MODEL_SAMPLE_RATE:
        from scipy.signal import resample_poly

        g = gcd(int(sr), MODEL_SAMPLE_RATE)
        x = resample_poly(x, MODEL_SAMPLE_RATE // g, int(sr) // g).astype(np.float32)

    kinds = ("silero", "energy") if args.vad == "both" else (args.vad,)
    block = 1600
    print(f"{args.wav}  {len(x) / MODEL_SAMPLE_RATE:.1f}s\n")
    for kind in kinds:
        try:
            vad = make_vad(kind, here / "models", args.threshold)
        except FileNotFoundError as exc:
            print(f"{kind}: {exc}\n")
            continue
        t0 = time.time()
        flags = [vad(x[i * block : (i + 1) * block]) for i in range(len(x) // block)]
        cost = time.time() - t0
        # 连成区间,好一眼看出句子边界在哪
        spans, start = [], None
        for i, v in enumerate(flags):
            if v and start is None:
                start = i
            elif not v and start is not None:
                spans.append((start * 0.1, i * 0.1))
                start = None
        if start is not None:
            spans.append((start * 0.1, len(flags) * 0.1))
        print(f"{vad.describe()}   RTF {cost / max(len(x) / MODEL_SAMPLE_RATE, 1e-9):.4f}")
        print("  " + "".join("█" if v else "·" for v in flags))
        print(f"  判为人声 {sum(flags)}/{len(flags)} 块,{len(spans)} 个区间")
        for a, b in spans[:12]:
            print(f"    {a:6.1f} - {b:6.1f}s  ({b - a:.1f}s)")
        if len(spans) > 12:
            print(f"    ... 还有 {len(spans) - 12} 个")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
