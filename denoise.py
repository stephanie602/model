#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降噪,接在麦克风和 VAD 之间。用 sherpa-onnx 的 GTCRN。

放在 VAD 前面而不是 ASR 前面是有原因的:噪声大的时候 realtime_asr.py 的能量 VAD
会先崩 —— 底噪 RMS 抬到 0.18 时,阈值被 THRESHOLD_MAX 钳在 0.05,于是永远判定
"有人在说话",每句都是撞 --max-segment 强制切出来的 15 秒。先降噪,VAD 和 ASR 一起受益。

GTCRN 是按嵌入式做的真因果流式模型:权重只有 500 KB,状态留在 C++ 侧,
几乎不占 CPU,而且固定工作在 16 kHz —— 和识别模型一致,整条链路连重采样都省了。
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """和 realtime_asr.resample_to_model_rate 用同一套 scipy 多相滤波,行为保持一致。"""
    if sr == target_sr:
        return audio
    from scipy.signal import resample_poly

    g = gcd(sr, target_sr)
    return resample_poly(audio, target_sr // g, sr // g).astype(np.float32)


class SherpaDenoiser:
    """sherpa-onnx 的 GTCRN 语音增强,固定 16 kHz。

    整段接口是 __call__(audio, sr),给 --simulate 和自检用;
    实时链路走 stream(),那才是它原生的逐帧因果模式。
    """

    def __init__(
        self,
        model_path: str = "gtcrn_simple.onnx",
        num_threads: int = 1,
        provider: str = "cpu",
    ) -> None:
        import sherpa_onnx

        p = Path(model_path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"找不到 GTCRN 模型 {p}。跑一下 setup_windows.ps1,或者从\n"
                "https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models\n"
                "下载 gtcrn_simple.onnx 放进 models\\。"
            )

        config = sherpa_onnx.OnlineSpeechDenoiserConfig(
            model=sherpa_onnx.OfflineSpeechDenoiserModelConfig(
                gtcrn=sherpa_onnx.OfflineSpeechDenoiserGtcrnModelConfig(model=str(p)),
                debug=False,
                num_threads=num_threads,
                provider=provider,
            )
        )
        if not config.validate():
            raise RuntimeError(f"GTCRN 配置不合法,看上面的错误日志:\n{config}")

        self._sherpa = sherpa_onnx
        self._config = config
        self.model_name = "GTCRN"
        self.model_path = str(p)
        probe = sherpa_onnx.OnlineSpeechDenoiser(config)
        self.sr = int(probe.sample_rate)
        self.frame_shift = int(probe.frame_shift_in_samples)

    def _new(self):
        # 每条流一个实例:内部 GRU 状态是有记忆的,两路音频共用一个会互相污染
        return self._sherpa.OnlineSpeechDenoiser(self._config)

    def stream(self, sr: int) -> "SherpaStreamDenoiser":
        return SherpaStreamDenoiser(self, sr)

    def __call__(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """整段降噪。给 --simulate 和自检用,实时链路走 stream()。"""
        x = resample(np.ascontiguousarray(audio, dtype=np.float32), sr, self.sr)
        sd = self._new()
        out = []
        for start in range(0, len(x), self.frame_shift):
            chunk = x[start : start + self.frame_shift]
            out.append(np.asarray(sd(chunk, self.sr).samples, dtype=np.float32))
        out.append(np.asarray(sd.flush().samples, dtype=np.float32))
        y = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

        y = resample(y, self.sr, sr)
        if len(y) < len(audio):
            y = np.pad(y, (0, len(audio) - len(y)))
        return y[: len(audio)]


class SherpaStreamDenoiser:
    """把 0.1 秒的块流喂给 GTCRN,吐回等长的块流。

    块长(0.1s @16k = 1600)和 GTCRN 的帧移(256)对不齐,所以两头都带缓冲:
    输入攒够一帧才送,输出攒够一块才吐。

    关键是开头先往输出缓冲里塞一帧静音。不塞的话每块都会差一点:第 N 块进来时
    模型只吐得出 floor(N*1600/256)*256 个采样点,比 N*1600 少最多 255 个,于是
    每块都要临时补零 —— 补出来的静音是插在信号中间的,等于把整条流一段一段地
    拉长,VAD 的块边界会越错越远。预置一帧之后余量永远够,再也不会欠载,
    代价只是整条流固定延后 256 点(16 ms),块与块之间始终严丝合缝。
    """

    def __init__(self, denoiser: SherpaDenoiser, sr: int) -> None:
        if sr != denoiser.sr:
            raise ValueError(
                f"GTCRN 只工作在 {denoiser.sr} Hz,收到 {sr} Hz。"
                "整条链路应该直接采 16 kHz。"
            )
        self.denoiser = denoiser
        self.sr = sr
        self.frame_shift = denoiser.frame_shift
        self._sd = denoiser._new()
        self._in = np.zeros(0, dtype=np.float32)
        # 预置一帧静音,见类文档:保证之后每一块都能凑够,不用中途补零
        self._out = np.zeros(self.frame_shift, dtype=np.float32)

    def push(self, block: np.ndarray) -> np.ndarray:
        self._in = np.concatenate([self._in, np.asarray(block, dtype=np.float32)])

        n = self.frame_shift
        while len(self._in) >= n:
            chunk, self._in = self._in[:n], self._in[n:]
            got = np.asarray(self._sd(chunk, self.sr).samples, dtype=np.float32)
            if got.size:
                self._out = np.concatenate([self._out, got])

        want = len(block)
        if len(self._out) >= want:
            out, self._out = self._out[:want], self._out[want:]
            return out
        # 兜底:块长小于一帧,或者模型的算法延迟比预置的一帧还长。补在后面 ——
        # 已经出来的样本保持在原来的位置上,不会被往后推
        out = np.concatenate(
            [self._out, np.zeros(want - len(self._out), dtype=np.float32)]
        )
        self._out = self._out[:0]
        return out


def denoised_blocks(blocks, denoiser: SherpaDenoiser, sr: int):
    """把块生成器包一层降噪,块长和采样率都不变,下游 VAD / 分句逻辑无需改动。"""
    stream = denoiser.stream(sr)
    for block in blocks:
        yield stream.push(block)
