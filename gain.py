#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动增益(AGC):把过轻的麦克风信号拉到模型和 VAD 认得出的电平。

为什么需要:
    实测这块板子上的麦克风,正常说话录出来 RMS 只有 0.004-0.007、峰值 0.05,
    比正常电平(峰值 0.1-0.4)低 20-30 dB。这个电平会同时卡死两道阈值 ——
    VAD 的下限 THRESHOLD_MIN 是 0.006,和说话本身一样大,于是半句话被判成静音;
    唤醒词那边更直接:同一条录音原始电平 0 检出,放大 8 倍后 100% 检出。

    根子在硬件(系统输入音量 + 麦克风加强没开满),但软件这一层必须扛住 ——
    不是每块板子都能进设置页调,换个机器又要重来一遍。

为什么是 AGC 而不是固定倍数:
    固定 8 倍在你正常说话时刚好,凑近麦克风说一句就削顶(削顶比过轻还难识别,
    波形被削平之后高频全是谐波垃圾)。AGC 按最近几秒的实际峰值动态调,
    离得远就多补、凑近了就少补。

三条关键设计,顺序不能乱:

    1. 只在"有人说话"的块上更新增益。安静时冻住不动 —— 否则底噪会被一路
       放大到和说话一样响,VAD 直接瞎掉(这是 AGC 最常见的翻车方式)。
    2. 增益本身对所有块一视同仁地乘上去(包括静音块),所以信噪比一点不变,
       只是整体抬高。VAD 的自适应底噪估计跟着一起涨,阈值关系不受影响。
    3. 压下来要快、抬上去要慢。突然的大声必须立刻压住防削顶;
       安静的人声慢慢抬,避免一句话中间增益跳变导致音量忽大忽小。

单独测:
    .venv\\Scripts\\python.exe gain.py 某段录音.wav              # 只看电平分析
    .venv\\Scripts\\python.exe gain.py 某段录音.wav -o 处理后.wav  # 写出处理后的音频
"""

from __future__ import annotations

import math
import sys
from collections import deque
from pathlib import Path

import numpy as np

# 目标峰值。0.5 而不是 0.9:留一半余量给突然的大声,免得 AGC 还没反应过来就削了顶。
TARGET_PEAK = 0.5
# 死区:峰值包络落在这个区间里就一点都不动。
# AGC 的职责是治"长期电平不对",不是把每段音频都碾成一样响 —— 本来就正常的输入
# 碰都不该碰。实测过反面例子:一段峰值 0.34、已经很健康的噪声录音被推到 0.67,
# 字与字之间的噪声跟着抬起来,唤醒词误报立刻多出好几次。
# 下限定 0.1 而不是 0.2:0.2 会把"稍微轻一点的正常说话"也当成要救的对象。
# 实测一段峰值 0.34 的录音,里面较安静的段落包络只有 0.15,被抬了 1.7 倍,
# 字缝里的噪声跟着起来,唤醒词就在那一段多误报了一次。
# 0.1 这条线:实测最轻的那支麦克风峰值 0.033-0.056,稳稳在线下,照样救得到。
DEADBAND_LO = 0.1
DEADBAND_HI = 0.8
# 低于这条线才允许"一步到位"地补增益(见下面的 snap)。
# 刚开口那一块的包络本来就偏小,拿它当长期电平会严重高估该补的量:
# 实测一段电平健康的录音,起音那块 env 才 0.165,snap 直接跳到 3 倍,
# 之后 0.4 秒才收回来 —— 就这 0.4 秒的过冲,唤醒词多误报了一次。
# 定在 0.08:实测那支有问题的麦克风峰值 0.033-0.056,一开口就够得着;
# 而正常麦克风的起音块通常已经在这条线以上,只走平滑爬升,最终落在死区里什么也不做。
SNAP_BELOW = 0.08
# 增益上下限。上限 32(+30 dB)够把这块板子的麦克风(需要 8 倍)拉起来还有富余;
# 下限 0.25 是给凑得太近、本来就爆表的情况留的衰减空间。
MAX_GAIN = 32.0
MIN_GAIN = 0.25
# 每块最多往上抬 25%(0.1 秒一块 → 约 2 dB/块),从 1 倍爬到 8 倍要 0.9 秒。
# 再快会在一句话中间听出音量爬升,再慢则跟不上说话人忽远忽近。
# 注意这个爬升速度只管"已经在工作之后"的微调,开机第一句靠下面的 snap 一步到位。
RELEASE_RATE = 1.25
# 压下来一步到位打七折。注意这只是"不会削顶、只是有点大"时的平滑收敛;
# 真要削顶了会直接一步降到位,见 __call__ 里那句 —— 削顶是不可逆的,
# 波形被削平之后高频全是谐波垃圾,识别率掉得比过轻还狠。
ATTACK_RATE = 0.7
# 电平估计用"最近这么久里最响的一块",不用衰减包络。
#
# 这一点走了两次弯路,记下来:衰减包络(env = max(peak, env*decay))掉得快了是
# 压缩器,会把句内弱音和字缝里的噪声一起顶上来 —— 实测在一段本来就够响的录音上
# 把增益推到 6.9 倍,唤醒词误报从 0 涨到 4 次;掉得慢了(0.98,半衰期 3.4 秒)
# 仍然扛不住正常的说话停顿:静两秒包络就减半,掉出死区,增益又开始往上爬。
#
# 滑动窗口最大值没有这个毛病:比窗口短的停顿完全不影响估计值。
# 10 秒足够长(说话人真的换了位置也能在 10 秒内跟上),又不至于记住十分钟前的事。
LEVEL_WINDOW_SECONDS = 10.0
# 判"这段音频里到底有没有信号"的判据:峰值包络 / 底噪 RMS 超过这个倍数。
#
# 为什么不逐块判"这一块是不是人声":那需要一个可信的底噪值,而底噪要从安静的块
# 里估 —— 先有鸡还是先有蛋。实测踩过:一条 1.5 秒的录音,人声在前、静音在后,
# 冷启动时窗口里最小的那块本身就含人声,阈值被抬到人声之上,16 块里 0 块被认成
# 语音,增益一动不动。
#
# 换成看包络和底噪的比值就没这个问题:人声的峰值远高于底噪 RMS。
SIGNAL_OVER_NOISE = 8.0
# 再加一道绝对下限:包络不到这个值就当成"没人说话",增益冻住。
# 光靠上面的比值不够 —— 实测一段室内底噪的峰值能到自身 RMS 的 12 倍(空调、
# 电流声这些不是高斯白噪声),比值判据会把它当信号,一路放大到 32 倍。
# 这道线定在 0.01:实测最轻的一条人声录音峰值 0.033,而室内底噪峰值 0.002-0.005,
# 两边各留了 3 倍余量。
MIN_SIGNAL_PEAK = 0.01
# 底噪的绝对下限,防止在完全数字静音(全 0)的输入上除出天文数字
NOISE_FLOOR_MIN = 1e-5
# 低于这个 RMS 的块当成"数字静音"—— 真麦克风再安静也有底噪,录不出这么干净的东西。
# 只有两种来源:--simulate 读的文件里补的零,或者驱动/权限没配好交上来的一整条静音。
# 拿它们估底噪会把底噪压到地板上,信噪比算出来直接虚高几十 dB。
DIGITAL_SILENCE = 1e-4
# 底噪取最近这么长时间里最安静的一块。用"窗口最小值"而不是"第一块"或者 EMA:
#   - 拿第一块当初值,程序启动时正好有人在说话就废了 —— 底噪被初始化成人声电平,
#     之后所有说话都够不到 4 倍,增益永远不动。实测就是这么翻的:单独跑一条
#     1.5 秒的录音,16 块里 0 块被认成语音。
#   - 纯 EMA 会被持续的说话慢慢抬上去,时间一长同样认不出语音。
# 3 秒里总归有字与字之间的停顿,最小值就是真实底噪。
NOISE_WINDOW_SECONDS = 3.0


class AutoGain:
    """逐块自动增益。每来一块音频调一次 __call__,返回处理后的块。

        agc = AutoGain()
        for block in blocks:
            block = agc(block)
    """

    def __init__(
        self,
        target_peak: float = TARGET_PEAK,
        max_gain: float = MAX_GAIN,
        min_gain: float = MIN_GAIN,
        fixed: float | None = None,
    ) -> None:
        self.target_peak = target_peak
        self.max_gain = max_gain
        self.min_gain = min_gain
        self.fixed = fixed  # 给了就是固定增益,不自适应
        self.gain = 1.0 if fixed is None else fixed
        self.noise = NOISE_FLOOR_MIN
        self._rms_hist: deque[float] = deque(maxlen=max(2, int(NOISE_WINDOW_SECONDS / 0.1)))
        self._peak_hist: deque[float] = deque(maxlen=max(2, int(LEVEL_WINDOW_SECONDS / 0.1)))
        self.env = 0.0  # 最近 LEVEL_WINDOW_SECONDS 里的最大峰值
        # 统计,给启动提示和 --timing 用
        self.peak_in = 0.0
        self.peak_out = 0.0
        self.clipped_blocks = 0
        self.speech_blocks = 0
        # 有人说话时的 RMS(输入端,增益前),配合 self.noise 算信噪比。
        # 取 90 分位而不是均值:一句话里大半是弱音和字间停顿,均值会被它们拖低。
        self._speech_rms: deque[float] = deque(maxlen=300)

    def __call__(self, block: np.ndarray) -> np.ndarray:
        peak = float(np.abs(block).max()) if block.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64)))) if block.size else 0.0
        self.peak_in = max(self.peak_in, peak)

        if self.fixed is None:
            if rms > DIGITAL_SILENCE:
                self._rms_hist.append(rms)
            if self._rms_hist:
                self.noise = max(min(self._rms_hist), NOISE_FLOOR_MIN)
            self._peak_hist.append(peak)
            self.env = max(self._peak_hist)
            if self.env < max(self.noise * SIGNAL_OVER_NOISE, MIN_SIGNAL_PEAK):
                pass  # 只有底噪:增益冻住,否则底噪会被一路放大到和人声一样响
            else:
                if DEADBAND_LO <= self.env <= DEADBAND_HI:
                    # 电平本来就正常 —— 注意这里要主动往 1.0 收,不能只是"不动手":
                    # 说话人刚开口那几块包络还小,增益已经被抬上去了,
                    # 不收的话等音量起来之后那个偏大的增益会一直留着,照样削顶。
                    want = 1.0
                else:
                    want = self.target_peak / max(self.env, 1e-6)
                want = min(max(want, self.min_gain), self.max_gain)
                if peak * self.gain > 1.0:
                    # 这一块已经要削顶了:立刻降到位,平滑收敛在这儿不适用
                    self.gain = want
                elif not self.speech_blocks and self.env < SNAP_BELOW:
                    # 开机后第一次听到人说话、而且电平明显偏低:一步到位,不走爬升。
                    # 否则第一句话("你好小智"就一秒半)还没爬到位就说完了 ——
                    # 实测慢爬时第一句只补到该补的一半,唤醒词照样检不出。
                    self.gain = want
                elif want < self.gain:
                    self.gain = max(want, self.gain * ATTACK_RATE)  # 压:快
                else:
                    self.gain = min(want, self.gain * RELEASE_RATE)  # 抬:慢
                self.speech_blocks += 1
                self._speech_rms.append(rms)

        out = block * self.gain
        # 硬削顶保护。真被削到了要记一笔 —— 一直在削说明 target_peak 定高了,
        # 或者输入本身已经在爆表,光靠 AGC 救不回来。
        if np.abs(out).max() > 1.0:
            self.clipped_blocks += 1
            out = np.clip(out, -1.0, 1.0)
        self.peak_out = max(self.peak_out, float(np.abs(out).max()) if out.size else 0.0)
        return out.astype(np.float32, copy=False)

    def describe(self) -> str:
        if self.fixed is not None:
            return f"固定 {self.fixed:.1f}x"
        return f"自动(当前 {self.gain:.1f}x,底噪 {self.noise or 0:.5f})"

    @property
    def snr_db(self) -> float | None:
        """说话时的信噪比(dB)。还没听到人说话就返回 None。

        增益对它没有影响 —— 放大是把语音和噪声一起放大的,信噪比是录进来那一刻
        就定死的。这也正是它值得单独报出来的原因:电平低了软件能补,信噪比低了
        只能靠离麦克风近一点、或者把噪声源关掉。
        """
        if len(self._speech_rms) < 5:
            return None
        speech = float(np.percentile(list(self._speech_rms), 90))
        return 20.0 * math.log10(max(speech, 1e-9) / max(self.noise, 1e-9))

    def report(self) -> str:
        s = (f"增益: 当前 {self.gain:.1f}x · 输入峰值 {self.peak_in:.3f}"
             f" → 输出峰值 {self.peak_out:.3f}")
        snr = self.snr_db
        if snr is not None:
            s += f" · 信噪比 {snr:.0f} dB"
        if self.clipped_blocks:
            s += f" · 削顶 {self.clipped_blocks} 块"
        return s


def level_hint(peak_in: float, gain: float) -> str | None:
    """输入电平提示。peak_in 是增益前的输入峰值,gain 是 AGC 最后停在的倍数。

    分开报"电平"和"信噪比"是因为两者的处置完全不同:电平低是硬件没调好,
    进系统设置就能解决,AGC 只是替你兜底;信噪比低谁也补不了(见 snr_hint)。

    这里只在两头出声 —— 中间那一大段本来就是 AGC 该干活的区间,不用提醒。
    """
    if peak_in <= 0.0:
        return ("输入峰值 0 —— 一整条数字静音,麦克风根本没录到东西。"
                "\n  设置 → 隐私和安全性 → 麦克风,确认「让桌面应用访问麦克风」是开的"
                "(关着的时候设备看得到、录出来却是全 0,不报任何错)。"
                "\n  或者用 --list-devices 换一条输入设备试试。")
    if peak_in > 0.95:
        return (f"输入峰值 {peak_in:.2f},已经贴着满量程 —— 录进来那一刻就削顶了,"
                f"AGC 只能往下压,救不回被削平的波形。"
                f"\n  系统 → 声音 → 输入,音量往下调一档,或者把「麦克风加强」关掉、离麦克风远一点。")
    if peak_in < 0.06:
        s = (f"输入峰值 {peak_in:.3f},比正常电平(0.1-0.4)低了 20-30 dB。"
             f"AGC 已经补到 {gain:.1f}x,但硬件那头调好了效果更稳 —— 放大是连底噪一起放大的。"
             f"\n  系统 → 声音 → 输入,音量拉到 80-100;"
             f"设备属性 → 级别里把「麦克风加强」开到 +20 dB。")
        if gain >= MAX_GAIN * 0.99:
            s += (f"\n  注意增益已经顶到上限 {MAX_GAIN:.0f}x,再轻就补不动了,"
                  f"这一项现在是必须调的。")
        return s
    return None


def snr_hint(snr: float | None) -> str | None:
    """信噪比提示。低于 12 dB 才提一句,而且只说"值得先排查",不下判决。

    为什么不下判决:这个指标和实际效果的相关性没有想象中强。手上两段素材,
    一段整体信噪比 3 dB(人离麦克风近、背景噪声大),唤醒词 17 次喊中 15 次;
    另一段算出来 15 dB,一次都没醒。所以低信噪比是"值得先查的方向",
    不是"一定不行"。

    这个量单独报出来的价值在于:它是整条链路里唯一软件补不回来的东西。
    电平低了 AGC 能补,信噪比低了谁也补不了 —— 放大是连噪声一起放大的。
    """
    if snr is None or snr >= 12:
        return None
    return (f"信噪比 {snr:.0f} dB(人声比噪声只高这么点)。识别不准时先从这儿查:"
            f"\n  离麦克风近一半(距离减半 ≈ +6 dB)、关掉噪声源、或者换指向性麦克风。"
            f"\n  增益补不回信噪比 —— 放大是连噪声一起放大的。"
            f"\n  分句被噪声顶死的话(每句都是碎片)加 --denoise,它救的就是这个。")


def gained_blocks(blocks, agc: AutoGain):
    """把逐块的音频流套上 AGC。块长和采样率都不变,插在哪一层都行。"""
    for block in blocks:
        yield agc(block)


def parse_gain(value: str | None) -> AutoGain | None:
    """把命令行的 --gain 参数翻成 AutoGain。

    off/no/0 → None(不处理);auto → 自适应;数字 → 固定倍数。
    """
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("off", "no", "none", "0"):
        return None
    if v in ("auto", "agc", "on", ""):
        return AutoGain()
    try:
        f = float(v)
    except ValueError:
        raise ValueError(f"--gain 只认 auto / off / 一个倍数,给的是 {value!r}")
    if f <= 0:
        raise ValueError("--gain 的倍数要大于 0")
    return AutoGain(fixed=f)


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse

    import soundfile as sf

    import winutil

    winutil.setup_console()

    p = argparse.ArgumentParser(description="AGC 自检:分析一段录音的电平,可选写出处理后的音频")
    p.add_argument("wav", help="要分析的录音")
    p.add_argument("-o", "--out", default=None, help="把处理后的音频写到这个文件")
    p.add_argument("--gain", default="auto", help="auto / off / 固定倍数")
    p.add_argument("--block", type=float, default=0.1, help="按多长的块处理(秒)")
    args = p.parse_args()

    x, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    x = np.ascontiguousarray(x[:, 0])
    agc = parse_gain(args.gain)
    if agc is None:
        print("--gain off,不做处理")
        return 0

    step = max(1, int(args.block * sr))
    out = np.concatenate([agc(x[i : i + step]) for i in range(0, len(x), step)])

    def stat(a: np.ndarray) -> str:
        return (f"峰值 {np.abs(a).max():.3f}  RMS {np.sqrt(np.mean(np.square(a, dtype=np.float64))):.4f}")

    print(f"音频 {len(x) / sr:.1f}s @ {sr} Hz,{args.wav}")
    print(f"  处理前: {stat(x)}")
    print(f"  处理后: {stat(out)}")
    print(f"  {agc.report()}")
    print(f"  语音块 {agc.speech_blocks} / 共 {len(x) // step + 1} 块")
    hint = level_hint(agc.peak_in, agc.gain)
    if hint:
        print(f"\n[!] {hint}")

    if args.out:
        sf.write(args.out, out, sr, subtype="PCM_16")
        print(f"\n已写出 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
