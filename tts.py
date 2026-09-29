#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把助手的回答读出来:文本 → Kokoro (sherpa-onnx) → 扬声器。

不是"整段回答生成完再合成再播",而是三级流水线,每一级都边收边出:

    LLM 流式吐字 --按句切--> 句子队列 --合成线程--> 音频队列 --播放线程--> 扬声器

Kokoro 在普通 x86 CPU 上的 RTF 大约 0.2-0.4(合成 1 秒音频花 0.2-0.4 秒),所以第一句
一开始播,后面的句子就能在播放的空当里合成完,接起来听不出断点。用户实际等的只有
"第一句话生成 + 合成"这一段,大约 1 秒,而不是整段回答的时间。

回声问题:喇叭放出来的声音会被麦克风重新录进去,识别成用户说的话,再回答一次 ——
自己跟自己聊起来。realtime_asr.py 用 blocking_mic() 在出声期间把 VAD 静音,播完还要
再等 TAIL_GUARD 秒让房间混响衰下去。代价是半双工:正在念的时候听不见用户插话。

单独测试:
    .venv\\Scripts\\python.exe tts.py "你好呀,今天天气不错"
    .venv\\Scripts\\python.exe tts.py --list-voices
"""

from __future__ import annotations

import queue
import sys
import re
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

# 默认用 Matcha:同一台机器上量过,合成一句"正常,"Kokoro 要 1.98 秒、
# Matcha(配 vocos 声码器)0.05 秒。语音助手里合成比实时慢就是致命伤 ——
# 句子越排越多,声音永远追不上。音色只有一个、听感比 Kokoro 素一点,认了。
# 要换回 Kokoro:--tts-model models\kokoro-multi-lang-v1_0(音色见 --list-voices)
DEFAULT_TTS_DIR = "models\\matcha-icefall-zh-baker"
DEFAULT_VOICE = "zf_xiaoxiao"  # 女声,普通话;zm_yunxi 是男声,见 --list-voices

KOKORO_VOICES = {
    "zf_xiaobei": "女声 · 偏稳",
    "zf_xiaoni": "女声 · 偏亮",
    "zf_xiaoxiao": "女声 · 默认",
    "zf_xiaoyi": "女声 · 偏软",
    "zm_yunjian": "男声 · 偏沉",
    "zm_yunxi": "男声 · 默认",
    "zm_yunxia": "男声 · 偏年轻",
    "zm_yunyang": "男声 · 偏播音",
}

# sherpa-onnx 按编号选音色,不认名字。kokoro-multi-lang-v1_0 里的 8 个中文音色
# 排在 45-52,和上面的名字一一对应,所以 --voice zf_xiaoxiao 这种写法能用。
# (编号出自 sherpa-onnx 的 scripts/kokoro/v1.0/generate_voices_bin.py;换成 v1_1
#  之类的其它 Kokoro 包时要重新核对。)
KOKORO_SIDS = {
    "zf_xiaobei": 45,
    "zf_xiaoni": 46,
    "zf_xiaoxiao": 47,
    "zf_xiaoyi": 48,
    "zm_yunjian": 49,
    "zm_yunxi": 50,
    "zm_yunxia": 51,
    "zm_yunyang": 52,
}

# 播完之后再多静音这么久,等房间混响衰掉,免得自己的尾音又触发一次 VAD。
# 0.35 太短:实测助手念完最后一个字,那句话还是会被录回去识别成用户命令。
# 混响之外还有一层 —— 见 _play_loop 里排空输出缓冲那段注释。
TAIL_GUARD = 0.6
FADE_SECONDS = 0.005  # 每段音频头尾各 5 ms 淡入淡出,消掉拼接处的"啪"声
PLAY_SLICE_SECONDS = 0.05  # 播放按小片写,好让 interrupt() 能立刻掐断
# 系统安静多久之后才允许备货(秒)。备货是把 4 个线程吃满的活,它跟 ASR、降噪
# 抢 CPU,而且在跑的那一条没法中断 —— 实测备货期间提问,"提问到出声"从
# 0.00s 变成 1.77s。所以只在真闲下来的空当里备,用户一开口就停。
PRECACHE_IDLE_SECONDS = 2.0
# 停了多久之后,下一句开口前要先垫一小段静音(秒),以及垫多长。
#
# 数字输出(HDMI / S-PDIF)在没有数据时会失锁,声音重新开始时前一两百毫秒
# 被吃掉 —— 表现就是"吞字",而且吞的总是每句话的开头。板子默认的输出设备
# 正是 HDMI 音频,所以这一层是必须的。中途连着播不受影响(间隔远小于阈值)。
IDLE_RESYNC_SECONDS = 0.4
RESYNC_SILENCE_SECONDS = 0.15

# 句末标点:见到就立刻送去合成。带上 ~ 和 ~ 是因为口语化的回答很爱用它们收尾
_SENTENCE_END = "。！？!?；;…~～\n"
# 句中停顿:攒够 MIN_CLAUSE_CHARS 个字才在这里切,否则"好的,"这种碎片会被单独合成,
# 听起来一顿一顿的
_CLAUSE_END = "，,、:：—"
MIN_CLAUSE_CHARS = 12
# 一段回答的**第一小段**攒够几个字就先送去合成,不等标点。默认 0 = 不这么干。
#
# 这是给慢合成准备的补丁:Kokoro 合成一句 13 字要 5.5 秒,第一段切短能早几秒
# 出声。但中文没有词边界,按字数硬切会切在词中间("当前封口温|度")。
# 换成 Matcha 之后一句只要零点几秒,这点等待不值得拿听感去换,所以关掉。
# 万一又换回慢模型,把它设成 8 就行。
FIRST_CHUNK_CHARS = 0
# 不能在这些字后面断句 —— 断在数字中间,念出来就不是那个数了
_NUMERALS = set("0123456789零一二三四五六七八九十百千万点负")

# Markdown 记号和 emoji 念出来是灾难(星号会被读成"星"、emoji 会变成一长串描述)。
# 系统提示里已经要求不要用了,这里再兜一层。
_UNSPEAKABLE = re.compile(
    r"[*#`_>~～|\[\]{}<>]"
    r"|[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0000FE00-\U0000FE0F\U00002190-\U000021FF]"
)


# --------------------------------------------------------------------------- #
# 数字读法
# --------------------------------------------------------------------------- #
# 模型包自带 number-zh.fst,但它只在整段文本按规则匹配得上时才生效,
# 实测"今天已经完成 3898 个包装"这种句子里的数字会漏网,念成"三八九八"。
# 数量念错在这个场景里是硬伤 —— 产量、温度本来就是用户唯一要听的信息,
# 所以在送进合成之前自己转一遍,不指望 fst。
_ZH_DIGITS = "零一二三四五六七八九"
_ZH_UNITS = ("", "十", "百", "千")
_ZH_BIG = ("", "万", "亿", "万亿")


def _four_to_zh(s: str) -> str:
    """四位以内的数字段 → 汉字。零的处理是这里唯一的难点:
    3006 → 三千零六(中间的零要念,而且连着几个只念一个),3060 → 三千零六十。
    """
    out = []
    zero_pending = False
    n = len(s)
    for i, ch in enumerate(s):
        d = int(ch)
        unit = _ZH_UNITS[n - 1 - i]
        if d == 0:
            zero_pending = True  # 先记着,后面真有非零位才补一个"零"
            continue
        if zero_pending and out:
            out.append("零")
        zero_pending = False
        out.append(_ZH_DIGITS[d] + unit)
    return "".join(out)


def _int_to_zh(s: str) -> str:
    """整数串 → 汉字读法,按万/亿分节。"""
    s = s.lstrip("0") or "0"
    if s == "0":
        return "零"
    groups = []  # 从低位往高位,每四位一节
    while s:
        groups.append(s[-4:])
        s = s[:-4]
    parts = []
    for i in range(len(groups) - 1, -1, -1):
        seg = _four_to_zh(groups[i])
        if not seg:
            continue
        # 一节的值不足四位时,和上一节之间要补个"零":1000005 → 一百万零五
        if parts and len(groups[i].lstrip("0")) < 4 and groups[i][0] == "0":
            parts.append("零")
        parts.append(seg + _ZH_BIG[i])
    out = "".join(parts)
    # 口语里 12 念"十二"不是"一十二";但 112 仍然是"一百一十二",所以只削开头
    if out.startswith("一十"):
        out = out[1:]
    return out


# 数字后面跟着这些字的,按一位一位念更自然:年份、编号、型号
_DIGIT_BY_DIGIT_AFTER = ("年",)
_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")


def spell_numbers(text: str) -> str:
    """把阿拉伯数字换成中文读法。3898 → 三千八百九十八,168.2 → 一百六十八点二。

    三类不按位值念,按一位一位念:
      · 年份(2026年 → 二零二六年)—— 念成"两千零二十六年"就成笑话了
      · 开头是 0 的(007)—— 那是编号,不是数量
      · 9 位以上的 —— 电话、身份证、订单号,没人按位值念
    """
    def repl(m: re.Match) -> str:
        s = m.group()
        percent = s.endswith("%")
        if percent:
            s = s[:-1]
        head, _, frac = s.partition(".")
        tail = text[m.end():m.end() + 1]
        if tail in _DIGIT_BY_DIGIT_AFTER or head.startswith("0") or len(head) > 9:
            body = "".join(_ZH_DIGITS[int(c)] for c in head)
            if frac:
                body += "点" + "".join(_ZH_DIGITS[int(c)] for c in frac)
        else:
            body = _int_to_zh(head)
            if frac:
                body += "点" + "".join(_ZH_DIGITS[int(c)] for c in frac)
        return f"百分之{body}" if percent else body

    return _NUMBER.sub(repl, text)


def clean_for_tts(text: str) -> str:
    return spell_numbers(_UNSPEAKABLE.sub("", text)).strip()


def _sid(voice: str) -> int:
    """音色名 → sherpa-onnx 的 speaker id。也接受直接给编号,方便试其它语种的音色。"""
    v = (voice or "").strip()
    if v.lstrip("-").isdigit():
        return int(v)
    if v in KOKORO_SIDS:
        return KOKORO_SIDS[v]
    raise ValueError(
        f"不认识的音色 {v!r}。支持: {', '.join(KOKORO_SIDS)},或者直接给 0-52 的编号。"
    )


def list_voices() -> None:
    print("Kokoro 中文音色:")
    for name, desc in KOKORO_VOICES.items():
        print(f"  --voice {name:<12} {desc}   (sid={KOKORO_SIDS[name]})")
    print("\n模型包里还有英/日/法/意/西等音色,用 --voice <编号> 直接试(0-52)。")


class SentenceBuffer:
    """把流式吐出来的碎片攒成一句一句,好让合成尽早开工。

    句末标点直接切;逗号只在攒够字数之后才切 —— 太碎的片段合成出来韵律是断的,
    而且每段的固定开销(g2p + 一次前向)会被摊得很难看。

    **第一句例外**,而且这个例外是量出来的。在这台机器上扫 Kokoro 的合成时间:

        5 字   0.46s   RTF 0.39      ← 断崖在这儿
        7 字   2.27s   RTF 1.51
        9 字   4.15s   RTF 2.31
        13 字  5.50s   RTF 2.30

    短到 5 个字以内是另一档速度,再长就掉进 RTF 2.3 那一档。所以第一段**不等标点**,
    攒够 FIRST_CHUNK_CHARS 就先送出去:"说完到出声"从 5 秒多压到半秒。

    这只对第一段这么干。整段一次合成的吞吐反而最高(整段 RTF 2.34,每 6 字切一段
    RTF 2.91 —— 切碎是亏的),所以第一段之后照旧按标点攒长句,让后面在播放的空当里
    追上来。
    """

    def __init__(self, min_clause_chars: int = MIN_CLAUSE_CHARS,
                 first_chunk_chars: int = FIRST_CHUNK_CHARS) -> None:
        self.min_clause_chars = min_clause_chars
        self.first_chunk_chars = first_chunk_chars
        self._buf: list[str] = []
        self._first_done = False

    def feed(self, chunk: str) -> list[str]:
        done: list[str] = []
        for ch in chunk:
            self._buf.append(ch)
            if ch in _SENTENCE_END or (
                ch in _CLAUSE_END
                # 第一小段不论多短,见到逗号就切:合成时间对长度有断崖
                # (5 字 0.46s、13 字 5.50s),短的第一段能让声音早好几秒出来。
                # 标点是天然的词边界,这样切不会把词劈开
                and (not self._first_done or len(self._buf) >= self.min_clause_chars)
            ):
                self._first_done = True
                done.extend(self._take())
            elif (
                not self._first_done
                and self.first_chunk_chars
                and len(self._buf) >= self.first_chunk_chars
                # 数字中间不能断:"一百六" / "十八度" 分两次念就不是那个数了
                and ch not in _NUMERALS
            ):
                self._first_done = True
                done.extend(self._take())
        return done

    def flush(self) -> list[str]:
        """回答结束时把剩下的半句吐出来(多数回答不以标点结尾)。"""
        self._first_done = True
        return self._take()

    def _take(self) -> list[str]:
        s = "".join(self._buf).strip()
        self._buf.clear()
        return [s] if s else []


class Speaker:
    """句子进,声音出。合成和播放各一个线程,互不阻塞。

    say() 只管排队,立刻返回 —— 调用方是 ChatWorker 的流式循环,一秒都不能卡。
    """

    def __init__(
        self,
        model_path: str = DEFAULT_TTS_DIR,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
        device=None,
        on_error: Callable[[str], None] | None = None,
        num_threads: int = 2,
        timing: bool = False,
    ) -> None:
        # 合成慢的时候整条链路的表现是"文字早就打完了,声音还在后面慢慢挤出来"。
        # 这一段在 LLM 那边的计时里完全看不到,所以单独记。
        self.timing = timing
        self.voice = voice
        self.speed = speed
        self.device = device
        self.on_error = on_error or (lambda msg: sys.stderr.write(msg + "\n"))
        # 每个房间的混响不一样,留个口子。太小 = 自己的尾音被当成用户命令,
        # 太大 = 助手念完之后你得多等一会儿才能说话
        self.tail_guard = TAIL_GUARD
        # LLM 也在抢 CPU,别把核全占了
        self.num_threads = num_threads

        self.model = None
        self.sample_rate = 24000
        self.sid = 0
        self._use_gen_config = True
        self._init_model(model_path, voice)

        # 固定话术的合成结果:文本 → 音频。报警词、解除词这些是写死的句子,
        # 每次现合成纯属重复劳动 —— 而 Kokoro 在这台机器上 RTF 2.5,
        # 一句报警要合成七八秒,报警恰恰是最不能等的那一类。见 precache()。
        self._cache: dict[str, np.ndarray] = {}
        # 备货队列,和主队列分开:合成线程闲下来才动它。见 precache()
        self._precache_q: "queue.Queue[str]" = queue.Queue()
        self._text_q: "queue.Queue[tuple[int, str] | None]" = queue.Queue()
        # 音频队列里带着原文:播放线程出声时要把它回调给终端(见 on_speak)
        self._audio_q: "queue.Queue[tuple[int, np.ndarray, str] | None]" = queue.Queue()
        # _pending 是"已经交办、还没出完声"的句子数,合成和播放两级共用一个计数,
        # 在最后处置的那一级减掉。麦克风闸门就看它。
        self._lock = threading.Lock()
        self._pending = 0
        self._quiet_at = 0.0
        self._generation = 0  # interrupt() 自增;拿到旧号的句子自己扔掉
        self._closed = False
        self._stream = None
        # 由外面(ChatWorker)塞进来的回调:回答还在生成时返回 True。见 blocking_mic()
        self.busy_hint = None
        # 每句**真正开始出声**的那一刻回调一次,参数是这句话的文本。
        # 终端靠它跟着声音打字 —— 合成慢的时候(这台机器 RTF 2.5),
        # 文字按 LLM 的速度刷完只要一秒,声音还得念十秒,看着像两个程序在跑
        self.on_speak = None
        # 最近一次"有事发生"的时刻:说话、出声、被打断都算。备货看它决定该不该动
        self._last_activity = 0.0
        self._last_write = 0.0      # 上一次往声卡写数据的时刻,见 IDLE_RESYNC_SECONDS

        self._synth_thread = threading.Thread(target=self._synth_loop, daemon=True)
        self._play_thread = threading.Thread(target=self._play_loop, daemon=True)

    # ---------------- 对外接口 ---------------- #
    def start(self) -> None:
        self._synth_thread.start()
        self._play_thread.start()

    def warmup(self) -> None:
        """第一次合成有一次性开销(onnx 图预热 / 词典载入),放在这里空跑一次,
        别让第一句回答背锅。"""
        self._synth("你好。")

    def note_activity(self) -> None:
        """告诉 Speaker"现在有人在说话/在交互",备货让路。VAD 检测到语音就调。"""
        self._last_activity = time.time()

    def precache(self, texts) -> int:
        """把固定话术排进合成队列,合成结果留着,以后 say() 同一句直接取。

        排队而不是当场合成:合成必须串在合成线程里做(OfflineTts 不保证多线程安全),
        而且这样启动时不用等 —— 队列空着的时候它自己慢慢把这几句备好,
        真要用的时候大概率已经在缓存里了。

        预合成走**另一条队列**,而且优先级更低:合成线程先看有没有真要出的声,
        没有才回头备货。同一条队列会出事 —— 开机排了几十条要备的,
        这时候来一句报警,得等前面几十条全备完才轮到它。

        返回排了几句。已经在缓存里的不重复排。
        """
        n = 0
        for t in texts:
            t = clean_for_tts(t)
            if t and t not in self._cache:
                self._precache_q.put(t)
                n += 1
        return n

    def say(self, text: str) -> None:
        self._last_activity = time.time()
        text = clean_for_tts(text)
        if not text:
            return
        with self._lock:
            if self._closed:
                return
            self._pending += 1
            gen = self._generation
        self._text_q.put((gen, text))

    def interrupt(self) -> None:
        """立刻闭嘴:队列里排着的全扔掉,正在播的那段也掐掉。换新回答时用。"""
        with self._lock:
            self._generation += 1
        dropped = self._drain(self._text_q) + self._drain(self._audio_q)
        if dropped:
            self._done(dropped)

    def shutup(self) -> None:
        """马上闭嘴,而且之后 say() 全部当没听见。Ctrl-C 退出时用 ——
        不然还得站在那儿听它把最后一句念完。"""
        with self._lock:
            self._closed = True
        self.interrupt()

    def blocking_mic(self) -> bool:
        """现在该不该把麦克风静音。播完还要再压 TAIL_GUARD 秒等混响衰掉。

        busy_hint 是给"回答还在生成"用的:LLM 边吐字边念,两句之间可能有空档,
        这时 _pending 会短暂归零。不管它的话闸门就在这些空档里一开一合,
        助手自己的话被录回去、识别成用户命令、再触发一次回答 —— 自己跟自己聊。
        """
        if self.busy_hint is not None and self.busy_hint():
            return True
        with self._lock:
            if self._pending > 0:
                return True
            quiet_at = self._quiet_at
        return time.time() - quiet_at < self.tail_guard

    def wait_idle(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._pending == 0:
                    return
            time.sleep(0.05)

    def close(self, timeout: float = 30.0) -> None:
        """等最后一句念完再收 —— 收尾时把话切一半很怪。"""
        self.wait_idle(timeout)
        with self._lock:
            self._closed = True
        self._text_q.put(None)
        self._synth_thread.join(timeout=timeout)
        self._audio_q.put(None)
        if self._play_thread.is_alive():
            self._play_thread.join(timeout=timeout)
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    # ---------------- 内部 ---------------- #
    @staticmethod
    def _drain(q: queue.Queue) -> int:
        # 备货队列不在这里 —— 它是单独一条,interrupt() 不该把备好的货也扔了
        n = 0
        while True:
            try:
                item = q.get_nowait()
            except queue.Empty:
                return n
            if item is None:  # 收工信号不能吞掉
                q.put(None)
                return n
            n += 1

    def _done(self, n: int = 1, quiet_in: float = 0.0) -> None:
        """标记 n 句已经处置完。quiet_in 是"还要过多久才真的没声音"。

        quiet_in 存在的原因见 _play_loop:stream.write() 返回不等于声音播完了。
        """
        with self._lock:
            self._pending = max(0, self._pending - n)
            if self._pending == 0:
                self._quiet_at = time.time() + quiet_in

    def _init_model(self, model_dir: str, voice: str) -> None:
        d = Path(model_dir).expanduser()
        if not d.is_dir():
            raise FileNotFoundError(
                f"找不到 TTS 模型目录 {d}。跑一下 setup_windows.ps1,"
                "或者用 --model 指定解压后的模型目录。"
            )
        # 按目录里有什么文件认模型类型,不看目录名 —— 用户可能改过名字
        if next(d.glob("model-steps-*.onnx"), None) is not None:
            self.kind = "matcha"
            self._init_matcha(d)
        else:
            self.kind = "kokoro"
            self._init_kokoro(d, voice)

    def _init_matcha(self, d: Path) -> None:
        """Matcha(声学模型)+ vocos(声码器)。中文单音色,但快得多。

        换它的唯一理由是延迟:同一台机器上 Kokoro 合成 1 秒音频要 2.5 秒
        (比实时还慢,句子越排越多),Matcha 在 0.1 上下。语音助手里合成慢是
        致命的,音色好听退居其次。

        和 Kokoro 的结构差别:声码器是**单独一个文件**,不在模型包里,
        要自己下(vocos-22khz-univ.onnx);中文分词走 jieba 词典目录 dict/,
        少了它多音字会念错。
        """
        import sherpa_onnx

        acoustic = sorted(d.glob("model-steps-*.onnx"))[-1]  # steps 越多质量越好
        vocoder = self._find_vocoder(d)
        if vocoder is None:
            raise FileNotFoundError(
                f"Matcha 还缺声码器。放一个 vocos-*.onnx 到 {d} 或它的上级 models\\ 里:\n"
                "  https://github.com/k2-fsa/sherpa-onnx/releases/download/"
                "vocoder-models/vocos-22khz-univ.onnx"
            )
        for name in ("tokens.txt", "lexicon.txt"):
            if not (d / name).is_file():
                raise FileNotFoundError(f"{d} 里缺 {name},模型包可能没解全")

        rules = [str(d / n) for n in ("date.fst", "phone.fst", "number.fst")
                 if (d / n).is_file()]
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                    acoustic_model=str(acoustic),
                    vocoder=str(vocoder),
                    lexicon=str(d / "lexicon.txt"),
                    tokens=str(d / "tokens.txt"),
                    # jieba 词典:中文要先分词才查得到词典里的读音
                    dict_dir=str(d / "dict") if (d / "dict").is_dir() else "",
                ),
                provider="cpu",
                num_threads=self.num_threads,
                debug=False,
            ),
            rule_fsts=",".join(rules),
            max_num_sentences=1,
        )
        if not config.validate():
            raise RuntimeError(f"Matcha 配置不合法,看上面的错误日志:\n{config}")
        self.model = sherpa_onnx.OfflineTts(config)
        self.sample_rate = int(getattr(self.model, "sample_rate", 22050))
        self.sid = 0  # 单音色,给什么编号都是它
        self.voice = f"matcha/{d.name}"
        self._use_gen_config = hasattr(sherpa_onnx, "GenerationConfig")

    @staticmethod
    def _find_vocoder(d: Path) -> Path | None:
        """声码器在模型目录里或 models\\ 下都行 —— 它是多个 Matcha 模型共用的。"""
        for where in (d, d.parent):
            hit = sorted(where.glob("vocos-*.onnx"))
            if hit:
                return hit[0]
        return None

    def _init_kokoro(self, d: Path, voice: str) -> None:
        import sherpa_onnx

        for name in ("model.onnx", "voices.bin", "tokens.txt"):
            if not (d / name).is_file():
                raise FileNotFoundError(f"{d} 里缺 {name},模型包可能没解全")

        # lexicon 是多语言版才有的,中英各一个;少了中文那个,汉字会走 espeak 音译,
        # 念出来是"外国人读拼音"的调子
        lexicons = [str(d / n) for n in ("lexicon-us-en.txt", "lexicon-zh.txt")
                    if (d / n).is_file()]

        # 文本规整规则,模型包自带。不接上的话数字和日期是按字面念的:
        # "2025年" 念成"二零二五年"、"13800138000" 念成一长串孤立数字。
        # LLM 的回答里日期和数字太常见了,这个必须开。
        # 顺序有讲究:date 和 phone 要排在 number 前面 —— number 规则会把所有
        # 连续数字都吃掉,排在前面的话日期和电话就再也匹配不上了。
        rules = [str(d / n) for n in ("date-zh.fst", "phone-zh.fst", "number-zh.fst")
                 if (d / n).is_file()]

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(d / "model.onnx"),
                    voices=str(d / "voices.bin"),
                    tokens=str(d / "tokens.txt"),
                    data_dir=str(d / "espeak-ng-data"),
                    lexicon=",".join(lexicons),
                ),
                provider="cpu",
                num_threads=self.num_threads,
                debug=False,
            ),
            rule_fsts=",".join(rules),
            # 上游 SentenceBuffer 已经按标点切好了,一次就送一句进来,不用它再攒
            max_num_sentences=1,
        )
        if not config.validate():
            raise RuntimeError(f"Kokoro 配置不合法,看上面的错误日志:\n{config}")

        self.model = sherpa_onnx.OfflineTts(config)
        # 新版有 .sample_rate 属性,老版没有。拿不到就先按 Kokoro 的 24 kHz 记着,
        # 第一次合成(warmup)会用结果里的真实值覆盖掉,那时播放流还没开
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self.sid = _sid(voice)
        self._use_gen_config = hasattr(sherpa_onnx, "GenerationConfig")

    def _synth(self, text: str) -> np.ndarray | None:
        import sherpa_onnx

        # sherpa-onnx 换过一次 generate() 的签名:新版收一个 GenerationConfig,
        # 老版是 sid/speed 两个关键字参数。两种都支持,按运行时实际有没有这个类来选。
        if self._use_gen_config:
            gen = sherpa_onnx.GenerationConfig()
            gen.sid = self.sid
            gen.speed = self.speed
            r = self.model.generate(text, gen)
        else:
            r = self.model.generate(text, sid=self.sid, speed=self.speed)
        # 必须 copy:下面的淡入淡出是原地改的,而 r.samples 可能还指着 C++ 侧的缓冲区
        audio = np.array(r.samples, dtype=np.float32, copy=True).reshape(-1)
        if not audio.size:
            return None
        self.sample_rate = int(r.sample_rate)

        # 头尾各淡 5 ms:两段音频直接拼在一起时,波形跳变会听成一声"啪"
        n = min(int(FADE_SECONDS * self.sample_rate), len(audio) // 2)
        if n > 0:
            ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
            audio[:n] *= ramp
            audio[-n:] *= ramp[::-1]
        return audio

    def _synth_loop(self) -> None:
        while True:
            try:
                # 真要出的声优先。等一小会儿是为了让出 CPU,不是空转
                item = self._text_q.get(timeout=0.05)
            except queue.Empty:
                # 闲着才备货:备一条要好几秒,这期间来了真活也得等它做完,
                # 所以每次只备一条,做完回头再看主队列
                idle_for = time.time() - self._last_activity
                if idle_for < PRECACHE_IDLE_SECONDS:
                    continue  # 刚刚还在交互,别去抢 CPU
                try:
                    text = self._precache_q.get_nowait()
                except queue.Empty:
                    continue
                t0 = time.time()
                try:
                    audio = self._synth(text)
                except Exception as exc:
                    self.on_error(f"[预合成失败] {exc}")
                    continue
                if audio is not None:
                    self._cache[text] = audio
                # 备完一条歇同样久再备下一条,把 CPU 让出去一半 ——
                # 但真活一来立刻醒:睡在这儿的话,用户的问题要等这一觉睡完
                nap = min(time.time() - t0, 3.0)
                end = time.time() + nap
                while time.time() < end and self._text_q.empty():
                    time.sleep(0.05)
                continue
            if item is None:
                return
            gen, text = item
            with self._lock:
                stale = gen != self._generation
            if stale:
                self._done()
                continue
            t0 = time.time()
            cached = self._cache.get(text)
            try:
                audio = cached if cached is not None else self._synth(text)
            except Exception as exc:  # 合成失败不该带崩对话
                self.on_error(f"[合成失败] {exc}")
                self._done()
                continue
            if audio is None:
                self._done()
                continue
            if self.timing:
                cost = time.time() - t0
                dur = len(audio) / max(self.sample_rate, 1)
                if cached is not None:
                    self.on_error(
                        f"\033[2m[计时] TTS 命中预合成缓存,直接播 · {len(text)} 字\033[0m"
                    )
                    self._audio_q.put((gen, audio, text))
                    continue
                # RTF 是这里唯一要看的数:合成 1 秒音频花了几秒。
                # >1 就是合成比播放还慢,句子会越积越多,声音永远追不上文字。
                self.on_error(
                    f"\033[2m[计时] TTS {cost:.2f}s / 音频 {dur:.1f}s"
                    f" · RTF {cost / max(dur, 1e-9):.2f} · {len(text)} 字\033[0m"
                )
            self._audio_q.put((gen, audio, text))

    def _play_loop(self) -> None:
        import sounddevice as sd

        while True:
            item = self._audio_q.get()
            if item is None:
                return
            gen, audio, text = item
            with self._lock:
                stale = gen != self._generation
            if stale:
                self._done()
                continue
            if self.on_speak is not None:
                # 声音要出了,通知终端把这句打出来。回调里出异常不能影响播放
                try:
                    self.on_speak(text)
                except Exception:
                    pass
            try:
                if self._stream is None:
                    self._stream = sd.OutputStream(
                        device=self.device,
                        samplerate=self.sample_rate,
                        channels=1,
                        dtype="float32",
                    )
                    self._stream.start()
                    # 刚打开的输出流会吞掉第一个缓冲区 —— 表现是第一句话被削掉
                    # 开头("你好,我在"只剩"在")。先灌 0.2 秒静音把流喂热
                    self._stream.write(
                        np.zeros(int(0.2 * self.sample_rate), dtype=np.float32)
                    )
                    self._last_write = time.time()
                if time.time() - self._last_write > IDLE_RESYNC_SECONDS:
                    # 隔了一会儿没出声:数字输出这段时间里已经失锁了,
                    # 先垫一小段静音让它重新锁上,再放真内容
                    self._stream.write(np.zeros(
                        int(RESYNC_SILENCE_SECONDS * self.sample_rate), dtype=np.float32))
                # 分小片写:阻塞式 write 一次性灌进去的话,interrupt() 得等整句播完才生效
                step = max(1, int(PLAY_SLICE_SECONDS * self.sample_rate))
                for start in range(0, len(audio), step):
                    with self._lock:
                        if gen != self._generation:
                            break
                    self._stream.write(audio[start : start + step])
                self._last_write = time.time()
                # write() 返回只代表数据进了输出缓冲,喇叭还要再响一个设备延迟。
                # Windows 共享模式 WASAPI 上这个延迟通常 100-300 ms,而尾部保护
                # 一共才 0.6 秒 —— 不把它算进去,等于自己吃掉一半保护时间,
                # 助手念完的最后一句话正好被麦克风录回去。
                latency = float(getattr(self._stream, "latency", 0.0) or 0.0)
            except Exception as exc:
                self.on_error(f"[播放失败] {exc}")
                latency = 0.0
            self._done(quiet_in=latency)


def main() -> int:
    import argparse

    import winutil

    winutil.setup_console()

    p = argparse.ArgumentParser(description="文本 → 语音 → 扬声器")
    p.add_argument("text", nargs="*", help="要念的文本")
    p.add_argument("--model", default=DEFAULT_TTS_DIR, help=f"默认 {DEFAULT_TTS_DIR}")
    p.add_argument("--voice", default=DEFAULT_VOICE, help=f"默认 {DEFAULT_VOICE}")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--num-threads", type=int, default=2)
    p.add_argument("--output-device", default=None, help="输出设备编号或名称")
    p.add_argument("--list-voices", action="store_true")
    args = p.parse_args()

    if args.list_voices:
        list_voices()
        return 0

    text = " ".join(args.text) or "你好,我是语音助手,现在开始说话。"
    device = args.output_device
    if device is not None and str(device).lstrip("-").isdigit():
        device = int(device)

    t0 = time.time()
    spk = Speaker(
        args.model, args.voice, args.speed, device, num_threads=args.num_threads,
    )
    print(f"TTS 就绪({args.voice}),用时 {time.time() - t0:.1f}s")
    spk.start()

    buf = SentenceBuffer()
    for s in buf.feed(text) + buf.flush():
        spk.say(s)
    t0 = time.time()
    spk.close()
    print(f"念完,用时 {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
