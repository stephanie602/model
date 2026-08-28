#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唤醒词模块:说"你好小智"之前,整条识别链路睡着不动。

为什么要单独做这个,而不是识别完再匹配文字:
    SenseVoice 每 1.2 秒解一次临时结果、每句再解一次定稿,一个人不说话的房间里
    也是这个开销 —— 底噪偶尔顶过 VAD 阈值就白解一次。KWS 模型只有 3.3 MB,
    一条流式 zipformer 逐块吃 0.1 秒音频,1 线程 RTF≈0.02,比 SenseVoice 便宜
    一个数量级。唤醒之前只跑它,唤醒之后才放 ASR 出来干活。

两种模式:

    kws (默认,省 CPU)   sherpa-onnx KeywordSpotter,专门的关键词检出模型。
                        睡眠时不跑 ASR,这才是省 CPU 的那条路。
                        要下模型:sherpa-onnx-kws-zipformer-wenetspeech-3.3M
    text (兜底,不省 CPU) 没有 KWS 模型时用。照常跑 ASR,只是拿识别出来的文字
                        去匹配唤醒词 —— 省的是 LLM 和 TTS,ASR 该跑还是跑。

还有一层能量闸(energy_gate,默认关):安静的块连特征都不提,直接跳过。
默认关掉是实测的结论 —— 它省不了多少,却会漏检:跳过的块在流里留下断口,
后面所有音频的分块对齐整体偏移,同一段录音开着闸检出 6 次、关掉 11 次。
KWS 本身 RTF 才 0.012(一个核的 1.2%),闸掉能省到 0.004,这点差别不值得拿漏检换。
想省到极致再用 energy_gate=True。

单独测(对着麦克风说"你好小智"):
    .venv\\Scripts\\python.exe wakeword.py
    .venv\\Scripts\\python.exe wakeword.py --wav test.wav
    .venv\\Scripts\\python.exe wakeword.py --keywords 你好小智 小爱同学
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

MODEL_SAMPLE_RATE = 16000

DEFAULT_KEYWORD = "你好小智"

# 下面三个默认值不是抄来的,是拿一段 60 秒双麦真人录音(带持续噪声,喊了 17 次
# "你好小智")扫出来的。sherpa-onnx 的原始默认值(int8 / score 1.5 / 阈值 0.25 /
# beam 4)在那段录音上只检出 1 次,基本等于不能用。
#
# 检出阈值:模型给出的关键词概率超过它才算数。
# 0.25 是 sherpa-onnx 的默认值,实测偏保守(score 3.0 下 9 次 vs 0.1 下 11 次)。
# 误唤醒多就往 0.25-0.4 调,叫不醒就往 0.03 调。
DEFAULT_THRESHOLD = 0.1
# 解码时给关键词路径的加分,和 ASR 热词是一回事。
# 实测这个最灵:1.5 → 3 次,3.0 → 11 次。再往上到 5.0 反而掉(路径被压过头)。
DEFAULT_SCORE = 3.0
# 解码时保留几条候选路径。sherpa-onnx 默认 4,实测 4 → 11 次、8 → 14 次、10 → 15 次,
# 再往上(16)会掉。代价小得可以忽略:RTF 从 0.0114 涨到 0.0121。
# 噪声里关键词的声学证据本来就弱,候选路径太少的话它在中途就被别的词挤掉了。
DEFAULT_MAX_ACTIVE_PATHS = 10

# 命令词用 KWS 直接检出时的阈值(--command-kws),比唤醒词那个 0.1 保守得多。
# 理由是两件事不一样:唤醒词只有一条、叫不醒最多再喊一声;命令表有几十条、
# 而且一命中就真的去执行,误触发的代价大得多。
# 实测(31 条命令表,混入真实噪声):
#   阈值 0.10  干净 2对0误 / 15dB 2对2误 / -5dB 1对1误   ← 误报
#   阈值 0.25  各档全程 0 误报,5 dB 仍然 2/2             ← 取这个
#   阈值 0.40  5 dB 掉到 0/2                            ← 太保守
DEFAULT_COMMAND_THRESHOLD = 0.25

# 能量闸的静音判据:块 RMS 低于这个值就当没人说话,不喂给 KWS。
# 注意能量闸默认是关的,见模块开头 —— 跳过块会打乱流的分块对齐,漏检代价远大于省下的 CPU。
GATE_RMS = 0.003
# 能量闸的预滚:声音起来时先补喂这么长的历史音频
GATE_PRE_ROLL_SECONDS = 0.3
# 连续静音这么久就重置一次 KWS 流,清掉上一段说话残留的解码状态
GATE_RESET_SECONDS = 2.0
# 两次检出至少隔这么久,防同一句话被报两次。实测连着喊时最短间隔 1.5 秒,
# 所以 0.8 秒既吃得掉重复,又不会把"喊完没反应、马上再喊一次"那次吞掉。
DEBOUNCE_SECONDS = 0.8

# text 模式的同音变体。语音识别把"智"听成"知/志/致"是家常便饭,
# 只认一种写法的话十次里有三四次叫不醒。
TEXT_VARIANTS = {
    "你好小智": ["你好小智", "你好小知", "你好小志", "你好小致", "你好小治",
                 "你好晓智", "尼好小智", "你好晓知"],
}


def find_kws_model(models_dir: Path) -> Path | None:
    """在 models\\ 底下找 KWS 模型目录。没有就返回 None,让调用方决定怎么办。"""
    for p in sorted(models_dir.glob("sherpa-onnx-kws-*")):
        if p.is_dir():
            return p
    return None


DOWNLOAD_HELP = """\
没找到唤醒词模型。下一个 3.3 MB 的中文 KWS 模型就行(解压到 models\\ 底下):

  https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/
      sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

或者重新跑一遍 setup_windows.ps1,它现在会顺带把这个模型下下来。

不想下模型的话可以用 --wake-mode text:拿 ASR 的识别结果匹配唤醒词。
能用,但省不了 CPU —— ASR 照样一直在跑,省的只是 LLM 和 TTS 那两头。"""


def _load_vocab(tokens: Path) -> set[str]:
    """tokens.txt 里所有 token 的字面,用来提前判断关键词编不编得进去。"""
    vocab = set()
    for line in tokens.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line:
            continue
        parts = line.rsplit(" ", 1)  # 格式 "<token> <id>",从右边切
        if len(parts) == 2:
            vocab.add(parts[0])
    return vocab


def detect_modeling_unit(tokens: Path) -> str:
    """看 tokens.txt 判断这个模型是按什么单位建模的:cjkchar 还是 ppinyin。

    官方那个 3.3M 的中文 KWS 模型建模单元是拼音的声母 + 韵母(带声调),
    tokens.txt 里一个汉字都没有,长这样:

        zh 10
        ǎo 37
        iǎo 88

    所以关键词不能直接写成"你 好 小 智" —— 要先转成"n ǐ h ǎo x iǎo zh ì"。
    另一些 KWS 模型是 cjkchar 建模,词表里就是汉字本身。两种都得认。
    """
    for line in tokens.read_text(encoding="utf-8", errors="replace").splitlines():
        tok = line.rsplit(" ", 1)[0] if " " in line else line
        if any("一" <= c <= "鿿" for c in tok):
            return "cjkchar"
    return "ppinyin"


PYPINYIN_HELP = """\
唤醒词模型是拼音建模的,把汉字转成拼音要用 pypinyin(纯 Python,几百 KB):

    .venv\\Scripts\\python.exe -m pip install pypinyin

不想装的话,可以自己写好 keywords 文件用 --wake-keywords-file 指过来,
格式照抄模型目录里的 keywords.txt:

    n ǐ h ǎo x iǎo zh ì @你好小智"""


def _to_ppinyin(phrase: str) -> list[str]:
    """汉字 → 声母/韵母 token 串。"你好" → ['n', 'ǐ', 'h', 'ǎo']

    和 sherpa_onnx.text2token(tokens_type="ppinyin") 是同一套算法,自己写一遍
    是为了绕开它顶上那句无条件的 import sentencepiece —— 我们这条路根本用不到 bpe,
    没必要为它多拖一个几 MB 的 wheel。
    """
    try:
        from pypinyin import pinyin
        from pypinyin.contrib.tone_convert import to_finals_tone, to_initials
    except ImportError as exc:
        raise RuntimeError(PYPINYIN_HELP) from exc

    out: list[str] = []
    for py in (x[0] for x in pinyin(phrase)):
        initial = to_initials(py, strict=False)
        final = to_finals_tone(py, strict=False)
        # 声母韵母都切不出来的(英文、数字)就原样留着,让下面的查表去判它能不能用
        out.extend([t for t in (initial, final) if t] or [py])
    return out


def _toneless_per_char(text: str) -> list[str]:
    """逐字转成不带声调的拼音,长度和输入一一对应(非汉字原样留着)。"""
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise RuntimeError(PYPINYIN_HELP) from exc

    # 一个字一个字地转:整句转的话 pypinyin 会按词合并,拿不到逐字对齐
    return [lazy_pinyin(c, style=Style.NORMAL)[0] if "一" <= c <= "鿿" else c for c in text]


def _toneless(text: str) -> str:
    return "".join(_toneless_per_char(text))


def _write_keywords_file(
    keywords: list[str], tokens: Path, score: float, threshold: float
) -> tuple[str, list[str], list[str]]:
    """把关键词写成 sherpa-onnx 要的 keywords 文件。

    格式是一行一个:  n ǐ h ǎo x iǎo zh ì :1.5 #0.25 @你好小智
    前面是按建模单元切开的 token,:分数 #阈值 是这一条单独的覆盖值,
    @后面是检出时回报给我们的原文(不写的话回报的是那串 token,没法看)。

    返回 (临时文件路径, 收下的关键词, 编不进去的关键词)。
    """
    import tempfile

    vocab = _load_vocab(tokens)
    unit = detect_modeling_unit(tokens)

    keep: list[str] = []
    skipped: list[str] = []
    lines: list[str] = []
    for kw in keywords:
        phrase = kw.strip().replace(" ", "")
        if not phrase:
            continue
        toks = list(phrase) if unit == "cjkchar" else _to_ppinyin(phrase)
        # 词表里查不到的 token,sherpa-onnx 只会往 stderr 打一行英文然后跳过,
        # 用户完全看不出自己的唤醒词根本没生效,所以自己先查一遍。
        bad = [t for t in toks if t not in vocab]
        if bad:
            skipped.append(f"{phrase}(词表缺 {' '.join(sorted(set(bad)))})")
            continue
        keep.append(phrase)
        # 逐条阈值:词越短要求越严。sherpa 的 keywords 文件支持行内 #阈值 覆盖全局值。
        # 短词的声学证据本来就少,同一个阈值下"制冷"这种两音节的会乱触发 ——
        # 实测 31 条命令表在 15 dB 噪声里,"制冷"被凭空报了出来,而四音节的
        # "打开电灯""关闭电灯"一次误报都没有。
        n_syl = len(toks) if unit == "cjkchar" else len(toks) // 2
        th = threshold * (2.0 if n_syl <= 2 else 1.4 if n_syl == 3 else 1.0)
        lines.append(f"{' '.join(toks)} :{score} #{min(th, 0.95):.3f} @{phrase}")

    if not lines:
        return "", keep, skipped

    fd, tmp = tempfile.mkstemp(prefix="keywords_", suffix=".txt", text=True)
    with open(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return tmp, keep, skipped


# --------------------------------------------------------------------------- #
# KWS 模式
# --------------------------------------------------------------------------- #
class WakeWord:
    """流式唤醒词检出器。

    用法就两句:每来一块 16 kHz 音频调一次 accept(),返回非空字符串就是被唤醒了。

        ww = WakeWord(model_dir)
        for block in blocks:
            hit = ww.accept(block)
            if hit:
                ...
    """

    mode = "kws"

    def __init__(
        self,
        model_dir: str | Path,
        keywords: list[str] | None = None,
        threshold: float = DEFAULT_THRESHOLD,
        score: float = DEFAULT_SCORE,
        num_threads: int = 1,
        provider: str = "cpu",
        energy_gate: bool = False,
        gate_rms: float = GATE_RMS,
        block_seconds: float = 0.1,
        keywords_file: str | None = None,
        max_active_paths: int = DEFAULT_MAX_ACTIVE_PATHS,
    ) -> None:
        import sherpa_onnx

        d = Path(model_dir).expanduser()
        if not d.is_dir():
            raise FileNotFoundError(f"唤醒词模型目录不存在: {d}\n\n{DOWNLOAD_HELP}")

        tokens = d / "tokens.txt"
        if not tokens.is_file():
            found = sorted(d.rglob("tokens.txt"))
            if not found:
                raise FileNotFoundError(f"{d} 里没有 tokens.txt")
            tokens = found[0]

        def hunt(prefix: str) -> Path:
            cands = sorted(d.glob(f"{prefix}-*.onnx")) or sorted(d.rglob(f"{prefix}-*.onnx"))
            if not cands:
                raise FileNotFoundError(
                    f"{d} 里没找到 {prefix}-*.onnx。KWS 模型要 encoder/decoder/joiner 三个文件。"
                )
            # 和 ASR 那边相反,这里 fp32 优先。同一段噪声录音上 int8 检出 3 次、
            # fp32 检出 8 次 —— 这个模型只有 3.3 M 参数,量化误差占的比重太大,
            # 噪声里本来就不多的那点声学证据直接被抹平了。
            # 省下来的那点内存(十几 MB)完全不值得拿漏检去换。
            fp32 = [p for p in cands if ".int8." not in p.name]
            return sorted(fp32 or cands)[0]

        self.model_dir = d
        self.unit = detect_modeling_unit(tokens)
        self.skipped: list[str] = []
        if keywords_file:
            # 自己写好的 keywords 文件:原样用,不做任何转换和检查
            self.keywords_file = str(Path(keywords_file).expanduser())
            if not Path(self.keywords_file).is_file():
                raise FileNotFoundError(f"keywords 文件不存在: {self.keywords_file}")
            self.keywords = [
                line.split("@", 1)[1].strip() if "@" in line else line.strip()
                for line in Path(self.keywords_file).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            self.keywords_file, self.keywords, self.skipped = _write_keywords_file(
                keywords or [DEFAULT_KEYWORD], tokens, score, threshold
            )
        if not self.keywords:
            raise ValueError(
                "所有唤醒词都编不进模型词表:\n  " + "\n  ".join(self.skipped)
                + "\n换几个常用字组成的唤醒词。"
            )

        self.threshold = threshold
        self.score = score
        self.max_active_paths = max_active_paths
        self.encoder = str(hunt("encoder"))
        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(tokens),
            encoder=self.encoder,
            decoder=str(hunt("decoder")),
            joiner=str(hunt("joiner")),
            num_threads=num_threads,
            max_active_paths=max_active_paths,
            keywords_file=self.keywords_file,
            keywords_score=score,
            keywords_threshold=threshold,
            # 关键词后面要跟够这么多个空白帧才算说完。实测 0/1/2/3 检出次数一样,
            # 那就用官方默认的 1,误触发交给 threshold 管。
            num_trailing_blanks=1,
            provider=provider,
        )
        self.stream = self.spotter.create_stream()

        self.energy_gate = energy_gate
        self.gate_rms = gate_rms
        self.block_seconds = block_seconds
        n_pre = max(1, int(GATE_PRE_ROLL_SECONDS / block_seconds))
        self._pre_roll: deque[np.ndarray] = deque(maxlen=n_pre)
        self._silence = 0.0
        self._feeding = False
        # 统计:开 --timing 时打出来,好判断能量闸到底省了多少
        self.blocks_seen = 0
        self.blocks_fed = 0
        self._since_hit = DEBOUNCE_SECONDS  # 一上来就允许触发

    def describe(self) -> str:
        s = (f"KWS {Path(self.encoder).name} / {self.unit} 建模 / 唤醒词 "
             f"{'、'.join(self.keywords)} @阈值 {self.threshold} 加分 {self.score}"
             f" beam {self.max_active_paths}")
        if self.energy_gate:
            s += f" / 能量闸 RMS>{self.gate_rms}"
        if self.skipped:
            s += "\n  [!] 编不进词表、已跳过的唤醒词:" + "、".join(self.skipped)
        return s

    def accept(self, block: np.ndarray) -> str | None:
        """喂一块 16 kHz 单声道 float32 音频。检出唤醒词时返回那个词,否则 None。"""
        self.blocks_seen += 1

        if self.energy_gate:
            rms = float(np.sqrt(np.mean(np.square(block, dtype=np.float64))))
            if rms < self.gate_rms and not self._feeding:
                # 安静:连特征都不提,只把块攒进预滚
                self._pre_roll.append(block)
                self._silence += self.block_seconds
                if self._silence >= GATE_RESET_SECONDS:
                    # 静了这么久,上一段说了一半的解码状态留着只会添乱
                    self.reset()
                    self._silence = 0.0
                return None
            if rms < self.gate_rms:
                # 刚才在喂,现在安静下来了:再喂一小会儿(尾音、字间停顿),
                # 攒够 0.3 秒静音才退回省电状态
                self._silence += self.block_seconds
                if self._silence >= GATE_PRE_ROLL_SECONDS:
                    self._feeding = False
            else:
                self._silence = 0.0
                if not self._feeding:
                    # 声音刚起来:先把预滚补进去,否则唤醒词的第一个字已经没了
                    self._feeding = True
                    while self._pre_roll:
                        self._feed(self._pre_roll.popleft())
                    self._pre_roll.clear()

        return self._feed(block)

    def _feed(self, block: np.ndarray) -> str | None:
        self.blocks_fed += 1
        self._since_hit += self.block_seconds
        self.stream.accept_waveform(
            MODEL_SAMPLE_RATE, np.ascontiguousarray(block, dtype=np.float32)
        )
        hit = None
        while self.spotter.is_ready(self.stream):
            self.spotter.decode_stream(self.stream)
            r = self.spotter.get_result(self.stream)
            if r:
                hit = r
        # 这里特意不调 reset_stream:它会把编码器的上下文一起清掉,紧跟着的
        # 下一次唤醒词就得从冷状态重新起,实测连喊时 15 次会掉到 13 次。
        # sherpa-onnx 命中后自己会把那条关键词路径清零,不会连报;
        # 真怕重复的话下面这道去抖足够了。
        if hit is not None:
            if self._since_hit < DEBOUNCE_SECONDS:
                hit = None  # 同一句话被报了两次,吃掉后一次
            else:
                self._since_hit = 0.0
        return hit

    def reset(self) -> None:
        """丢掉解码状态。唤醒之后、以及重新进入睡眠时调一次。"""
        self.stream = self.spotter.create_stream()
        self._since_hit = DEBOUNCE_SECONDS
        self._feeding = False
        self._silence = 0.0
        self._pre_roll.clear()

    def stats(self) -> str:
        if not self.blocks_seen:
            return "唤醒词: 还没收到音频"
        if not self.energy_gate:
            return f"唤醒词: 解码了 {self.blocks_seen} 块({self.blocks_seen * self.block_seconds:.0f}s 音频)"
        pct = 100.0 * self.blocks_fed / self.blocks_seen
        return (f"唤醒词: 收到 {self.blocks_seen} 块,真正解码 {self.blocks_fed} 块"
                f"({pct:.0f}%,其余被能量闸挡掉)")


# --------------------------------------------------------------------------- #
# text 模式(没有 KWS 模型时的兜底)
# --------------------------------------------------------------------------- #
class TextWakeWord:
    """拿 ASR 的定稿文字匹配唤醒词。

    省不了 ASR 的 CPU —— 识别照跑,只是不唤醒就不往 LLM/TTS 送。
    好处是不用额外下模型,而且能顺手把唤醒词后面跟的那半句命令切出来:
    "你好小智打开电灯" → 唤醒 + 命令"打开电灯",一句话就办完事。
    """

    mode = "text"

    def __init__(self, keywords: list[str] | None = None) -> None:
        self.keywords = [k.strip() for k in (keywords or [DEFAULT_KEYWORD]) if k.strip()]
        # 按拼音比,而不是按字面比:ASR 把"你好小智"听成"你好小纸 / 你好小志 /
        # 你好晓知"是常态,声母韵母却基本不会错。去掉声调再比,连"小知"和"小旨"
        # 这种调错的也一起收下。硬列同音字表列不全,这条路一步到位。
        self.by_pinyin = True
        try:
            self.targets = [(k, _toneless(k)) for k in self.keywords]
        except RuntimeError:
            # 没装 pypinyin:退回字面 + 一张手写的同音变体表,能扛住最常见那几种
            self.by_pinyin = False
            self.targets = []
            for k in self.keywords:
                # 长的排前面,"你好小智"要先于"小智"匹配上,否则命令会多带一个字
                for v in sorted(TEXT_VARIANTS.get(k, [k]), key=len, reverse=True):
                    self.targets.append((k, v))

    def describe(self) -> str:
        how = "拼音比对(不看声调、不看写法)" if self.by_pinyin else \
              f"字面比对 + {len(self.targets)} 个同音变体(装上 pypinyin 会准不少)"
        return f"文字匹配 / 唤醒词 {'、'.join(self.keywords)} / {how}(不省 ASR 的 CPU)"

    def match(self, text: str) -> tuple[bool, str]:
        """返回 (是否唤醒, 唤醒词后面剩下的话)。"""
        # 识别结果里的标点会把唤醒词切断:"你好,小智"
        flat = "".join(ch for ch in text if ch not in " ,,。..!!??、::;;~~")
        if not self.by_pinyin:
            for _, v in self.targets:
                i = flat.find(v)
                if i >= 0:
                    return True, flat[i + len(v):].strip()
            return False, ""

        # 逐字的拼音,按字对齐 —— 命中后才知道该从原文的哪个字后面切
        chars = _toneless_per_char(flat)
        for kw, target in self.targets:
            n = len(kw)
            for i in range(len(chars) - n + 1):
                if "".join(chars[i : i + n]) == target:
                    return True, flat[i + n:].strip()
        return False, ""


# --------------------------------------------------------------------------- #
# 唤醒状态机(realtime_asr.py 用的就是这个)
# --------------------------------------------------------------------------- #
class WakeGate:
    """睡 / 醒两个状态。睡着时上层不该把音频送去 ASR。

    时间全部按音频块累加,不看墙上时钟 —— 这样 --simulate 用文件喂音频时,
    超时行为和真麦克风下完全一致。
    """

    def __init__(self, detector: WakeWord, timeout: float = 15.0,
                 block_seconds: float = 0.1) -> None:
        self.detector = detector
        self.timeout = timeout
        self.block_seconds = block_seconds
        self.awake = False
        self.idle = 0.0

    def feed(self, block: np.ndarray) -> str | None:
        """睡着时调这个。返回非空表示刚被唤醒(已切到醒着状态)。"""
        hit = self.detector.accept(block)
        if hit:
            self.awake = True
            self.idle = 0.0
            self.detector.reset()
        return hit

    def tick(self, active: bool) -> bool:
        """醒着时每块调一次。active=这一块里有人说话。返回 True 表示刚睡回去。"""
        if not self.awake:
            return False
        self.idle = 0.0 if active else self.idle + self.block_seconds
        if self.idle >= self.timeout:
            self.sleep()
            return True
        return False

    def sleep(self) -> None:
        self.awake = False
        self.idle = 0.0
        self.detector.reset()


class TextWakeGate:
    """text 模式的状态机:唤醒判断发生在 ASR 出文字之后。

    超时按墙上时钟算 —— 这一路本来就是句子级的,没有逐块的时间基准。
    """

    mode = "text"

    def __init__(self, detector: TextWakeWord, timeout: float = 15.0) -> None:
        self.detector = detector
        self.timeout = timeout
        self.awake = False
        self._last = 0.0

    def feed_text(self, text: str) -> tuple[bool, str]:
        """把一句定稿文字喂进来,返回 (要不要处理这句, 实际要处理的内容)。

        睡着时命中唤醒词 → (True, 唤醒词后面剩下的话);剩下的话可能是空串,
        那就是纯唤醒,等下一句命令。
        """
        import time

        now = time.time()
        if self.awake and now - self._last > self.timeout:
            self.awake = False  # 太久没说话,睡回去
        if self.awake:
            self._last = now
            return True, text
        hit, rest = self.detector.match(text)
        if hit:
            self.awake = True
            self._last = now
            return True, rest
        return False, ""

    def sleep(self) -> None:
        self.awake = False


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
def main() -> int:
    import argparse
    import queue
    import threading
    import time

    import winutil

    winutil.setup_console()

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="唤醒词模块自检")
    p.add_argument("--model", default=None, help="KWS 模型目录,默认在 models\\ 里找")
    p.add_argument("--keywords", nargs="+", default=[DEFAULT_KEYWORD], help="唤醒词")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--score", type=float, default=DEFAULT_SCORE)
    p.add_argument("--device", default=None, help="输入设备编号")
    p.add_argument("--wav", default=None, help="用音频文件测,不开麦克风")
    p.add_argument("--energy-gate", action="store_true",
                   help="开能量闸:安静的块不解码,CPU 从 1.2%% 降到 0.4%%,代价是会漏检")
    p.add_argument(
        "--max-active-paths",
        type=int,
        default=DEFAULT_MAX_ACTIVE_PATHS,
        help="解码保留几条候选路径。调小省不了多少 CPU,却会明显漏检",
    )
    p.add_argument(
        "--keywords-file",
        default=None,
        help="自己写好的 keywords 文件,原样喂给 sherpa-onnx(格式照抄模型目录里的 keywords.txt)",
    )
    args = p.parse_args()

    model = args.model or find_kws_model(here / "models")
    if not model:
        print(DOWNLOAD_HELP)
        return 1

    t0 = time.time()
    ww = WakeWord(
        model,
        keywords=args.keywords,
        threshold=args.threshold,
        score=args.score,
        energy_gate=args.energy_gate,
        keywords_file=args.keywords_file,
        max_active_paths=args.max_active_paths,
    )
    print(f"唤醒词就绪,用时 {time.time() - t0:.1f}s")
    print(f"  {ww.describe()}\n")

    n_hit = 0
    t_start = time.time()

    if args.wav:
        import soundfile as sf

        audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
        audio = np.ascontiguousarray(audio[:, 0])
        if sr != MODEL_SAMPLE_RATE:
            from math import gcd

            from scipy.signal import resample_poly

            g = gcd(int(sr), MODEL_SAMPLE_RATE)
            audio = resample_poly(audio, MODEL_SAMPLE_RATE // g, int(sr) // g).astype(
                np.float32
            )
        step = int(0.1 * MODEL_SAMPLE_RATE)
        t0 = time.time()
        for i in range(0, len(audio), step):
            hit = ww.accept(audio[i : i + step])
            if hit:
                n_hit += 1
                print(f"  [{i / MODEL_SAMPLE_RATE:6.1f}s] 检出「{hit}」")
        dur = len(audio) / MODEL_SAMPLE_RATE
        cost = time.time() - t0
        print(f"\n音频 {dur:.1f}s,耗时 {cost:.2f}s,RTF {cost / max(dur, 1e-9):.3f}")
    else:
        import sounddevice as sd

        device = args.device
        if device is not None and str(device).lstrip("-").isdigit():
            device = int(device)
        q: "queue.Queue[np.ndarray]" = queue.Queue()
        stop = threading.Event()
        print("对着麦克风说「" + "」或「".join(args.keywords) + "」(Ctrl-C 退出)\n")
        with sd.InputStream(
            device=device,
            samplerate=MODEL_SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=int(0.1 * MODEL_SAMPLE_RATE),
            callback=lambda ind, f, t, s: q.put(ind[:, 0].copy()),
        ):
            try:
                while not stop.is_set():
                    try:
                        hit = ww.accept(q.get(timeout=0.5))
                    except queue.Empty:
                        continue
                    if hit:
                        n_hit += 1
                        print(f"  [{time.time() - t_start:6.1f}s] 检出「{hit}」 x{n_hit}")
            except KeyboardInterrupt:
                pass

    print(f"\n共检出 {n_hit} 次")
    print(ww.stats())
    return 0


if __name__ == "__main__":
    sys.exit(main())
