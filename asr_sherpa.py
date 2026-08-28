#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sherpa-onnx 识别后端,Windows 版。支持两类模型,按模型目录里有什么自动判断:

    transducer (zipformer)  encoder/decoder/joiner 三个文件
                            ★ 支持热词:解码时给指定词加分,能把"打开电灯"从
                              "打开电脑"手里抢回来。固定命令词场景必须用这个。
    SenseVoice (CTC)        model.onnx 一个文件
                            中/英/日/韩/粤五语,快(RTF 0.03),但不支持热词。

为什么热词只有 transducer 有:SenseVoice 是 CTC 模型,每帧独立输出、只有
greedy_search,没有可以加权的解码路径。transducer 的 modified_beam_search 在每步
维护多条候选,热词命中时给该路径加分 —— 有地方可加分,才谈得上热词。

热词能解决的是"声学证据不够强时被语言先验带偏":"电灯"和"电脑"声学上差得远,
但"打开电脑"在训练语料里常见得多,底噪一大解码就滑过去了。加了热词等于告诉模型
"这个场景下电灯更可能",把那个先验掰回来。

单独测:
    .venv\\Scripts\\python.exe asr_sherpa.py models\\sherpa-onnx-zipformer-... test.wav
    .venv\\Scripts\\python.exe asr_sherpa.py <模型目录> test.wav --hotwords hotwords.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

MODEL_SAMPLE_RATE = 16000

# 热词默认加分。太小不起作用,太大会把不该出现的词硬塞进来(比如安静时凭空
# 蹦出"打开空调")。1.5 是 sherpa-onnx 的默认值,2.0-3.0 是命令词场景的常用区间。
DEFAULT_HOTWORDS_SCORE = 2.0


def _pick(cands: list[Path]) -> Path | None:
    """一组同名的 int8 / fp32 文件里挑一个。int8 优先 —— 板子上内存和带宽比算力金贵,
    而中文识别上 int8 和 fp32 的差别基本听不出来。"""
    if not cands:
        return None
    int8 = [p for p in cands if ".int8." in p.name]
    return sorted(int8 or cands)[0]


def find_tokens(model_dir: Path) -> Path:
    """找 tokens.txt。sherpa-onnx 官方包放在根目录,icefall 导出的包放在
    data/lang_char/ 下面,所以根目录找不到就往下递归搜一层。"""
    p = model_dir / "tokens.txt"
    if p.is_file():
        return p
    found = sorted(model_dir.rglob("tokens.txt"))
    if found:
        return found[0]
    raise FileNotFoundError(f"{model_dir} 及其子目录里都没有 tokens.txt")


def detect_model(model_dir: Path) -> tuple[str, dict]:
    """认模型类型,返回 (kind, 文件路径字典)。

    同样要兼容两种布局:官方包三个 .onnx 平铺在根目录,icefall 包塞在 exp/ 里。
    """
    def hunt(prefix: str) -> Path | None:
        return _pick(sorted(model_dir.glob(f"{prefix}-*.onnx"))) or _pick(
            sorted(model_dir.rglob(f"{prefix}-*.onnx"))
        )

    enc, dec, joi = hunt("encoder"), hunt("decoder"), hunt("joiner")
    if enc and dec and joi:
        return "transducer", {"encoder": str(enc), "decoder": str(dec), "joiner": str(joi)}

    for name in ("model.int8.onnx", "model.onnx"):
        p = model_dir / name
        if p.is_file():
            return "sense_voice", {"model": str(p)}

    onnx = sorted(model_dir.glob("*.onnx"))
    if onnx:
        return "sense_voice", {"model": str(onnx[0])}

    raise FileNotFoundError(
        f"{model_dir} 里没找到可用的 .onnx 模型。\n"
        f"transducer 需要 encoder/decoder/joiner 三个文件,SenseVoice 需要 model.onnx。\n"
        f"注意:名字里带 rk3588 的包装的是 .rknn(瑞芯微 NPU 专用),Windows 上用不了。"
    )


class SherpaRecognizer:
    """整段音频进、文本出。两类模型共用这一个接口,上层不用管底下是哪种。"""

    def __init__(
        self,
        model_dir: str,
        language: str | None = None,
        num_threads: int = 4,
        provider: str | None = None,
        use_itn: bool = True,
        debug: bool = False,
        hotwords_file: str | None = None,
        hotwords_score: float = DEFAULT_HOTWORDS_SCORE,
        blank_penalty: float = 0.0,
    ) -> None:
        import sherpa_onnx

        d = Path(model_dir).expanduser()
        if not d.is_dir():
            raise FileNotFoundError(f"模型目录不存在: {d}")

        tokens = find_tokens(d)
        provider = provider or "cpu"
        self.kind, files = detect_model(d)
        self.provider = provider
        self.hotwords_file = None
        self.hotwords_score = hotwords_score
        self.n_hotwords = 0
        self.skipped_hotwords: list[str] = []

        if self.kind == "transducer":
            # 热词只在 modified_beam_search 下有效。没给热词就走 greedy_search ——
            # 快一截,而且没有热词时 beam search 对准确率的提升很有限。
            decoding = "greedy_search"
            hw = ""
            if hotwords_file:
                p = Path(hotwords_file).expanduser()
                if not p.is_file():
                    raise FileNotFoundError(f"热词文件不存在: {p}")
                # sherpa-onnx 的热词文件格式不支持注释 —— 它会把 # 开头的整行
                # 当成一条热词去编码,然后报 "Failed to encode some hotwords" 就跳过。
                # 注释对这种要手工调的表太有用了,所以自己先剥一遍,写到临时文件再喂过去。
                hw, self.n_hotwords, self.skipped_hotwords = _clean_hotwords(
                    p, _load_vocab(tokens)
                )
                if self.n_hotwords:
                    decoding = "modified_beam_search"
                    self.hotwords_file = str(p)
                else:
                    hw = ""

            self.rec = sherpa_onnx.OfflineRecognizer.from_transducer(
                encoder=files["encoder"],
                decoder=files["decoder"],
                joiner=files["joiner"],
                tokens=str(tokens),
                num_threads=num_threads,
                decoding_method=decoding,
                hotwords_file=hw,
                hotwords_score=hotwords_score,
                # modeling_unit=cjkchar:热词文件里直接写"打开电灯"就行,
                # sherpa-onnx 自己按汉字切成 token。写成"打 开 电 灯"也认。
                modeling_unit="cjkchar",
                blank_penalty=blank_penalty,
                provider=provider,
                debug=debug,
            )
            self.decoding_method = decoding
            self.language = None  # 这个模型是纯中文的,没有语言选项
            self.model_path = files["encoder"]
        else:
            # SenseVoice 的 language 取值是 auto/zh/en/ja/ko/yue,空串就是 auto
            self.language = _sense_voice_language(language)
            if hotwords_file:
                # 不静默忽略 —— 用户以为热词生效了、结果一点变化没有,最难查
                raise ValueError(
                    "SenseVoice 不支持热词(CTC 模型没有可加权的解码路径)。\n"
                    "要用热词请换 zipformer transducer 模型,比如\n"
                    "  sherpa-onnx-zipformer-multi-zh-hans-2023-9-2"
                )
            self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=files["model"],
                tokens=str(tokens),
                num_threads=num_threads,
                use_itn=use_itn,
                language=self.language,
                provider=provider,
                debug=debug,
            )
            self.decoding_method = "greedy_search"
            self.model_path = files["model"]

    def describe(self) -> str:
        """一行摘要,启动时打给用户看。"""
        s = f"{self.kind} / {Path(self.model_path).name} / {self.decoding_method}"
        if self.n_hotwords:
            s += f" / 热词 {self.n_hotwords} 条 @ {self.hotwords_score}"
        if self.skipped_hotwords:
            s += f"\n  [!] {len(self.skipped_hotwords)} 条热词编不进词表,已跳过(模型仍能识别出这些词,只是加不了分):"
            for x in self.skipped_hotwords:
                s += f"\n      {x}"
        return s

    def transcribe(self, audio: np.ndarray, partial: bool = False) -> str:
        audio = np.ascontiguousarray(audio, dtype=np.float32)

        # 削顶保护:峰值超过 1.0 的音频喂进特征提取会出问题
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if peak > 1.0:
            audio = audio / peak

        stream = self.rec.create_stream()
        stream.accept_waveform(MODEL_SAMPLE_RATE, audio)
        self.rec.decode_stream(stream)
        return (stream.result.text or "").strip()


def _load_vocab(tokens: Path) -> set[str]:
    """tokens.txt 里所有 token 的字面。用来提前判断热词编不编得进去。"""
    vocab = set()
    for line in tokens.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line:
            continue
        # 格式是 "<token> <id>",token 本身可能含空格以外的任何字符,所以从右边切
        parts = line.rsplit(" ", 1)
        if len(parts) == 2:
            vocab.add(parts[0])
    return vocab


def _clean_hotwords(path: Path, vocab: set[str]) -> tuple[str, int, list[str]]:
    """剥掉注释和空行,顺便剔掉编不进去的热词。

    返回 (临时文件路径, 有效条数, 被剔掉的原始行)。

    为什么要自己先查一遍:sherpa-onnx 遇到词表里没有的字只会往 stderr 打一行英文
    (Cannot find ID for token X),然后 "skip them already" 继续跑 —— 用户完全
    看不出自己写的热词有一半根本没生效。

    编不进去的原因是 modeling_unit=cjkchar 会把热词按汉字逐个查表,而有些模型
    (比如 multi-zh-hans)的词表只收了最常用的 1400 多个字,其余汉字是靠 256 个
    UTF-8 字节回退 token 拼出来的 —— 模型输出得了,cjkchar 却查不到。

    临时文件挂在 tempfile 目录里,sherpa-onnx 构造时就读完了,不用自己管生命周期。
    """
    import tempfile

    keep: list[str] = []
    skipped: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # 行尾的 ":分数" 是权重,不参与查表
        phrase = s.rsplit(":", 1)[0].strip() if ":" in s else s
        bad = [c for c in phrase.replace(" ", "") if "一" <= c <= "鿿" and c not in vocab]
        if bad:
            skipped.append(f"{phrase}(词表缺 {''.join(sorted(set(bad)))})")
            continue
        keep.append(s)

    if not keep:
        return "", 0, skipped

    fd, tmp = tempfile.mkstemp(prefix="hotwords_", suffix=".txt", text=True)
    with open(fd, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    return tmp, len(keep), skipped


def _sense_voice_language(language: str | None) -> str:
    """把上层那套语言名翻成 SenseVoice 认的代码。认不出来就交给 auto。"""
    if not language:
        return ""
    table = {
        "chinese": "zh", "mandarin": "zh", "zh": "zh",
        "cantonese": "yue", "yue": "yue",
        "english": "en", "en": "en",
        "japanese": "ja", "ja": "ja",
        "korean": "ko", "ko": "ko",
        "auto": "", "none": "",
    }
    return table.get(language.strip().lower(), "")


def main() -> int:
    import argparse
    import time

    import winutil

    winutil.setup_console()

    p = argparse.ArgumentParser(description="sherpa-onnx 识别后端自检")
    p.add_argument("model_dir", help="模型目录(transducer 或 SenseVoice)")
    p.add_argument("wav", help="要识别的 wav 文件")
    p.add_argument("--language", default="Chinese", help="仅 SenseVoice 有效")
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--provider", default=None, help="默认 cpu")
    p.add_argument("--hotwords", default=None, help="热词文件(仅 transducer)")
    p.add_argument("--hotwords-score", type=float, default=DEFAULT_HOTWORDS_SCORE)
    args = p.parse_args()

    import soundfile as sf

    t0 = time.time()
    rec = SherpaRecognizer(
        args.model_dir,
        language=args.language,
        num_threads=args.num_threads,
        provider=args.provider,
        hotwords_file=args.hotwords,
        hotwords_score=args.hotwords_score,
    )
    print(f"模型就绪,用时 {time.time() - t0:.1f}s")
    print(f"  {rec.describe()}")

    audio, sr = sf.read(args.wav, dtype="float32", always_2d=True)
    audio = np.ascontiguousarray(audio[:, 0])
    if sr != MODEL_SAMPLE_RATE:
        from math import gcd

        from scipy.signal import resample_poly

        g = gcd(sr, MODEL_SAMPLE_RATE)
        audio = resample_poly(audio, MODEL_SAMPLE_RATE // g, sr // g).astype(np.float32)

    t0 = time.time()
    text = rec.transcribe(audio)
    dur = len(audio) / MODEL_SAMPLE_RATE
    cost = time.time() - t0
    print(f"\n{text}\n")
    print(f"音频 {dur:.1f}s,解码 {cost:.2f}s,RTF {cost / max(dur, 1e-9):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
