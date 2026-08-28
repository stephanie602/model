#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
命令词匹配:把识别出来的文字对到一张固定的命令表上。

为什么需要这一层 —— 看噪声环境下 SenseVoice 的实际输出:

    打开电灯  →  大楷颠等
    关闭电灯  →  关闭点灯
    你好小智  →  你好小纸

**全是同音错字**。声学上模型听对了,错的是选字 —— 噪声让语言模型的先验占了上风,
"大楷"在通用语料里比"打开电灯"常见得多。拿汉字去比对,这三条一条都对不上;
换成拼音去比,三条全部精确命中("大楷颠等"和"打开电灯"的无声调拼音完全相同)。

所以这里按无声调拼音做匹配,分三档:

    完全相同        直接收下(上面那三条都走这一档)
    编辑距离很小    收下,并把距离报出来(轻微听错,比如少一个韵尾)
    差得多          不收 —— 宁可漏,不能把"打开电脑"匹配成"打开电灯"

第三档的分界线是量出来的,不是拍的:"打开电脑"(dakaidiannao)和"打开电灯"
(dakaidiandeng)的字符距离是 4/12 = 33%,而真正的同音错字基本都在 10% 以内。
默认阈值 0.15 把这两类分得很干净。

和热词的关系:热词(asr_sherpa.py)是在**解码时**给命令词加分,让模型更倾向于
选对字;这里是在**解码后**把已经选错的字捞回来。两者互补,固定命令词场景建议都用。
热词只对 transducer 模型有效,而这一层对哪个模型都能用。

用法:
    .venv\\Scripts\\python.exe intent.py "大楷颠等"                 # 单条测
    .venv\\Scripts\\python.exe intent.py --list                     # 看命令表
    .venv\\Scripts\\python.exe realtime_asr.py --commands hotwords.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

# 匹配阈值:拼音串的归一化编辑距离超过它就不算命中。
# 0.15 是量出来的 —— 同音错字通常 0-10%,而"电脑/电灯"这种真不同的词在 30% 以上。
DEFAULT_THRESHOLD = 0.15

# 识别结果里的标点会把命令切断("打开,电灯"),比对前先剥掉
_PUNCT = " ,，。.、!！?？:：;；~～\"'“”‘’()（）[]【】…-—_\n\t"


def _clean(text: str) -> str:
    return "".join(c for c in text if c not in _PUNCT)


# 声母混淆组:发音部位相同,噪声里最容易互相听错。同组内算"半个匹配"。
_INITIAL_GROUPS = [
    {"d", "t"}, {"b", "p"}, {"g", "k"}, {"z", "c", "zh", "ch"},
    {"s", "sh", "x"}, {"n", "l"}, {"f", "h"}, {"j", "q"}, {"r", "l"}, {"m", "n"},
]
# 韵母混淆组:前后鼻音不分是最常见的一类,南方口音和噪声都会造成
_FINAL_GROUPS = [
    {"an", "ang"}, {"en", "eng"}, {"in", "ing"}, {"ian", "iang"},
    {"uan", "uang"}, {"uen", "ueng"}, {"e", "o"}, {"ei", "ui"},
]

# 一个音节和另一个音节的差异代价:0 完全相同,0.5 混淆组内,inf 不沾边
_MATCH, _NEAR, _MISS = 0.0, 0.5, float("inf")

def max_near_cost(n_syllables: int) -> float:
    """允许多少"音节听混"的代价 —— 按命令词长度放宽。

    短命令必须严格,长命令可以宽松:两个音节的"制冷"只要允许一处混淆,
    "只能"(zhi neng,n/l 同组)就能撞上来 —— 实测就是这么冒出来的误匹配。
    四个音节的"打开电灯"有三个音节顶着,允许一两处混淆也不会被别的词撞到。
    """
    if n_syllables <= 2:
        return 0.0      # 必须逐个音节完全相同
    if n_syllables == 3:
        return 0.5      # 允许一处混淆
    return 1.0          # 四个音节以上:两处
# 命令词至少要被覆盖到这个比例,否则"灯"一个字就能匹配上"打开电灯"
MIN_COVERAGE = 0.6
# 覆盖到的音节数下限,防止两个字的噪声碎片乱匹配
MIN_SYLLABLES = 2


def _split(syl: str) -> tuple[str, str]:
    """把一个音节拆成声母 + 韵母。"deng" → ("d", "eng")"""
    for n in (2, 1):  # zh/ch/sh 是两个字母,要先试长的
        if len(syl) > n and syl[:n] in {
            "zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l",
            "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w",
        }:
            return syl[:n], syl[n:]
    return "", syl


def _syllable_cost(a: str, b: str) -> float:
    """两个音节的差异代价。这是整个匹配器的核心判据。"""
    if a == b:
        return _MATCH
    ia, fa = _split(a)
    ib, fb = _split(b)
    if fa == fb and any(ia in g and ib in g for g in _INITIAL_GROUPS):
        return _NEAR  # 韵母相同,声母是易混的("他开"vs"打开")
    if ia == ib and any(fa in g and fb in g for g in _FINAL_GROUPS):
        return _NEAR  # 声母相同,前后鼻音之类
    return _MISS


def _syllables(text: str) -> list[str]:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError as exc:
        raise RuntimeError(
            "命令词匹配要用 pypinyin(纯 Python,几百 KB):\n"
            "    .venv\\Scripts\\python.exe -m pip install pypinyin"
        ) from exc

    return [s for s in lazy_pinyin(text, style=Style.NORMAL) if s.strip()]


class Command:
    __slots__ = ("text", "syl", "action")

    def __init__(self, text: str, action: str | None = None) -> None:
        self.text = text
        self.syl = _syllables(text)
        self.action = action or text

    def __repr__(self) -> str:
        return f"Command({self.text!r}, {' '.join(self.syl)})"


class Match:
    __slots__ = ("command", "distance", "score", "rest", "start", "covered")

    def __init__(self, command: Command, distance: int, score: float, rest: str,
                 start: int = 0, covered: int = 0) -> None:
        self.command = command
        self.distance = distance
        self.score = score  # 0-1,1 是完全相同
        self.rest = rest    # 命令词之外剩下的话
        self.start = start      # 命中位置在识别结果的第几个音节
        self.covered = covered  # 对上了命令词里的几个音节

    def __repr__(self) -> str:
        return f"Match({self.command.text!r}, 相似度 {self.score:.0%})"


class Matcher:
    """一张命令表,外加"把识别结果对到表上"的能力。"""

    def __init__(self, commands: list[str] | None = None,
                 threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self.commands = [Command(c) for c in (commands or []) if c.strip()]

    @classmethod
    def from_file(cls, path: str | Path, threshold: float = DEFAULT_THRESHOLD) -> "Matcher":
        """读命令表。格式和 hotwords.txt 一样,好让两边共用一份 ——
        热词在解码时加分,这里在解码后兜底,本来就该是同一张表。

            打开电灯 :3.0     # 行尾的 :分数 是热词权重,这里忽略
            关灯              # 也可以只写词
            # 井号开头是注释
        """
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"命令表不存在: {p}")
        items: list[str] = []
        for raw in p.read_text(encoding="utf-8").splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            items.append(s.rsplit(":", 1)[0].strip() if ":" in s else s)
        return cls(items, threshold)

    def describe(self) -> str:
        return f"命令词 {len(self.commands)} 条 @ 阈值 {self.threshold}"

    def match(self, text: str) -> Match | None:
        """把一句识别结果对到命令表上。对不上返回 None。

        判据不是"整串编辑距离",而是:**识别出来的音节必须逐个对得上命令词里
        连续的一段**,命令词的头尾可以缺。这条规则把两类错法分开了:

            打开电灯 → 开电灯      缺了头,剩下的逐个对得上   → 收
            打开电灯 → 打开电      缺了尾,同上               → 收
            打开电灯 → 他开店等    "他/打"是易混声母,算半个   → 收
            打开电灯 → 打开电脑    "脑/灯"八竿子打不着       → 不收

        最后一条正是不能用编辑距离的原因:"打开电脑"和"打开电灯"距离只有 1 个
        音节,和"打开电"缺 1 个音节的距离一模一样,但一个是别的命令、一个是同一条
        命令被咬掉了尾巴。看"缺"还是"错"才分得开。
        """
        flat = _clean(text)
        if not flat or not self.commands:
            return None
        hyp = _syllables(flat)
        if not hyp:
            return None

        # 跨命令排序,三级:
        #   1. 命中位置越靠前越好 —— 一句话里连着说两条命令时,先说的那条先出,
        #      剩下的交给 match_all 接着捞;
        #   2. 对上的音节越多越好 —— "开电灯"里"打开电灯"对上 3 个、裸词"电灯"
        #      只对上 2 个,该选前者。(hotwords.txt 里那些裸词本来是给解码器
        #      加分用的,混在命令表里就会来抢;这一级把它们压下去。)
        #   3. 最后才比分数。
        best: Match | None = None
        best_key = ()
        for cmd in self.commands:
            m = self._align(hyp, cmd, flat)
            if m is None:
                continue
            key = (-m.start, m.covered, m.score)
            if best is None or key > best_key:
                best, best_key = m, key
        return best

    def _align(self, hyp: list[str], cmd: Command, flat: str) -> Match | None:
        """把 hyp 的前 k 个音节对到 cmd 里连续的一段上,取覆盖最多的那种对法。

        hyp 前面多出来的音节不管(那是别的话),后面剩下的进 rest ——
        "打开电灯谢谢"里的"谢谢"、以及连着说的下一条命令都靠这个往下传。
        """
        n = len(cmd.syl)
        best: tuple[float, int, int] | None = None  # (score, 用掉的 hyp 音节数, 覆盖数)
        # i = 命令词从第几个音节开始对(允许缺头),j = hyp 从第几个音节开始
        for i in range(n):
            for j in range(len(hyp)):
                cost = 0.0
                k = 0
                while i + k < n and j + k < len(hyp):
                    c = _syllable_cost(hyp[j + k], cmd.syl[i + k])
                    if c is _MISS or cost + c > max_near_cost(n):
                        break
                    cost += c
                    k += 1
                if k == 0:
                    continue
                covered = k
                if covered < min(MIN_SYLLABLES, n) or covered / n < MIN_COVERAGE:
                    continue
                # 命令词的尾巴没对完,而识别结果还在往下说 —— 那下一个音节是在
                # **反对**这条命令,不是"没听全"。"打开电脑"就死在这里:
                # 前三个音节和"打开电灯"一样,但第四个是"脑"不是没有。
                # 只有识别结果自己说完了(被咬掉尾巴),才算 truncation。
                if i + covered < n and j + covered < len(hyp):
                    continue
                # 缺的头尾按半个音节记账,和"混淆"同价 —— 都是"没听全",不是"听成别的"
                missing = (n - covered) * 0.5
                score = max(0.0, 1.0 - (cost + missing) / n)
                # 同一条命令内部:起点越靠前越好(先说的先算),再看覆盖和分数
                key = (-j, covered, score)
                if best is None or key > best[0]:
                    best = (key, score, j + k, covered, j)
        if best is None:
            return None
        _, score, used, covered, start = best
        # 剩下的话按音节数比例粗切 —— 音节和汉字数通常一致(一字一音),
        # 有英文数字时会偏,所以夹一下边界
        cut = min(len(flat), max(0, round(len(flat) * used / max(len(hyp), 1))))
        return Match(cmd, n - covered, score, flat[cut:].strip(), start, covered)

    def match_all(self, text: str, limit: int = 6) -> list[Match]:
        """一句话里可能连着好几条命令("打开电灯关闭电视"),挨个往下捞。

        VAD 按停顿分句,而人说命令时常常不停顿 —— 只匹配开头那条的话,
        后面的命令会带着同音错字原样漏下去。
        """
        out: list[Match] = []
        rest = text
        while rest and len(out) < limit:
            m = self.match(rest)
            if m is None:
                break
            out.append(m)
            if not m.rest or m.rest == rest:
                break
            rest = m.rest
        return out


def main() -> int:
    import argparse

    import winutil

    winutil.setup_console()

    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="命令词匹配自检")
    p.add_argument("text", nargs="*", help="要匹配的识别结果")
    p.add_argument("--commands", default=str(here / "hotwords.txt"), help="命令表文件")
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--list", action="store_true", help="打印命令表后退出")
    args = p.parse_args()

    m = Matcher.from_file(args.commands, args.threshold)
    if args.list:
        print(m.describe())
        for c in m.commands:
            print(f"  {c.text:12s} {c.pinyin}")
        return 0

    texts = args.text or [
        "大楷颠等", "关闭点灯", "打开电脑", "调亮一点", "今天天气不错", "大开点灯谢谢",
    ]
    print(f"{m.describe()}\n")
    for t in texts:
        r = m.match(t)
        if r:
            rest = f"  余下「{r.rest}」" if r.rest else ""
            print(f"  「{t}」 → 「{r.command.text}」 相似度 {r.score:.0%}"
                  f"(距离 {r.distance}){rest}")
        else:
            print(f"  「{t}」 → 不是命令")
    return 0


if __name__ == "__main__":
    sys.exit(main())
