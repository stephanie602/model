#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""设备手册问答:把厂家的 PDF 手册变成工人能用中文问的知识库。

场景是这样的:手册是**英文**的(Synda Pack VFFS 系列,11 页),而车间里的工人说中文。
现场最常问的也不是"这个按钮叫什么",而是"横封不牢怎么办""报警说伺服过载什么意思"——
答案就在手册第 10、11 页那两张"故障—原因—解决"表里。

三个决定,都是被这个场景逼出来的:

  **建库时就把手册翻成中文**,不是查询时翻。查询时翻译等于每问一句都多等几秒,
  而且同一段话每次翻得都不一样;建库时翻一遍,之后检索和回答全在中文里做,
  现场零等待。代价是换手册要重新入库一次(一两分钟)。

  **检索用 BM25,不用向量**。手册全文才一万七千字符,而工人的问法和手册里的词
  高度重合("横封""切刀""色标")—— 关键词检索在这个规模上又准又快,还不用再下
  一个几百 MB 的 embedding 模型、不用维护向量库。中文按**字二元组**切词
  (不用分词器),"横封不牢"能切出"横封/封不/不牢",和手册里的"横封"对得上。

  **答案必须带页码**。工人拿着这句话要去翻纸质手册、要跟维修说事,"手册第 10 页写着"
  比一句流畅的转述有用得多;也让人能自己核对模型有没有编。

用法:
    .venv\\Scripts\\python.exe manual.py --build            # 入库(要先起 LLM server)
    .venv\\Scripts\\python.exe manual.py "横封不牢怎么办"    # 查一条,看检索到什么
    .venv\\Scripts\\python.exe manual.py --list             # 看库里有哪些段
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 入库结果。放在项目目录里而不是 models/:它是几十 KB 的文本,跟着代码走比跟着模型走合理
INDEX_FILE = HERE / "manual_zh.json"

# 目标块长(字符)。太长:检索命中一大段,答非所问的内容跟着塞进上下文;
# 太短:表格的一行被切开,"原因"和"解决"分了家。表格行本来就短,这个值主要管散文段落。
CHUNK_CHARS = 700

# 术语表。4B 模型翻机械术语会飘("horizontal seal"翻成"水平密封"、
# "photosensor"翻成"光敏电阻"),而工人认的是车间里的叫法。这张表钉死几个高频词,
# 直接写进翻译的系统提示里 —— 比事后校对省事得多。
GLOSSARY = [
    ("vertical form-fill-seal machine / VFFS machine", "立式包装机"),
    ("form-fill-seal machine", "包装机"),
    ("horizontal seal", "横封"),
    ("vertical seal", "纵封"),
    ("heating rod", "加热棒"),
    ("photosensor", "色标光电眼"),
    ("marked color", "色标"),
    ("film", "卷膜"),
    ("pouch / sachet", "包袋"),
    ("pouch length", "袋长"),
    ("pull film / film pulling", "拉膜"),
    ("pull belt", "拉膜带"),
    ("cutter", "切刀"),
    ("gusset", "捏角"),
    ("servo motor", "伺服电机"),
    ("solenoid valve", "电磁阀"),
    ("middle hopper", "中间料斗"),
    ("weigher", "称重器"),
    ("elevator", "提升机"),
    ("touch panel", "触摸屏"),
    ("emergency switcher", "急停开关"),
    ("alarm reset", "报警复位"),
    ("formulation", "配方"),
    ("pneumatic system", "气路系统"),
    ("commissioning / debug", "调试"),
]

TRANSLATE_SYSTEM = (
    "你是包装机械行业的中英翻译。把用户给的英文手册片段翻成简体中文,"
    "给车间操作工看,所以用车间里的说法,不要书面语。\n"
    "规矩:\n"
    "1. 逐句翻完,不要漏、不要总结、不要加解释;\n"
    "2. 数字、单位、参数名、按钮名原样保留(比如 0.05~0.30S、380V、[START]);\n"
    "3. 表格行保持一行,用「现象|原因|解决」这样的顺序,不要重排;\n"
    "4. 只输出译文,不要写「以下是翻译」这类话。\n"
    "术语必须按这张表:\n"
    + "；".join(f"{en} = {zh}" for en, zh in GLOSSARY)
)


# --------------------------------------------------------------------------- #
# 建库:PDF → 分块 → 翻译 → JSON
# --------------------------------------------------------------------------- #

def extract_pages(pdf: Path) -> list[str]:
    """逐页抽文字。"""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "入库要用 pypdf(纯 Python):\n"
            "    .venv\\Scripts\\python.exe -m pip install pypdf"
        ) from exc
    reader = PdfReader(str(pdf))
    return [(p.extract_text() or "") for p in reader.pages]


# 章节标题:只认罗马数字和 "(II)." 这两种。
#
# 原来还认 "A. XXX" ——结果把故障表里的"A. LOW TEMPERATURE OF HEATING ROD"
# 也当成了标题,一条原因单独成块,和它对应的现象、解决全被拆散。表格是这份手册
# 最有用的部分,宁可少切,不能切碎。
_HEADING = re.compile(r"^\s*(?:[IVX]+\.|\([IVX]+\)\.?)\s*[A-Z][A-Z &/\-\.]{4,}\s*$")

# 表格区的表头。pypdf 把表格按单元格逐行吐出来,行与行的对应关系已经丢了 ——
# 硬拼只能靠猜,而猜错的后果是"横封不牢"配上"更换卷膜"。所以这两张表不走
# 自动流水线,改成下面 TROUBLES / ALARMS 那两份人工录入的中文结构化数据;
# 抽文字时把表格区跳过,免得同样的内容以碎片形式混进检索。
_TABLE_HEAD = re.compile(
    r"(NO\.?\s+TROUBLE|ALARM\s+CONTENT\s+REASON|TROUBLE\s+REASON)", re.I
)


def chunk_page(text: str, page: int) -> list[dict]:
    """把一页切成若干块。按空行分段,遇到章节标题强制起新块,超长了再切。

    pypdf 抽表格时会把"散文"和"表格单元"分开输出(实测第 11 页正文在前、
    表格行在后),所以这里不做二维还原 —— 表格行本身就是一行一条,
    按行保留反而正好是一条条可检索的记录。
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    chunks: list[dict] = []
    buf: list[str] = []
    in_table = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if len(body) < 20:  # 页码、页眉这类碎片不值得入库
            return
        chunks.append({"page": page, "en": body})

    for ln in lines:
        if _TABLE_HEAD.search(ln):
            flush()
            in_table = True          # 进了表格区,交给人工录入的那份数据
            continue
        if _HEADING.match(ln):
            in_table = False         # 表格区到下一个章节标题为止
            if buf:
                flush()
        if in_table:
            continue
        buf.append(ln)
        if sum(len(x) for x in buf) >= CHUNK_CHARS and not ln.strip():
            flush()
    flush()
    # 超长块再切一刀:第 4 页那串按钮说明有 2663 字符,一次翻译要吃掉大半个
    # 4096 上下文(输入 + 输出),模型容易在中途开始摘要。按行切到 900 以内,
    # 每块仍然是完整的几条说明
    out: list[dict] = []
    for c in chunks:
        if len(c["en"]) <= CHUNK_CHARS + 200:
            out.append(c)
            continue
        buf2: list[str] = []
        for ln in c["en"].splitlines():
            if buf2 and sum(len(x) for x in buf2) + len(ln) > CHUNK_CHARS:
                out.append({"page": page, "en": "\n".join(buf2).strip()})
                buf2 = []
            buf2.append(ln)
        if buf2:
            out.append({"page": page, "en": "\n".join(buf2).strip()})
    return out


def translate(text: str, url: str, model: str, timeout: float = 180.0) -> str:
    """把一块英文翻成中文。走本地 llama-server,不联网。"""
    import requests

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TRANSLATE_SYSTEM},
            {"role": "user", "content": text},
        ],
        # temperature 0:同一段每次翻得一样,重新入库时能和上一版逐行对比
        "temperature": 0.0,
        # 中文通常比英文短,但表格会展开,给够 2 倍
        "max_tokens": min(2048, max(256, len(text))),
    }
    r = requests.post(f"{url.rstrip('/')}/chat/completions", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def build(pdf: Path, url: str, model: str = "local", progress=print) -> dict:
    """入库:抽 → 切 → 翻 → 存。返回写下去的那份索引。"""
    pages = extract_pages(pdf)
    chunks: list[dict] = []
    for i, text in enumerate(pages, 1):
        chunks.extend(chunk_page(text, i))
    progress(f"{pdf.name}:{len(pages)} 页 → {len(chunks)} 块")

    t0 = time.time()
    for i, c in enumerate(chunks, 1):
        t1 = time.time()
        c["zh"] = translate(c["en"], url, model)
        progress(f"  [{i}/{len(chunks)}] 第 {c['page']} 页 "
                 f"{len(c['en'])} 字符 → {len(c['zh'])} 字,{time.time() - t1:.1f}s")
    index = {
        "source": pdf.name,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model,
        "chunks": chunks,
    }
    INDEX_FILE.write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    progress(f"写入 {INDEX_FILE.name},共 {len(chunks)} 块,用时 {time.time() - t0:.0f}s")
    return index


# --------------------------------------------------------------------------- #
# 两张表:人工录入的中文结构化数据
# --------------------------------------------------------------------------- #
# 手册第 10、11 页那两张表是现场最常查的东西("横封不牢怎么办""伺服过载什么意思"),
# 但也恰恰是版面抽取最不可靠的地方:pypdf 把单元格按阅读顺序逐行吐出来,
# "现象 / 原因 / 解决"三列的对应关系已经丢了,硬拼只能靠猜 —— 猜错就变成
# "横封不牢 → 更换卷膜"这种害人的答案。
#
# 所以这两张表照着原文逐条录进来,页码保留,方便工人回去核对纸质手册。
# 其余散文部分仍然走自动流水线(抽 → 翻 → 入库),换手册时只有这两张表要重录。

TROUBLES = [
    ("横封不牢",
     ["加热棒温度低", "封合时间太短", "物料夹在封条之间",
      "温度过高或封合时间过长", "气路系统故障"],
     ["重设温度", "重设横封参数", "清掉夹住的物料后重设横封参数",
      "换卷膜", "停机检查气路"]),
    ("纵封封不住",
     ["加热棒温度低", "封合时间太短", "拉膜方式不正确",
      "卷膜质量差", "气路系统故障"],
     ["重设纵封温度", "重设纵封参数", "调整拉膜系统", "换卷膜", "停机检查"]),
    ("卷膜切不断",
     ["切刀参数设置不合适", "刀片有豁口或机械故障", "气路系统故障"],
     ["重设切刀参数", "停机检查或清理刀片", "停机检查气路"]),
    ("切袋位置不准",
     ["色标光电眼位置移动了", "卷膜上的色标位置不对",
      "切刀位置设置不对", "色标传感器检测不准"],
     ["重新调整光电眼", "检查卷膜色标", "重新标定切刀位置", "重新调整色标传感器"]),
    ("打印不清、打印太深把袋子打破或划破",
     ["温度设置不合适", "打印参数设置不合适", "电源短路或气路故障",
      "打印色带送带太少", "字码离橡胶垫太近"],
     ["调整后重新打印", "重设打印参数", "停机检查", "重新调整送带", "重新调整间距"]),
    ("连包时物料被夹住",
     ["横封参数不合适", "称重量太大", "填充参数不合适"],
     ["重设横封参数", "重新调整称重值", "重设填充参数"]),
]

ALARMS_TABLE = [
    ("卷膜用完", "膜已经用完,指示灯亮", "换膜,或者调整卷膜开关"),
    ("急停状态", "急停开关被按下", "松开急停开关"),
    ("伺服电机过载", "伺服电机报警", "从机械和电气两方面检查伺服电机"),
    ("门没关", "前面两扇门没关好", "关上这两扇门"),
    ("拉膜带未闭合", "拉膜带没有闭合", "按「拉膜带关」按钮把它闭合"),
    ("前横封温度异常", "温度没到设定值 / 加热棒坏了 / 电路故障",
     "让温度升到设定值 / 换加热棒 / 检修电路"),
    ("后横封温度异常", "同前横封", "同前横封"),
    ("纵封温度异常", "同前横封", "同前横封"),
    ("卷膜刹车报警", "刹车的螺旋开关被打开了", "关掉螺旋开关"),
]


# 人工整理的现场问答。这十条是照着手册通读一遍写出来的,覆盖手册散落在各页的内容
# (安装要求在第 3 页、参数参考值在第 7-9 页、保养在第 11 页),而且是**中文原生**的,
# 比机器翻译的段落准、也更像工人会问的问法。
#
# 它们和机翻段落同在一个检索池里,但因为问题和答案写在同一段,工人的问法
# ("电源什么要求""开机能马上生产吗")能直接对上,通常排在机翻段落前面。
FAQ = [
    ("机器到货后安装要注意什么?电源和气源有什么要求?", 3,
     "拆包后安全吊装、按需定位,然后调整地脚螺栓让机器保持水平。接着依次检查:"
     "所有螺栓是否松动、线缆是否松脱或损坏、机内有无异物、各动作是否正常。"
     "之后连接联动线缆、压缩空气气管和气源电源。"
     "电源要求三相四线 AC380V,辅助设备的电源也要确认接线正确。"),
    ("开机后可以马上生产吗?", 2,
     "不建议。加热棒的温度要在开机约 5 分钟后才会稳定。"
     "另外,如果机器超过 5 天没有通电,PLC 里保存的参数会丢失,"
     "开机后需要按 [APPLY],把触摸屏里保存的参数重新下发给 PLC。"),
    ("按下急停按钮后怎样恢复运行?", 2,
     "先把红色急停开关复位(旋开),再在触摸屏上复位「报警与恢复」。"
     "其他报警也是同样思路:机器报警停机后,先排除故障原因,再按 [ALARM RESET],"
     "然后重新启动。"),
    ("包装膜有的有色标有的没有,参数该怎么选?", 7,
     "在 DATA 页面设置 [ENCODER(1)/PHOTOSENSOR(0)]:有色标的膜选光电模式(0),"
     "无色标的膜选编码器模式(1)。用编码器定袋长时,实际袋长等于"
     "「第一次拉膜长度 + 第二次拉膜长度」。在无编码器状态下,"
     "还需要按 [TIME ALARM] 关闭拉膜报警。"),
    ("切刀切歪了、切到图案上怎么调?", 4,
     "光电模式下,用调试页面的 [CUTTER UP/DOWN] 微调切刀位置,每按一次 DOWN 位置值减 1。"
     "调了还是不准,就按故障表逐项排查:光电传感器位置是否移动、膜上色标位置是否正确、"
     "切刀位置设定是否有误、色标传感器是否误检。"),
    ("拉膜相关参数该怎么设定?", 7,
     "PULL SPEED 范围 0~6,数字越小速度越快;PULL DELAY 参考值 0.05~0.30 秒;"
     "P.SENSOR VALID DELAY 参考值 0.05 秒,拉膜开始后在这段时间内检测到的色标信号"
     "会被视为无效;PULL STOP TIME 设为正常拉膜周期时间的 1.5 倍,"
     "作为检测不到色标时的保护停止时间;袋长不足时用 LENGTH CORRECTION 修正。"),
    ("有中间料斗时,时间参数有参考值吗?", 8,
     "有。DISCHARGE DELAY 0~0.8 秒,建议由大往小调,秤与中间料斗距离越远数值越大;"
     "MIDDLE WAITING TIME 0.3~0.8 秒,建议由小往大调;"
     "MIDDLE TIME(料斗开门时间)0.3~0.5 秒;"
     "FILL DELAY 有中间料斗时 0.3~1 秒,没有中间料斗时 0.1~0.5 秒,与秤的卸料时间一致。"),
    ("横封封不牢可能是什么原因?", 10,
     "说明书列出的原因:加热棒温度偏低、封口时间太短、物料夹在封口棒之间、"
     "温度过高或封口时间过长、气动系统故障。处理:重设温度、调整横封参数、更换薄膜,"
     "必要时停机检查。纵封封不好还要检查拉膜路径是否正确,以及膜的质量。"),
    ("屏幕上的理论速度和实际速度有什么区别?PACK SPEED 怎么设?", 4,
     "理论速度由 PLC 根据已设定的参数自动计算,实际速度不会超过它;"
     "实际速度是运行中的真实速度,每分钟刷新一次。"
     "当 PACK SPEED 设定值大于或等于理论速度时,它不会影响实际速度。"),
    ("日常保养要做哪些工作?", 11,
     "现场安装后先做短时间试运行;定期给各关节和活动部位加润滑油并检查磨损;"
     "定期检查固定部件是否松动;定期给气动三联件加油,每班结束后排放油水分离器的积水,"
     "可用缝纫机油或润滑油;定期检查机内有无异物、电气触点是否氧化。"
     "换班或换料时可以在停机状态下按 [CLEAN MATERIAL] 清空秤斗余料。"
     "解决不了的问题联系 Synda 售后:+86-18019552399。"),
]


def curated_chunks() -> list[dict]:
    """两张表变成可检索的段。一条故障一段 —— 检索命中的就是完整的一条,
    不会出现"现象在这一段、解决在那一段"。"""
    out = [{
        "page": 10, "curated": True,
        "zh": f"【故障】{name}。可能原因:" + "；".join(f"{i+1}.{r}" for i, r in enumerate(reasons))
              + "。处理办法:" + "；".join(f"{i+1}.{f}" for i, f in enumerate(fixes)) + "。",
    } for name, reasons, fixes in TROUBLES]
    out += [{
        "page": page, "curated": True,
        # 问题也写进这一段:工人的问法("电源什么要求")和问题句对得上,
        # 光有答案的话只能靠答案里恰好出现"电源"两个字
        "zh": f"【现场问答】{q}\n{a}",
    } for q, page, a in FAQ]
    out += [{
        "page": 11, "curated": True,
        "zh": f"【报警】{name}。原因:{reason}。处理:{fix}。"
              f"这类报警会让机器停机,排除之后要先按报警复位再重启。",
    } for name, reason, fix in ALARMS_TABLE]
    return out


# --------------------------------------------------------------------------- #
# 检索:BM25 over 字二元组
# --------------------------------------------------------------------------- #

# 检索时要扔掉的字:标点和"的了吗呢"这类没有区分度的虚字
_DROP = set(" \t\n，。、；：？！（）()「」【】《》\"'…—-·/|的了是在有和与吗呢啊吧我你他它")

K1, B = 1.5, 0.75
# 命中分低于这个值就认为"手册里没有",别硬答。0.8 是拿现场问法量出来的:
# 真能在手册里找到答案的问题分数都在 1.5 以上,而"中午吃什么"这类在 0.3 以下。
MIN_SCORE = 0.8


def tokens(text: str) -> list[str]:
    """中文切字二元组,英文和数字按词切。

    为什么不用分词器:装一个 jieba 是为了把"横封不牢"切成"横封/不牢",
    而字二元组切出来是"横封/封不/不牢" —— 检索要的是和文档对得上的特征,
    多一个"封不"不影响命中,少一个依赖却省事。
    """
    out: list[str] = []
    han: list[str] = []
    word: list[str] = []

    def flush_han() -> None:
        if len(han) == 1:
            out.append(han[0])          # 单字也留着,"膜""刀"本身就是检索词
        for i in range(len(han) - 1):
            out.append(han[i] + han[i + 1])
        han.clear()

    def flush_word() -> None:
        if word:
            out.append("".join(word))
            word.clear()

    for ch in text.lower():
        if ch in _DROP:                  # 标点和虚字:在这儿断开,本身不入词表
            flush_han()
            flush_word()
        elif "\u4e00" <= ch <= "\u9fff":
            flush_word()
            han.append(ch)
        elif ch.isalnum():               # 英文单词、参数值(380v、0.05)
            flush_han()
            word.append(ch)
        else:
            flush_han()
            flush_word()
    flush_han()
    flush_word()
    return out


class Manual:
    """入库好的手册 + 检索。没入库过就抛 FileNotFoundError。"""

    def __init__(self, path: Path | str = INDEX_FILE) -> None:
        p = Path(path)
        # 没入库也能用:两张表是写在代码里的,没有它们才叫"没有手册"。
        # 现场最常问的就是这两张表,所以这一路不依赖 LLM 翻译那一步
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            self.source = data.get("source", "手册")
            self.built_at = data.get("built_at", "")
            built = data["chunks"]
        else:
            self.source = "VFFS 手册(只有故障表,散文部分还没入库)"
            self.built_at = ""
            built = []
        self.chunks = curated_chunks() + built
        # 建库时就把倒排和长度算好,加载只花几毫秒
        self._docs = [tokens(c.get("zh") or c["en"]) for c in self.chunks]
        self._len = [len(d) or 1 for d in self._docs]
        self._avg = sum(self._len) / max(len(self._len), 1)
        self._df: dict[str, int] = {}
        for d in self._docs:
            for t in set(d):
                self._df[t] = self._df.get(t, 0) + 1
        self._tf = [{t: d.count(t) for t in set(d)} for d in self._docs]

    def describe(self) -> str:
        return (f"手册《{self.source}》{len(self.chunks)} 段"
                f"{'(' + self.built_at + ' 入库)' if self.built_at else ''}")

    def search(self, query: str, k: int = 3) -> list[tuple[float, dict]]:
        """BM25 取前 k 段,按分数从高到低。"""
        q = tokens(query)
        if not q or not self.chunks:
            return []
        n = len(self.chunks)
        scored: list[tuple[float, dict]] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            for t in q:
                f = tf.get(t)
                if not f:
                    continue
                idf = math.log(1 + (n - self._df[t] + 0.5) / (self._df[t] + 0.5))
                s += idf * f * (K1 + 1) / (f + K1 * (1 - B + B * self._len[i] / self._avg))
            if s > 0:
                scored.append((s, self.chunks[i]))
        scored.sort(key=lambda x: -x[0])
        return scored[:k]

    def context(self, query: str, k: int = 2, budget: int = 800) -> tuple[str, list[int]]:
        """检索结果拼成给 LLM 的上下文,外加命中的页码。

        budget 是字符上限。不是按上下文窗口算的,是按**预填时间**算的:
        手册内容每多 400 字符,首字就慢两三秒(这台机器上 1200 字符要等 9-13 秒)。
        两段 800 字符够覆盖一条故障或一组参数,再多是拿延迟换用不上的内容。
        """
        hits = [(s, c) for s, c in self.search(query, k) if s >= MIN_SCORE]
        if not hits:
            return "", []
        # 第一名明显领先时只给它一段。
        #
        # 实测教训:问"膜没有色标参数怎么选",第一名(10.3 分)正是那条对的问答,
        # 第二名(5.6 分)是拉膜参数表 —— 两段一起给,4B 模型把第二段的数字拿来凑,
        # 答成"选 PHOTOSENSOR,PULL SPEED 0.1",既搞反了对应关系又编了数值。
        # 材料越少越不容易串台,而领先这么多本来就说明只有一段相关。
        if len(hits) > 1 and hits[0][0] >= 1.8 * hits[1][0]:
            hits = hits[:1]
        parts, pages, used = [], [], 0
        for s, c in hits:
            body = (c.get("zh") or c["en"]).strip()
            if used + len(body) > budget and parts:
                break
            parts.append(f"【手册第 {c['page']} 页】\n{body}")
            pages.append(c["page"])
            used += len(body)
        return "\n\n".join(parts), pages


def system_prompt(context: str) -> str:
    """把检索到的手册内容包成系统提示。

    三条约束都是为了别让 4B 模型自由发挥:只许用手册内容、必须报页码、
    找不到就直说。设备手册答错比答不出来危险得多 —— 工人真会照着做。
    """
    return (
        "下面是设备手册里查到的内容。回答只能依据它,不要用常识补充、不要猜。\n"
        # 这条要压过系统提示里那句"不超过 15 个字"。故障有好几条原因,
        # 砍到 15 字只剩"重设温度"——工人照着做等于白做。40 字能说清
        # 最可能的两条原因加对应处理,念出来约 8 秒,还在可听范围内
        "这类问题允许说到 40 个字:先说最可能的一两条原因,再说对应怎么处理。\n"
        # 不报页码:念出来是"手册第十页写着"五个字的开销,而工人手上正忙着,
        # 要的是"先查什么、再调什么"。页码仍然打在终端里,想翻手册的人看得到
        "直接说怎么做,不要提页码,也不要说「手册上写着」。\n"
        "资料有好几段时,只用最相关的那一段回答;"
        "不同段落的数字不要混在一起用,资料里没有的数值一个都不要写。\n"
        "手册里没写的就说「手册里没写」,不要编;手册只说了原则没给具体数值的,"
        "就把原则说出来,别当成没写。\n\n"
        + context
    )


def main() -> int:
    import argparse

    import winutil

    winutil.setup_console()

    p = argparse.ArgumentParser(description="设备手册入库与检索")
    p.add_argument("query", nargs="*", help="要查的问题")
    p.add_argument("--build", action="store_true", help="重新入库(要先起 LLM server)")
    p.add_argument("--pdf", default=None, help="手册 PDF,默认找目录里第一个")
    p.add_argument("--chat-url", default="http://127.0.0.1:8080/v1", help="LLM server 地址")
    p.add_argument("--list", action="store_true", help="列出库里每一段")
    p.add_argument("-k", type=int, default=3, help="检索取前几段")
    args = p.parse_args()

    if args.build:
        pdf = Path(args.pdf) if args.pdf else next(iter(sorted(HERE.glob("*.pdf"))), None)
        if pdf is None:
            print("目录里没有 PDF,用 --pdf 指一个")
            return 1
        build(pdf, args.chat_url)
        return 0

    m = Manual()
    print(m.describe())
    if args.list:
        for i, c in enumerate(m.chunks, 1):
            head = (c.get("zh") or c["en"]).replace("\n", " ")[:60]
            print(f"  {i:>3}. 第 {c['page']:>2} 页  {head}")
        return 0

    queries = args.query or [
        "横封不牢怎么办", "切不断怎么回事", "伺服过载是什么意思",
        "急停按了怎么恢复", "拉膜延时设多少", "多久加一次润滑油", "中午吃什么",
    ]
    for q in queries:
        hits = m.search(q, args.k)
        print(f"\n「{q}」")
        if not hits or hits[0][0] < MIN_SCORE:
            best = hits[0][0] if hits else 0.0
            print(f"  手册里没有(最高分 {best:.2f} < {MIN_SCORE})→ 交给 LLM")
            continue
        for s, c in hits:
            if s < MIN_SCORE:
                continue
            body = (c.get("zh") or c["en"]).replace("\n", " ")
            print(f"  {s:5.2f} 第 {c['page']:>2} 页  {body[:90]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
