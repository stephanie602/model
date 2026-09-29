#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
包装机的状态和报警 —— 给语音助手一台"有东西可说"的设备。

演示里的两类交互,方向是相反的,所以这里也分成两半:

  机器主动说(报警)   缺原材料 / 传送带被异物卡住
                     触发源是设备,不经过麦克风,也不该经过 LLM ——
                     报警词是安全相关的固定话术,必须每次一字不差,
                     而且要立刻出声。所以走"固定文案 → TTS"这条直路,
                     大模型只在事后被告知发生过什么(见 note_alert)。

  用户主动问(状态)   现在温度和速度多少 / 今天包了多少个
                     这类问题的答案是**数字**,让 4B 模型去编只会编错。
                     所以先拿拼音把问题对到一张查询表上,命中就直接
                     用真实状态拼一句话回答,对不上才落到 LLM 手里。

                     只接**问设备**的那些。笼统的"怎么样""正常吗"要同时
                     点到机器才算("机器怎么样"接管,"今天怎么样"不接管)——
                     否则什么问题都被答成"封口温度一百六十八度"。

  工人回话(解除)     料加好了 / 杂物清理干净了
                     报警响的时候工人手上正忙着,回来按键不现实,
                     说一句就该把报警撤掉。见 voice_clear()。

为什么按拼音对而不是按汉字:和 intent.py 同一个理由 —— 噪声下 SenseVoice 输出的
是同音错字("温度"→"文度"、"包装"→"包妆"),汉字比对全军覆没,拼音全中。

状态本身是模拟的:温度在设定值附近慢慢漂,产量按节拍往上走。真机接上以后,
把 Machine.snapshot() 换成读 PLC / MES 就行,上面那两条链路一行都不用改。

单独试:
    .venv\\Scripts\\python.exe machine.py                 # 打一遍状态和报警文案
    .venv\\Scripts\\python.exe machine.py "现在温度多少"   # 试问答匹配
"""

from __future__ import annotations

import math
import sys
import threading
import time

# --------------------------------------------------------------------------- #
# 报警
# --------------------------------------------------------------------------- #


class Alert:
    """一条报警。speech 是念出来的话,detail 是只打在终端里的补充。

    speech 至少写成"现象 + 该干什么"两段:光说"缺料"操作工还得自己想下一步,
    而报警的意义就是让人不用想。剩余时间、余量这类**估算值**不进 speech ——
    念出来就是承诺,和现场对不上比不说更糟;要看放 detail 里,终端上打给会看数的人。
    长度控制在 20 字上下 —— Kokoro 合成大约 2 秒,再长了人就等不及、会走开。

    cleared 是故障排除后的解除播报:报警响过必须有个收尾,不然操作工不知道
    该不该继续等 —— 现场最怕的是"报了警,然后没下文"。解除词要说清**是什么解除了**,
    不能只说"已恢复正常":同时挂着两条报警时,光说恢复听不出好的是哪一条。

    stops 区分两类信号,这是手册第 10、11 页本身的分法:
    **报警**(第 11 页)会让机器停机,必须排除后复位再启动;**故障**(第 10 页)
    是封口不牢、切不断这类质量问题,机器照跑,但包出来的东西不合格。
    两者的播报语气也不同 —— 报警说"注意",故障说"提醒"。

    clear_words 是工人说什么算这条解除了(拼音匹配,见 voice_clear)。
    留空就只认通用说法("好了""处理完了"),而且只在单独挂着这一条时才认。

    fix 是**一句话**的处理办法(六到八个字),给"现在有什么异常"那种一次报好几条的
    场合用 —— 那时候每条只能给一句,把 speech 整句念完,三条就要念一分钟。
    speech 仍然是单独报这一条时的完整话术。
    """

    __slots__ = ("key", "title", "speech", "detail", "cleared", "stops",
                 "clear_words", "page", "fix")

    def __init__(self, key: str, title: str, speech: str, detail: str = "",
                 cleared: str = "", stops: bool = True,
                 clear_words: "tuple[str, ...]" = (), page: int = 0,
                 fix: str = "") -> None:
        self.fix = fix or "按手册处理"
        self.key = key
        self.title = title
        self.speech = speech
        self.detail = detail
        self.cleared = cleared or "故障已排除，设备恢复正常运行。"
        self.stops = stops
        self.clear_words = clear_words
        self.page = page          # 手册页码,终端里打给想翻手册的人

    def __repr__(self) -> str:
        return f"Alert({self.key!r}, {self.title!r})"


ALERTS: list[Alert] = [
    Alert(
        "material",
        "缺原材料",
        "注意，包装机原材料不足，请及时补充原材料。",
        "料位传感器低于下限 · 建议在 5 分钟内上料，否则整线停机",
        cleared="原材料已补充，料位恢复正常，设备继续运行。",
        clear_words=("料加好", "加料", "上料", "补料", "补充好", "料补充完",
                     "加满", "料满", "上完", "加完"), fix="补充原材料",
    ),
    Alert(
        "jam",
        "传送带异物堵塞",
        "注意，传送带有异物卡住，设备已自动停机，请清理杂物后重新启动。",
        "输送电机过载保护动作 · 清理后需手动复位再启动",
        cleared="传送带杂物已清理，设备已重新启动，恢复正常运行。",
        clear_words=("清理", "清干净", "清掉", "清除", "清完", "杂物", "异物",
                     "拿出来", "取出来", "掏出来", "不卡了", "通了"), fix="清理传送带杂物",
    ),

    # ---- 手册第 11 页那张报警表:9 条,都会让机器停机 ----
    # 播报词是照着手册的"原因 / 处理"两列写的 —— 报警的意义是让人不用想下一步,
    # 所以每条都带上该干什么
    Alert("film_out", "卷膜用完",
          "注意，卷膜已经用完，请更换卷膜或者调整卷膜开关。",
          "手册第 11 页 · 膜已用完，指示灯亮", page=11,
          cleared="卷膜已更换，设备恢复运行。",
          clear_words=("换膜", "膜换好", "换好膜", "新膜", "换卷膜"), fix="更换卷膜"),
    Alert("estop", "急停被按下",
          "注意，急停开关被按下，设备已停机。处理完之后松开急停开关再启动。",
          "手册第 11 页 · 松开急停开关后需在触摸屏复位", page=11,
          cleared="急停已复位，设备恢复运行。",
          clear_words=("急停复位", "松开急停", "急停恢复", "急停松了"), fix="松开急停开关"),
    Alert("servo", "伺服电机过载",
          "注意，伺服电机过载报警，请从机械和电气两方面检查伺服电机。",
          "手册第 11 页 · 伺服电机报警", page=11,
          cleared="伺服电机已检查，报警复位，设备恢复运行。",
          clear_words=("伺服检查", "电机检查", "过载复位", "伺服好了"), fix="检查伺服电机"),
    Alert("door", "前门没关",
          "注意，前面两扇门没关好，设备已停机，请关门后重新启动。",
          "手册第 11 页 · 前两扇门未闭合", page=11,
          cleared="门已关好，设备恢复运行。",
          clear_words=("门关好", "关门", "门已经关", "门关上"), fix="关好前面两扇门"),
    Alert("belt_open", "拉膜带未闭合",
          "注意，拉膜带没有闭合，请按拉膜带关按钮把它闭合。",
          "手册第 11 页 · 按「拉膜带关」闭合", page=11,
          cleared="拉膜带已闭合，设备恢复运行。",
          clear_words=("拉膜带闭合", "带子合上", "拉膜带关", "带闭合"), fix="闭合拉膜带"),
    Alert("temp_front", "前横封温度异常",
          "注意，前横封温度异常，请检查温度有没有到设定值、加热棒和电路。",
          "手册第 11 页 · 温度未到设定值 / 加热棒坏 / 电路故障", page=11,
          cleared="前横封温度恢复正常，设备继续运行。",
          clear_words=("前横封好", "前横封温度正常", "前面温度上来"), fix="检查前横封加热棒"),
    Alert("temp_back", "后横封温度异常",
          "注意，后横封温度异常，请检查温度有没有到设定值、加热棒和电路。",
          "手册第 11 页 · 同前横封", page=11,
          cleared="后横封温度恢复正常，设备继续运行。",
          clear_words=("后横封好", "后横封温度正常", "后面温度上来"), fix="检查后横封加热棒"),
    Alert("temp_vert", "纵封温度异常",
          "注意，纵封温度异常，请检查温度有没有到设定值、加热棒和电路。",
          "手册第 11 页 · 同前横封", page=11,
          cleared="纵封温度恢复正常，设备继续运行。",
          clear_words=("纵封好", "纵封温度正常", "纵封温度上来"), fix="检查纵封加热棒"),
    Alert("roll_brake", "卷膜刹车报警",
          "注意，卷膜刹车报警，刹车的螺旋开关被打开了，请把它关掉。",
          "手册第 11 页 · 关掉螺旋开关", page=11,
          cleared="刹车开关已关闭，设备恢复运行。",
          clear_words=("刹车关", "螺旋开关关", "刹车好了"), fix="关掉刹车螺旋开关"),

    # ---- 手册第 10 页那张故障表:6 条,机器照跑但包不合格 ----
    # stops=False:这类不停机。报出来是为了让人赶紧调,不是为了叫停线
    Alert("seal_h", "横封不牢",
          "提醒，横封不牢。先查加热棒温度和封合时间，再看物料是不是夹在封条之间。",
          "手册第 10 页 · 温度低 / 时间短 / 物料夹住 / 气路故障", page=10, stops=False,
          cleared="横封已调好，封口恢复正常。",
          clear_words=("横封调好", "横封好了", "封好了", "横封没问题"), fix="查横封温度和时间"),
    Alert("seal_v", "纵封封不住",
          "提醒，纵封封不住。查加热棒温度、封合时间和拉膜方式，膜的质量也看一下。",
          "手册第 10 页 · 温度低 / 时间短 / 拉膜不正 / 膜质量差", page=10, stops=False,
          cleared="纵封已调好，封口恢复正常。",
          clear_words=("纵封调好", "纵封没问题"), fix="查纵封温度和拉膜"),
    Alert("cut_fail", "卷膜切不断",
          "提醒，卷膜切不断。检查切刀参数和刀片，必要时停机清理刀片。",
          "手册第 10 页 · 切刀参数 / 刀片豁口 / 气路故障", page=10, stops=False,
          cleared="切刀已调好，切断恢复正常。",
          clear_words=("切刀调好", "刀片清理", "刀好了", "刀换了"), fix="查切刀参数和刀片"),
    Alert("cut_pos", "切袋位置不准",
          "提醒，切袋位置不准。检查色标光电眼位置和卷膜色标，重新标定切刀位置。",
          "手册第 10 页 · 光电眼移位 / 色标不对 / 切刀位置", page=10, stops=False,
          cleared="切袋位置已校准，恢复正常。",
          clear_words=("位置调好", "光电眼调好", "标定好", "对准了"), fix="校准光电眼和切刀"),
    Alert("print_bad", "打印不清或打破袋子",
          "提醒，打印有问题。调整打印温度和参数，再看看色带送带够不够。",
          "手册第 10 页 · 温度 / 打印参数 / 色带送带 / 字码离垫太近", page=10, stops=False,
          cleared="打印已调好，恢复正常。",
          clear_words=("打印调好", "色带调好", "打印好了"), fix="调打印温度和色带"),
    Alert("clapped", "连包时物料被夹",
          "提醒，连包时物料被夹住。调整横封参数、称重量和填充参数。",
          "手册第 10 页 · 横封参数 / 称重量太大 / 填充参数", page=10, stops=False,
          cleared="参数已调整，连包恢复正常。",
          clear_words=("参数调好", "称重调好", "填充调好"), fix="调横封和填充参数"),
]


# --------------------------------------------------------------------------- #
# 设备状态
# --------------------------------------------------------------------------- #

# 封口温度设定值和正常波动范围(摄氏度)。真机上这两个数来自配方参数。
TEMP_SET = 168.0
TEMP_SWING = 1.5
# 额定节拍(包/分钟)
SPEED_SET = 42.0
# 今天开工到现在已经包了多少 —— 演示从一个像样的基数起步,
# 再按节拍往上加,免得刚启动就回答"今天完成 0 个"
PACKAGES_BASE = 3860


class Machine:
    """一台模拟包装机。线程安全 —— 报警是热键线程触发的,状态是识别线程在读。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0 = time.time()
        self.running = True
        # 当前挂着的信号,存 Alert.key。原来是 material_low / jammed 两个布尔,
        # 接进手册那 15 条之后再加布尔就失控了 —— 改成集合,加一条信号
        # 只要往 ALERTS 里加一行,状态这边不用动
        self.active: set[str] = set()
        self.last_alert: Alert | None = None
        self._alert_i = 0
        # 停机期间产量不该再涨,所以把"停机那一刻的产量"冻结下来
        self._frozen: int | None = None

    # ---------------- 状态 ---------------- #

    def snapshot(self) -> dict:
        """当前状态。真机接上以后,把这个方法换成读 PLC 就行。"""
        with self._lock:
            elapsed = time.time() - self._t0
            # 温度按一条 40 秒周期的正弦漂,像 PID 在设定值附近来回修正;
            # 停机后按指数往室温掉,掉得比升温慢 —— 和真机的热惯量一个量级
            if self.running:
                temp = TEMP_SET + TEMP_SWING * math.sin(elapsed / 40.0 * 2 * math.pi)
            else:
                temp = TEMP_SET - 12.0 * (1 - math.exp(-elapsed / 600.0))
            speed = SPEED_SET if self.running else 0.0
            if self._frozen is not None:
                packages = self._frozen
            else:
                packages = PACKAGES_BASE + int(elapsed * SPEED_SET / 60.0)
            return {
                "running": self.running,
                "temperature": round(temp, 1),
                "speed": round(speed, 1),
                "packages": packages,
                # 这两个键留着不是为了好看:状态答句和 LLM 上下文都在用它们,
                # 改成从集合派生,调用方一行都不用改
                "material_low": "material" in self.active,
                "jammed": "jam" in self.active,
                "active": sorted(self.active),
            }

    def status_text(self) -> str:
        """一句人话的状态,给 LLM 当上下文用(不是念给用户听的)。"""
        s = self.snapshot()
        parts = [
            f"设备{'运行中' if s['running'] else '已停机'}",
            f"封口温度 {s['temperature']:.1f} 摄氏度",
            f"包装速度 每分钟 {s['speed']:.0f} 包",
            f"今日完成 {s['packages']} 包",
        ]
        by_key = {a.key: a for a in ALERTS}
        for key in s["active"]:
            a = by_key.get(key)
            if a is not None:
                parts.append(("报警:" if a.stops else "故障:") + a.title)
        return "；".join(parts) + "。"

    # ---------------- 报警 ---------------- #

    def next_alert(self) -> Alert:
        """取下一条报警并让它作用到状态上,循环取。"""
        with self._lock:
            index = self._alert_i % len(ALERTS)
            self._alert_i += 1
        return self.raise_alert(index)

    def raise_alert(self, index: int) -> Alert:
        """直接触发第 index 条信号(从 0 起)。演示时要单独放某一条,不用一路按过去。

        信号必须改状态,不然演示会自相矛盾:刚播完"传送带卡住已停机",
        用户接着问速度,还答"每分钟 42 包"。
        """
        with self._lock:
            alert = ALERTS[index % len(ALERTS)]
            # 直选之后,空格接着按要从这一条的下一条继续,别跳回开头
            self._alert_i = (index % len(ALERTS)) + 1
            self.active.add(alert.key)
            if alert.stops and self.running:
                # 停机瞬间把产量冻住,之后再问"今天多少个"数字不会继续涨
                elapsed = time.time() - self._t0
                self._frozen = PACKAGES_BASE + int(elapsed * SPEED_SET / 60.0)
            if alert.stops:
                self.running = False
            self.last_alert = alert
            return alert

    def is_active(self, index: int) -> bool:
        """第 index 条信号现在还挂着吗。"""
        with self._lock:
            return ALERTS[index % len(ALERTS)].key in self.active

    def active_alert(self) -> Alert | None:
        """当前挂着的信号(多条时给最近那条),没有就是 None。"""
        with self._lock:
            return self.last_alert if self.active else None

    def active_alerts(self) -> list[Alert]:
        """当前挂着的全部信号,按 ALERTS 里的顺序。"""
        with self._lock:
            keys = set(self.active)
        return [a for a in ALERTS if a.key in keys]

    def clear(self, index: int) -> Alert | None:
        """解除第 index 条信号,只解这一条。本来就没响则返回 None。

        按条解除而不是一把全清:好几条可以同时挂着(缺料还没补,传送带又卡了),
        这时候"全清"会连没处理的那条一起抹掉,状态就和现场对不上了。
        """
        alert = ALERTS[index % len(ALERTS)]
        with self._lock:
            if alert.key not in self.active:
                return None
            self.active.discard(alert.key)
            self._resume_if_clear()
            if not self.active:
                self.last_alert = None
            return alert

    def _resume_if_clear(self) -> None:
        """没有会停机的信号挂着了就恢复运行。调用方必须已经持锁。"""
        if any(a.key in self.active and a.stops for a in ALERTS):
            return
        self.running = True
        if self._frozen is not None:
            # 停机期间不产出:把时间原点往后挪,让产量从冻结值接着涨
            self._t0 = time.time() - (self._frozen - PACKAGES_BASE) * 60.0 / SPEED_SET
            self._frozen = None

    def clear_all(self) -> "list[Alert]":
        """全部解除,返回清掉的那几条(按 ALERTS 顺序)。R 键和"都好了"都走这里。"""
        gone = self.active_alerts()
        self.reset()
        return gone

    def recover(self) -> Alert | None:
        """全部解除,回到正常运行,返回最近那条报警(本来就正常则返回 None)。

        和 reset() 的区别只在**有没有话要说**:reset 是把状态推回原点,
        recover 还要告诉调用方"解除的是哪一条",好念出对应的解除词。
        """
        alert = self.active_alert()
        self.reset()
        return alert

    def reset(self) -> None:
        """故障处理完,恢复运行。演示要连着跑好几遍,得有路回到正常状态。"""
        with self._lock:
            self.running = True
            self.active.clear()
            self.last_alert = None
            if self._frozen is not None:
                # 停机期间不产出:把时间原点往后挪,让产量从冻结值接着涨
                self._t0 = time.time() - (self._frozen - PACKAGES_BASE) * 60.0 / SPEED_SET
                self._frozen = None


# --------------------------------------------------------------------------- #
# 状态问答:拼音查询表
# --------------------------------------------------------------------------- #

def _key(text: str) -> tuple[str, str]:
    """把一句话变成可比对的串,连同用的是哪种表示。

    有 pypinyin 就转无声调拼音(同音错字照样命中);没有就退回汉字 ——
    命中率低一截,但不该因为少一个纯 Python 的小依赖就整个问答不可用。
    """
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:
        return text, "hanzi"
    return "".join(lazy_pinyin(text, style=Style.NORMAL)), "pinyin"


# 每条查询三组词:
#   any   任一命中就算问到了这一项
#   with  还必须同时出现的词(只有产量需要 —— "包装"两个字本身太常见)
#   not   出现就一票否决。用来和控制命令划清界限:"调高温度"是让机器改参数,
#         不是在问温度,光看"温度"会把它也答成状态播报
_QUERIES = [
    {
        "key": "temperature",
        "pinyin": {"any": ("wendu",),
                   "not": ("tiaogao", "tiaodi", "diaogao", "diaodi", "shezhi", "gaidao")},
        "hanzi": {"any": ("温度",), "not": ("调高", "调低", "设置", "改到")},
    },
    {
        "key": "speed",
        "pinyin": {"any": ("sudu", "jiepai", "duokuai", "kuaiman"),
                   "not": ("tiaokuai", "tiaoman", "jiakuai", "jianman", "shezhi")},
        "hanzi": {"any": ("速度", "节拍", "多快"), "not": ("调快", "调慢", "加快", "减慢", "设置")},
    },
    {
        "key": "packages",
        # "今天包了多少" / "完成多少个包装" / "产量多少" —— 三种问法一网打尽
        # "完成多少"也算 —— 在包装线上问"完成了多少",问的不会是别的
        "pinyin": {"any": ("chanliang", "baozhuang", "baoshu", "baole", "wancheng"),
                   "with": ("duoshao", "jige", "wancheng", "chanliang", "jintian", "zongshu"),
                   "not": ()},
        "hanzi": {"any": ("产量", "包装", "包了", "完成"),
                  "with": ("多少", "几个", "完成", "产量", "今天", "总数"),
                  "not": ()},
    },
]


# "正常吗""怎么样""什么情况" —— 问的是好坏,不是某个具体的数
_JUDGE = {
    "pinyin": ("zhengchang", "zenmeyang", "haoma", "meiwenti", "qingkuang",
               "zhuangtai", "yunxingdemakeyi", "keyima"),
    "hanzi": ("正常", "怎么样", "好吗", "没问题", "情况", "状态"),
}
# 说的是**这台机器**吗。笼统的问法必须点到主语才接管:
# "机器怎么样"是问设备,"今天怎么样""你怎么样"不是 —— 早先没这一层,
# 结果什么问题都被答成"封口温度一百六十八度、速度四十二包",很蠢。
_SUBJECT = {
    "pinyin": ("jiqi", "shebei", "baozhuangji", "shengchanxian", "xianti",
               "zhetaiji", "zheitaiji", "jitai", "shengchan", "zhuangtai"),
    "hanzi": ("机器", "设备", "包装机", "生产线", "线体", "机台", "生产", "状态"),
}


# 直接问异常:"有什么异常""什么故障""哪里有问题"。
# 和 _JUDGE 那一类分开 —— 那一类问的是"好不好",这一类问的是"具体是哪几条",
# 答法完全不同:前者报三个数,后者要把挂着的逐条念出来。
_FAULT_ASK = {
    "pinyin": ("yichang", "guzhang", "baojing", "maobing", "youwenti",
               "naliyou", "huaile", "bumaobing"),
    "hanzi": ("异常", "故障", "报警", "毛病", "有问题", "哪里有", "坏了"),
}


def _hits(text: str) -> list[str]:
    """这句话在问哪几项状态。什么都没问到就是空列表。"""
    probe, style = _key(text)
    out = []
    for q in _QUERIES:
        rule = q[style]
        if any(bad in probe for bad in rule.get("not", ())):
            continue
        if not any(k in probe for k in rule["any"]):
            continue
        need = rule.get("with")
        if need and not any(k in probe for k in need):
            continue
        out.append(q["key"])

    if any(k in probe for k in _FAULT_ASK[style]):
        out.append("faults")

    # 笼统问法:只有同时点到"机器"和"好坏"才算问设备,单说"怎么样"不接管
    judge = any(k in probe for k in _JUDGE[style])
    if judge and any(k in probe for k in _SUBJECT[style]):
        out.append("overview")
    elif judge and out:
        out.append("judge")  # "温度正常吗" —— 答具体项,前面加一句结论
    return out


# 答句拆成"固定帧 + 数字"。固定帧写死,开机就合成好存着;数字单独一小段,现合。
#
# 为什么拆到这个粒度 —— 量出来的 Kokoro 合成时间对长度有断崖:
#
#     5 字   0.46s        13 字  5.50s
#     7 字   2.27s        21 字  7.58s
#
# 整句一次合成要等 9-10 秒才响第一声。拆开之后:固定帧命中缓存 0 秒出声,
# 中间只剩一个数字要现合,而数字都在 5 字以内(“一百六十八”“四十二”),
# 半秒就好 —— 而且这半秒被前面那段正在播的固定帧盖住了,听不出来。
FRAMES = {
    "temp_head": "当前封口温度",
    "temp_tail": "摄氏度，",
    "speed_head": "包装速度 每分钟",
    "speed_tail": "包，",
    "stopped": "设备当前已停机，速度为零，",
    "pkg_head": "今天已经完成",
    "pkg_tail": "个包装，",
    "material_low": "另外原材料不足，请及时补充。",
    # "现在怎么样""一切正常吗"这类笼统的问法,答的是一份体检报告:先给结论,
    # 再报三个数。结论也是写死的两句之一,照样预合成
    "ok": "一切正常，",
    "not_ok": "设备有异常，",
    # 问异常时逐条报出来。现场同时挂三四条是常态(缺料 + 温度异常 + 横封不牢),
    # 只答一句"设备有异常"等于没说 —— 工人还得自己去翻触摸屏
    "fault_head": "当前有",
    "fault_tail": "条异常，分别是",
    "fault_none": "目前没有报警，设备一切正常。",
    "fault_fix_head": "请",
    "fault_stopped": "设备已停机，处理完说一声就能恢复。",
    "fault_running": "设备还在运行，但包装质量会受影响。",
}

# 开机预合成的清单:固定帧,加上取值有限的那几个数 ——
# 温度就在设定值上下几度,速度不是额定就是零。产量每包都在变,缓存不了,
# 但换成 Matcha 之后念一个四位数只要零点几秒,现合完全跟得上
# (Kokoro 时代要两三秒,当时是把产量拆成"三千八百"+"六十"两段去凑缓存,
#  代价是只能报个约数"三千八百六十多个" —— 现在不用将就了,报准确数)。
PRECACHE_TEXTS = (
    list(FRAMES.values())
    + [str(v) for v in range(int(TEMP_SET) - 4, int(TEMP_SET) + 5)]
    + [str(int(SPEED_SET)), "0"]
    + [str(v) for v in range(1, 10)]          # "当前有 3 条异常"里的条数
    + [a.title + "。" for a in ALERTS]        # 单独挂一条时就是这一句
)


def answer_parts(text: str, machine: Machine) -> list[str] | None:
    """状态问答的答案,切成"固定帧 / 数字"交替的几段,按顺序念。

    对不上查询表返回 None(交回给 LLM)。
    """
    if not text:
        return None
    hits = _hits(text)
    if not hits:
        return None

    s = machine.snapshot()
    parts: list[str] = []

    if "faults" in hits:
        # 逐条报全。挂着三四条是常态,只说"设备有异常"等于没说
        act = machine.active_alerts()
        if not act:
            parts.append(FRAMES["fault_none"])
        else:
            parts += [FRAMES["fault_head"], str(len(act)), FRAMES["fault_tail"],
                      "、".join(a.title for a in act) + "。",
                      # 光报名字工人还得自己想下一步,把每条的一句话办法接上。
                      # 用 fix 而不是 speech:三条 speech 念完要一分钟
                      FRAMES["fault_fix_head"], "、".join(a.fix for a in act) + "。",
                      FRAMES["fault_stopped"] if not s["running"]
                      else FRAMES["fault_running"]]
        # 问异常时不顺带报三个数,除非用户同时点名了某一项
        hits = [h for h in hits if h in ("temperature", "speed", "packages")]
        if not hits:
            return parts

    if "overview" in hits or "judge" in hits:
        # 问好坏就先给一句结论(按状态判,不猜)。问题里点名了具体哪一项,
        # 就只答那一项 —— "这个温度正常吗"别把产量也倒出来
        act = machine.active_alerts()
        if not act:
            parts.append(FRAMES["ok"])
        elif "overview" in hits:
            # "机器怎么样"是问整体,有异常就说清是哪几条,再报三个数
            parts += [FRAMES["fault_head"], str(len(act)), FRAMES["fault_tail"],
                      "、".join(a.title for a in act) + "。"]
        else:
            # "温度正常吗"问的是某一项,只给一句结论就够 ——
            # 把四条异常全念一遍再答温度,等于答非所问
            parts.append(FRAMES["not_ok"])
        specific = [k for k in ("temperature", "speed", "packages") if k in hits]
        hits = specific or ["temperature", "speed", "packages"]
    if "temperature" in hits:
        # 念整数:小数点后那一位既没人关心,又要多合成一个"点几"(多花一秒)
        parts += [FRAMES["temp_head"], f"{s['temperature']:.0f}", FRAMES["temp_tail"]]
    if "speed" in hits:
        # 停机时报"每分钟 0 包"听着像故障没被察觉,直接把原因说出来
        if s["running"]:
            parts += [FRAMES["speed_head"], f"{s['speed']:.0f}", FRAMES["speed_tail"]]
        else:
            parts.append(FRAMES["stopped"])
    if "packages" in hits:
        parts += [FRAMES["pkg_head"], str(s["packages"]), FRAMES["pkg_tail"]]
    if s["material_low"] and "packages" in hits:
        parts.append(FRAMES["material_low"])
    # 最后一段收尾:逗号换成句号,不然念出来像话没说完
    if parts and parts[-1].endswith("，"):
        parts[-1] = parts[-1][:-1] + "。"
    return parts


def answer_query(text: str, machine: Machine) -> str | None:
    """同一个答案,拼成一整句 —— 打印和记进对话历史用这个。"""
    parts = answer_parts(text, machine)
    if not parts:
        return None
    # 拼回去要做两件事:把拆开念的数字合回一个数(念的是"三千八百"+"六十",
    # 写出来该是 3860),数字两边补空格,不然是"当前封口温度168摄氏度"挤在一起
    out: list[str] = []
    num = 0
    for p in parts:
        if p.isdigit():
            num += int(p)
            continue
        if num:
            out.append(f" {num} ")
            num = 0
        out.append(p)
    if num:
        out.append(f" {num} ")
    return "".join(out).replace("  ", " ").strip()


# --------------------------------------------------------------------------- #
# 语音解除报警
# --------------------------------------------------------------------------- #
#
# 报警响了以后,工人手上多半正忙着(在上料、在掏传送带里的杂物),不可能回来按键。
# 说一句"料加好了"就该把报警撤掉 —— 这是这个场景里语音真正顶用的地方。
#
# 分两类词:
#   专用   只解对应那一条("补料"只解缺料,不会顺手把传送带的报警也撤了)
#   通用   "好了""处理完了" —— 解当前挂着的那条;两条都挂着时不动,
#          让工人说清楚是哪一条,免得把没处理的那条也抹掉
_CLEAR_WORDS = {
    "material": {
        "pinyin": ("buliao", "jialiao", "shangliao", "huanliao", "liaojiahao",
                   "liaobuhao", "juanmo", "buchong", "yuancailiao", "jiaman",
                   "liaoman", "huanmo", "shangwan", "jiawan", "buwan", "buhao"),
        "hanzi": ("补料", "加料", "上料", "换料", "料加", "补好", "补充", "补完",
                  "卷膜", "换膜", "原材料", "加满", "料满", "上完", "加完"),
    },
    "jam": {
        "pinyin": ("qingli", "qinggan", "qingdiao", "qingchu", "qingwan",
                   "yiwu", "zawu", "nachulai", "quchulai", "taochulai",
                   "chongxinqidong", "chongqi", "bukale", "tongle"),
        "hanzi": ("清理", "清干", "清掉", "清除", "清完", "异物", "杂物",
                  "拿出来", "取出来", "掏出来", "重新启动", "重启", "不卡了", "通了"),
    },
}
_CLEAR_GENERIC = {
    "pinyin": ("haole", "chulihao", "chuliwan", "jiejuele", "gaoding", "nonghaole",
               "huifu", "fuwei", "keyiliao", "wanchengle"),
    "hanzi": ("好了", "处理完", "处理好", "解决", "搞定", "弄好", "恢复", "复位",
              "可以了", "完成了"),
}


class ClearResult:
    """一次语音解除的结果:解了哪几条,或者要不要反问一句。

    ask 非空表示"工人说了句笼统的'处理好了',但现在挂着好几条" —— 这时候
    既不能一句话全清(万一他只修了一条,另外几条被默默抹掉,机器带病继续跑),
    也不该逼他一条一条念(七八条要来回七八轮)。折中:把挂着的报一遍,
    问一句"都处理好了吗",他应一声才全清。
    """

    __slots__ = ("cleared", "ask")

    def __init__(self, cleared: "list[Alert]" = (), ask: "list[Alert]" = ()) -> None:
        self.cleared = list(cleared)
        self.ask = list(ask)

    def __bool__(self) -> bool:
        return bool(self.cleared or self.ask)


# 明确断言"全部都好了"的说法:这种话工人是主动担保的,直接全清,不反问。
# 和下面的 _CLEAR_GENERIC 分开 —— "好了"含糊,"都好了"不含糊。
_CLEAR_ALL = {
    "pinyin": ("doubaohaole", "douhaole", "quanhaole", "quanbuhaole", "quanbuchulihao",
               "douchulihaole", "douchuliwanle", "quanbuchulíwan", "quanbujiechu",
               "doujiejuele", "quangaodingle", "doungaodingle"),
    "hanzi": ("都好了", "全好了", "全部好了", "都处理好", "全部处理好", "都处理完",
              "全部解除", "都解决了", "全搞定", "都搞定"),
}
# 确认反问的应答:"是的""对""都好了"
_CONFIRM = {
    "pinyin": ("shide", "dui", "duide", "meicuo", "haole", "doubaohaole", "douhaole",
               "quanhaole", "keyi", "chulihaole"),
    "hanzi": ("是的", "对", "没错", "好了", "都好了", "全好了", "可以", "处理好了"),
}


def voice_clear(text: str, machine: Machine, confirming: bool = False) -> ClearResult | None:
    """工人说"料加好了""清理完了",就把对应的报警解除。

    没有报警挂着时一律不动 —— 否则日常对话里一句"好了"就会莫名其妙播一段解除词。
    confirming=True 表示上一句助手刚问过"都处理好了吗",这一句是在回答。
    """
    if not text or not machine.active_alerts():
        return None
    probe, style = _key(text)
    active = machine.active_alerts()

    # 正在等确认:应一声就全清。这一步要排在最前面 ——
    # "都好了"里也有"好了",落到下面会被当成只解一条
    if confirming and any(w in probe for w in _CONFIRM[style]):
        return ClearResult(cleared=machine.clear_all())

    for i, alert in enumerate(ALERTS):
        # 每条信号自带解除词(Alert.clear_words),加一条信号时顺手写上就行,
        # 不用再维护一张平行的词表
        words = alert.clear_words
        if not words:
            continue
        probes = words if style == "hanzi" else tuple(_key(w)[0] for w in words)
        if not any(w in probe for w in probes):
            continue
        # 说到了某一条的专用词,就到此为止:说中的那条正挂着就解,没挂着就什么都不做。
        # 不能往下掉到通用词那一层 —— "料加好了"里也有"好了",堵塞报警挂着时
        # 会被当成"处理好了"把堵塞解掉,而工人根本没碰传送带
        if not machine.is_active(i):
            return None
        return ClearResult(cleared=[machine.clear(i)])

    # 明确说"都好了""全部处理好了":工人自己担保,全清
    if any(w in probe for w in _CLEAR_ALL[style]):
        return ClearResult(cleared=machine.clear_all())

    # 笼统的"处理好了":挂着一条就解那条,挂着好几条就反问,别替他认了没做的事
    if any(w in probe for w in _CLEAR_GENERIC[style]):
        if len(active) == 1:
            i = ALERTS.index(active[0])
            return ClearResult(cleared=[machine.clear(i)])
        return ClearResult(ask=active)
    return None


def cleared_speech(cleared: "list[Alert]") -> str:
    """解除之后念的那句。一条就念它自己的解除词,多条要逐条点名。

    只说"全部解除"是不够的 —— 工人得听见**哪几条**被解了,
    才知道系统和他理解的是不是同一件事。
    """
    if not cleared:
        return ""
    if len(cleared) == 1:
        return cleared[0].cleared
    return (f"{len(cleared)} 条异常全部解除，分别是"
            + "、".join(a.title for a in cleared) + "。设备恢复正常运行。")


def confirm_question(active: "list[Alert]") -> str:
    """挂着好几条时,笼统的"处理好了"要先问一句。"""
    return (f"当前有 {len(active)} 条异常，分别是"
            + "、".join(a.title for a in active)
            + "。都处理好了吗？")


def system_context(machine: Machine) -> str:
    """追加到 LLM 系统提示后面的一段。

    查询表只管得住问得直白的那些("温度多少");问得绕的("这个温度正常吗"、
    "还能撑多久")还是要靠模型,那就得让模型手里有真实数字 —— 否则它一定会编。
    每次请求都重新取一次,数字才是当下的。
    """
    return (
        "【背景资料，不是用户说的话】这条包装线当前状态："
        + machine.status_text()
        + "用途:用户问到某一项时照这里的数字答,不要自己编。"
        # 这句是必须的:不写的话,模型会把整串状态当成"该说的内容"背出来 ——
        # 问"你好"答"封口温度168度、速度42包每分钟、已完成3860包"
        + "用户没问到就不要提,更不要整串复述;和设备无关的话题就正常聊。"
    )


def main() -> int:
    import winutil

    winutil.setup_console()

    m = Machine()
    if len(sys.argv) > 1:
        for t in sys.argv[1:]:
            a = answer_query(t, m)
            print(f"  「{t}」 → {a or '(不是状态查询，交给 LLM)'}")
        return 0

    print("设备状态:", m.status_text(), "\n")
    print("报警文案:")
    for a in ALERTS:
        print(f"  [{a.title}] 报警 {a.speech}\n"
              f"            解除 {a.cleared}\n"
              f"            终端 {a.detail}")
    print("\n问答自检:")
    for t in ("机器目前温度和速度是什么", "今天我们完成了多少个包装",
              "文度是多少", "调高温度", "今天天气不错"):
        a = answer_query(t, m)
        print(f"  「{t}」 → {a or '(不是状态查询，交给 LLM)'}")
    print("\n语音解除(报警挂着时工人说一句就撤):")
    for raise_i, said in ((0, "料加好了"), (1, "杂物清理干净了"), (1, "料加好了"),
                          (0, "今天天气不错")):
        probe = Machine()
        probe.raise_alert(raise_i)
        got = voice_clear(said, probe)
        print(f"   {ALERTS[raise_i].title:8s}报警中,说「{said}」 → "
              + (f"解除,念「{got.cleared}」" if got else "不动它"))

    print("\n热键那条线(1/2 报警,3/4 解除对应那条):")
    n = len(ALERTS)
    for ch in "1234":
        i = int(ch) - 1
        if i < n:
            print(f"   [{ch} 报警] {m.raise_alert(i).speech}")
        else:
            a = m.clear(i - n)
            print(f"   [{ch} 解除] {a.cleared if a else '当前没有这条报警,忽略'}")
        print(f"            {m.status_text()}  ·  问速度 → {answer_query('速度多少', m)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
