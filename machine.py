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
    """

    __slots__ = ("key", "title", "speech", "detail", "cleared")

    def __init__(self, key: str, title: str, speech: str, detail: str = "",
                 cleared: str = "") -> None:
        self.key = key
        self.title = title
        self.speech = speech
        self.detail = detail
        self.cleared = cleared or "故障已排除，设备恢复正常运行。"

    def __repr__(self) -> str:
        return f"Alert({self.key!r}, {self.title!r})"


ALERTS: list[Alert] = [
    Alert(
        "material",
        "缺原材料",
        "注意，包装机原材料不足，请及时补充原材料。",
        "料位传感器低于下限 · 建议在 5 分钟内上料，否则整线停机",
        cleared="原材料已补充，料位恢复正常，设备继续运行。",
    ),
    Alert(
        "jam",
        "传送带异物堵塞",
        "注意，传送带有异物卡住，设备已自动停机，请清理杂物后重新启动。",
        "输送电机过载保护动作 · 清理后需手动复位再启动",
        cleared="传送带杂物已清理，设备已重新启动，恢复正常运行。",
    ),
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
        self.material_low = False
        self.jammed = False
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
                "material_low": self.material_low,
                "jammed": self.jammed,
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
        if s["material_low"]:
            parts.append("原材料不足")
        if s["jammed"]:
            parts.append("传送带被异物卡住")
        return "；".join(parts) + "。"

    # ---------------- 报警 ---------------- #

    def next_alert(self) -> Alert:
        """取下一条报警并让它作用到状态上,循环取。"""
        with self._lock:
            index = self._alert_i % len(ALERTS)
            self._alert_i += 1
        return self.raise_alert(index)

    def raise_alert(self, index: int) -> Alert:
        """直接触发第 index 条报警(从 0 起)。演示时要单独放某一条,不用一路按过去。

        报警必须改状态,不然演示会自相矛盾:刚播完"传送带卡住已停机",
        用户接着问速度,还答"每分钟 42 包"。
        """
        with self._lock:
            alert = ALERTS[index % len(ALERTS)]
            # 直选之后,空格接着按要从这一条的下一条继续,别跳回开头
            self._alert_i = (index % len(ALERTS)) + 1
            if alert.key == "material":
                self.material_low = True
            elif alert.key == "jam":
                self.jammed = True
                if self.running:
                    # 停机瞬间把产量冻住,之后再问"今天多少个"数字不会继续涨
                    elapsed = time.time() - self._t0
                    self._frozen = PACKAGES_BASE + int(elapsed * SPEED_SET / 60.0)
                self.running = False
            self.last_alert = alert
            return alert

    def is_active(self, index: int) -> bool:
        """第 index 条报警现在还挂着吗。"""
        alert = ALERTS[index % len(ALERTS)]
        with self._lock:
            return self.material_low if alert.key == "material" else self.jammed

    def active_alert(self) -> Alert | None:
        """当前挂着的报警(多条时给最近那条),没有就是 None。"""
        with self._lock:
            return self.last_alert if (self.material_low or self.jammed) else None

    def clear(self, index: int) -> Alert | None:
        """解除第 index 条报警,只解这一条。本来就没响则返回 None。

        按条解除而不是一把全清:两条报警可以同时挂着(缺料还没补,传送带又卡了),
        这时候"全清"会连没处理的那条一起抹掉,状态就和现场对不上了。
        """
        alert = ALERTS[index % len(ALERTS)]
        with self._lock:
            if alert.key == "material":
                if not self.material_low:
                    return None
                self.material_low = False
            else:
                if not self.jammed:
                    return None
                self.jammed = False
                self.running = True
                if self._frozen is not None:
                    # 停机期间不产出:把时间原点往后挪,让产量从冻结值接着涨
                    self._t0 = time.time() - (self._frozen - PACKAGES_BASE) * 60.0 / SPEED_SET
                    self._frozen = None
            if not (self.material_low or self.jammed):
                self.last_alert = None
            return alert

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
            self.material_low = False
            self.jammed = False
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
    if "overview" in hits or "judge" in hits:
        # 问好坏就先给一句结论(按状态判,不猜)。问题里点名了具体哪一项,
        # 就只答那一项 —— "这个温度正常吗"别把产量也倒出来
        parts.append(FRAMES["ok"] if s["running"] and not s["material_low"]
                     else FRAMES["not_ok"])
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


def voice_clear(text: str, machine: Machine) -> Alert | None:
    """工人说"料加好了""清理完了",就把对应的报警解除。

    没有报警挂着时一律不动 —— 否则日常对话里一句"好了"就会莫名其妙播一段解除词。
    返回被解除的那条报警,没解除返回 None。
    """
    if not text or machine.active_alert() is None:
        return None
    probe, style = _key(text)

    for i, alert in enumerate(ALERTS):
        words = _CLEAR_WORDS.get(alert.key)
        if not words or not any(w in probe for w in words[style]):
            continue
        # 说到了某一条的专用词,就到此为止:说中的那条正挂着就解,没挂着就什么都不做。
        # 不能往下掉到通用词那一层 —— "料加好了"里也有"好了",堵塞报警挂着时
        # 会被当成"处理好了"把堵塞解掉,而工人根本没碰传送带
        return machine.clear(i) if machine.is_active(i) else None

    # 通用说法:只有正好挂着一条时才认,两条都挂着就不猜
    if any(w in probe for w in _CLEAR_GENERIC[style]):
        active = [i for i in range(len(ALERTS)) if machine.is_active(i)]
        if len(active) == 1:
            return machine.clear(active[0])
    return None


def system_context(machine: Machine) -> str:
    """追加到 LLM 系统提示后面的一段。

    查询表只管得住问得直白的那些("温度多少");问得绕的("这个温度正常吗"、
    "还能撑多久")还是要靠模型,那就得让模型手里有真实数字 —— 否则它一定会编。
    每次请求都重新取一次,数字才是当下的。
    """
    return (
        "你正在给一条包装生产线当语音助手。设备实时状态："
        + machine.status_text()
        + "回答涉及数字时只能用上面给出的数字，不要自己编。"
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
