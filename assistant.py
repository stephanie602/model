#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 接在 ASR 后面:识别出的文字送进去,流式吐回答。

模型跑在一个独立的 OpenAI 兼容 server 里(llama.cpp 的 llama-server.exe),
这边只负责流式收字。为什么不在进程里加载模型:

  · llama.cpp 在 Windows 上有多个后端(CPU / Vulkan / CUDA / SYCL),换后端只是
    换一个 llama-server.exe,这边一行代码都不用动。
  · 模型换大换小(4B ↔ 1.7B)变成改一个启动参数,不用碰代码,也不用重启语音进程。
  · ASR 和 TTS 已经在抢 CPU 了,LLM 单独一个进程,OOM 的时候死的是它,
    不会把整条语音链路带走。
  · LLM 还可以直接放到另一台机器上,--chat-url 指过去就行。

起 server(start_llm.ps1 干的就是这件事):
    .\\llama-server.exe -m Qwen3-4B-Instruct-2507-Q4_K_M.gguf -c 4096 -t 6 --port 8080

ASR 的输入有两个特点,系统提示里要交代清楚:没有标点,而且可能有同音错字。
所以让模型按上下文猜意思,别去纠字面。
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque

DEFAULT_CHAT_URL = "http://127.0.0.1:8080/v1"  # llama-server 的默认落点

DEFAULT_SYSTEM = (
    "你是一个语音助手。用户的话是语音识别转出来的,没有标点,可能有同音错字,"
    "按上下文理解意思,不要纠正字面。"
    "回答要短:一句话说完,不超过 15 个字,口语化,"
    "直接给结论,不要用“好的”“嗯”“我看一下”这类开场白。"
    "不要用列表、不要用 Markdown、不要加解释性前缀。"
    # 试过让模型"先给一个短语再加逗号"(为了让第一段短、合成快),结果它把
    # "逗号"两个字念进了回答里 —— "没问题,逗。"。这类对输出格式的细指令
    # 4B 模型接不住,撤掉。短第一段改在切句那一层做(tts.SentenceBuffer)
    # 长度不只是风格问题:合成比实时慢(RTF 2.3),回答每多 10 个字,
    # 用户就要多等 4 秒才听完 —— 这条约束是延迟的一部分

)


class HttpAssistant:
    """把生成扔给一个 OpenAI 兼容的 server,自己只负责流式收字。

    带多轮上下文,一次只服务一个说话人,不做并发。
    """

    def __init__(
        self,
        base_url: str = DEFAULT_CHAT_URL,
        model: str = "local",
        system: str = DEFAULT_SYSTEM,
        max_tokens: int = 200,
        history_turns: int = 6,
        api_key: str | None = None,
        timeout: float = 120.0,
        context: "callable | None" = None,
    ) -> None:
        import requests

        self._requests = requests
        # 每次请求前调一次,把返回的文字追加到系统提示后面。设备状态这类
        # **会变的**上下文只能这样给:写死在 system 里的话,温度是启动那一刻的,
        # 十分钟后模型还照着念老数字,比不知道更糟。
        self.context = context
        self.base_url = base_url.rstrip("/")
        self.url = f"{self.base_url}/chat/completions"
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.timeout = timeout
        # 只保留最近几轮,免得 KV cache 和上下文一直涨
        self.history: deque[dict[str, str]] = deque(maxlen=history_turns * 2)

        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

        self._session = requests.Session()

    def _messages(self, user_text: str,
                  extra_system: str | None = None) -> list[dict[str, str]]:
        extra = ""
        if self.context is not None:
            try:
                extra = self.context()
            except Exception:
                extra = ""  # 取状态失败不该让回答整个失败,退回没有状态的普通对话
        # 会变的东西(设备状态)放在**最后一条消息**里,不放系统提示。
        #
        # 位置不是风格问题,是速度问题:llama-server 会缓存前缀的 KV,前缀没变
        # 就不用重新预填。设备状态每次请求都不一样,搁在系统提示(第 0 条)里,
        # 后面所有消息的缓存全作废,每轮都要从头预填一遍 —— 实测首字从
        # 0.39s 涨到 0.95s。挪到末尾,系统提示和历史都还在缓存里。
        msgs = [{"role": "system", "content": self.system}]
        msgs.extend(self.history)
        # 顺序:系统提示 → 历史 → (本轮的手册内容) → 设备状态 + 用户这句话。
        # 手册内容放在最后一条 system 里而不是塞进用户消息,是为了让模型分得清
        # "这是资料"和"这是用户说的话" —— 混在一起它会把手册里的句子当成用户的要求
        if extra_system:
            msgs.append({"role": "system", "content": extra_system})
        # 设备状态必须走 system,不能贴在用户那句话前面。
        #
        # 原来是 f"{状态}\n{用户说的话}" 一起塞进 user —— 4B 模型分不清哪半句是
        # 用户说的,于是把状态原样背出来:实测 5 句里 4 句中招,问"你好"答
        # "封口温度168度，速度42包/分钟，已完成3860包",问"今天天气不错"也一样。
        # 换成独立的 system 消息,角色分清楚了;仍然放在最后,所以前缀缓存不受影响。
        if extra:
            msgs.append({"role": "system", "content": extra})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def probe(self) -> str:
        """启动时确认 server 真的在,顺便把它实际加载的模型名报出来。

        不做这一步的话,server 没起来会等到用户说完第一句才报错 —— 那时候已经
        加载完 ASR 和 TTS 了,白等一分钟。
        """
        try:
            r = self._session.get(f"{self.base_url}/models", headers=self._headers, timeout=10)
            r.raise_for_status()
            data = r.json().get("data") or []
            return data[0].get("id", self.model) if data else self.model
        except Exception as exc:
            raise RuntimeError(
                f"连不上 LLM server {self.base_url}:{exc}\n"
                "先把它起起来,在另一个 PowerShell 窗口里:\n"
                "  .\\start_llm.ps1"
            ) from exc

    def reply_stream(self, user_text: str, extra_system: str | None = None,
                     max_tokens: int | None = None):
        """逐段产出回答文本。调用方负责打印,这样能边生成边显示。

        extra_system 只对这一轮生效 —— 手册检索的结果走这里:它是"这个问题"
        查到的内容,不该留在系统提示里影响下一个问题,也不该进对话历史
        (历史里留的是问和答,不是当时翻到的手册页)。
        """
        payload = {
            "model": self.model,
            "messages": self._messages(user_text, extra_system),
            # 手册问答那一轮要得比闲聊多:故障有好几条原因,额度不够会被截在半句
            "max_tokens": max_tokens or self.max_tokens,
            # temperature=0:同一句话每次回答一致,调试和复现都方便
            "temperature": 0.0,
            "stream": True,
        }

        pieces: list[str] = []
        with self._session.post(
            self.url,
            json=payload,
            headers=self._headers,
            stream=True,
            timeout=self.timeout,
        ) as resp:
            if resp.status_code >= 400:
                raise RuntimeError(f"LLM server 返回 {resp.status_code}: {resp.text[:200]}")

            # 必须自己按字节收、自己解 UTF-8,不能用 decode_unicode=True。
            # requests 对 text/* 的响应在没有 charset 声明时按 RFC 2616 默认
            # ISO-8859-1 解码,而 llama.cpp 的 SSE 正是 Content-Type: text/event-stream
            # 不带 charset,发的又是 UTF-8 中文 —— 于是"你好"变成"ä½ å¥½"。
            # 乱码还能通过 json.loads(语法没坏),一路流到 TTS 去念,听起来就是一串怪音。
            #
            # 按行切是安全的:UTF-8 的多字节序列里不会出现 0x0A,所以行边界不会
            # 劈开一个汉字;每行又都是完整的 JSON,逐行解码不会截断字符。
            for raw in resp.iter_lines(decode_unicode=False):
                if not raw:
                    continue  # SSE 的心跳空行
                line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
                if not line.startswith("data:"):
                    continue  # SSE 的注释行
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"]
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue  # 半行或者 server 自己加的字段,跳过就是了
                chunk = delta.get("content")
                if chunk:
                    pieces.append(chunk)
                    yield chunk

        answer = "".join(pieces).strip()
        # 历史里只留用户原话,不留那份设备状态 —— 它是当时的快照,
        # 留着会让模型看到一堆互相矛盾的旧温度,也白占前缀
        self.history.append({"role": "user", "content": user_text})
        self.history.append({"role": "assistant", "content": answer})

    def reply(self, user_text: str) -> str:
        return "".join(self.reply_stream(user_text)).strip()


# --------------------------------------------------------------------------- #
# 接在 ASR 后面的工作线程
# --------------------------------------------------------------------------- #
class ChatWorker:
    """把定稿文本排进队列,单独一个线程生成回答。

    必须和识别线程分开:4B 模型生成一句话要一两秒,压在识别线程里会让后面的语音
    全部积压。队列只保留最新一条 —— 用户连着说了好几句时,回答最后一句才符合直觉,
    也免得回答排队排到几十秒之后。

    带上 speaker 时,回答是边生成边念的:文字流按标点攒成一句就立刻丢给 TTS 合成,
    不等整段回答写完。详见 tts.py。
    """

    def __init__(
        self,
        assistant: HttpAssistant,
        print_lock: threading.Lock,
        prefix: str = "[答] ",
        min_chars: int = 2,
        speaker=None,
        timing: bool = False,
        min_clause_chars: int | None = None,
        sync_text: bool = True,
        first_chunk_chars: int | None = None,
        fillers: "list[str] | None" = None,
    ) -> None:
        self.assistant = assistant
        self.print_lock = print_lock
        self.prefix = prefix
        self.min_chars = min_chars
        self.speaker = speaker
        self.timing = timing
        # 文字跟着声音走,而不是跟着 LLM 走。
        #
        # 默认(False)是边生成边打:LLM 一秒就把整段吐完,而 TTS 合成比实时慢
        # (这台机器上 Kokoro 的 RTF 是 2.5),声音得念十秒 —— 屏幕上早就打完了,
        # 喇叭还在念第一句,看着像两个不相干的程序。开着这个开关,每句话等到
        # **真正开始出声**的那一刻才打出来,读到哪儿听到哪儿。
        #
        # 代价是"第一个字出现"变晚了(要等第一句合成完)。它治的是不同步,
        # 不是延迟本身 —— 延迟的根子在 TTS 合成速度,换模型才治得了。
        self.sync_text = sync_text and speaker is not None
        # 攒够多少个字才肯在逗号处切一句去合成。这个数直接决定"说完到出声"的延迟:
        # 调小 → 第一声来得早,但句子碎、韵律断;调大 → 听感连贯,但要多等。
        self.min_clause_chars = min_clause_chars
        # 第一小段攒够几个字就先送去合成(不等标点)。合成时间对长度有断崖,
        # 短的第一段能把"说完到出声"从 5 秒压到半秒 —— 见 tts.SentenceBuffer
        self.first_chunk_chars = first_chunk_chars
        # 应答词:一开口先念的那两个字("好的""我看看")。
        #
        # 为什么要它 —— 这条路的延迟是拆开量过的:ASR 0.09s、LLM 首字 0.39s,
        # 而回答的第一句要合成 5.5 秒。也就是说用户等的 6 秒里,有 5.5 秒
        # 是在等 TTS,LLM 早就答完了。第一句的合成没法变快(换模型才行),
        # 但可以让它别挡在最前面:应答词是写死的,开机预合成好,一有回答
        # 立刻出声,正文在它响的这一秒里接着合成。
        #
        # 这是把"等待"藏到应答后面,不是把总时长变短 —— 但人对"叫了有回应"
        # 和"叫了没动静"的感受完全不同,而后者会让人以为没听见,再问一遍。
        self.fillers = [f for f in (fillers or []) if f.strip()]
        self._filler_i = 0
        self._cv = threading.Condition()
        self._pending: str | None = None
        self._pending_extra: str | None = None  # 本轮的手册内容,见 submit()
        self._pending_tokens: int | None = None
        self._closed = False
        # "正在生成回答"。麦克风闸门要看它 —— 见 tts.Speaker.blocking_mic():
        # 边生成边念时两句之间会有空档,光看"有没有在播"会漏。
        self._busy = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        # 闸门要问"还在生成吗",在这里挂上去
        if self.speaker is not None:
            self.speaker.busy_hint = self.busy
        self._thread.start()

    def _wait_spoken(self, timeout: float) -> None:
        """等当前这段回答念完,但新问题一到就立刻返回。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._cv:
                if self._pending is not None:
                    return  # 下一句已经在门口了,别让它等
            self.speaker.wait_idle(timeout=0.2)
            with self.speaker._lock:
                if self.speaker._pending == 0:
                    return

    def busy(self) -> bool:
        """正在生成回答(可能刚念完一句、下一句还没吐出来)。"""
        with self._cv:
            return self._busy

    def submit(self, text: str, extra_system: str | None = None,
               max_tokens: int | None = None) -> None:
        text = text.strip()
        if len(text) < self.min_chars:
            return  # 单字多半是噪声误触发,不值得叫模型
        with self._cv:
            self._pending = text
            self._pending_extra = extra_system
            self._pending_tokens = max_tokens
            self._cv.notify()

    def note(self, user_text: str, answer: str) -> None:
        """把一轮"没经过 LLM 的对话"补记进历史。

        设备状态问答和主动报警都是绕开模型直接出声的(见 machine.py)。不补这一笔,
        模型就完全不知道刚才发生过什么,用户接着说"那怎么处理"时它只能反问
        "你指的是什么" —— 听起来像刚失忆。
        """
        if not user_text or not answer:
            return
        self.assistant.history.append({"role": "user", "content": user_text})
        self.assistant.history.append({"role": "assistant", "content": answer})

    def close(self, timeout: float = 60.0) -> None:
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._thread.join(timeout=timeout)

    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._closed:
                    self._cv.wait()
                if self._pending is None:
                    return
                text, self._pending = self._pending, None
                extra_system, self._pending_extra = self._pending_extra, None
                turn_tokens, self._pending_tokens = self._pending_tokens, None
                self._busy = True

            buf = None
            if self.speaker is not None:
                from tts import SentenceBuffer

                # 上一条回答可能还在念,新问题来了就掐掉 —— 用户已经问下一个了,
                # 再把旧答案念完只会打架
                self.speaker.interrupt()
                if self.fillers:
                    # 轮着用,免得每次都是同一句,听多了像录音
                    self.speaker.say(self.fillers[self._filler_i % len(self.fillers)])
                    self._filler_i += 1
                kw = {}
                if self.min_clause_chars:
                    kw["min_clause_chars"] = self.min_clause_chars
                if self.first_chunk_chars is not None:
                    kw["first_chunk_chars"] = self.first_chunk_chars
                buf = SentenceBuffer(**kw)

            # 用户实际感知的延迟是"说完到出声",而这段里最容易被误判的是
            # 到底卡在 LLM 生成还是 TTS 合成。分开记,别猜。
            t0 = time.time()
            t_first_token = None
            t_first_say = None
            n_chars = 0

            if self.sync_text:
                # 播放线程出声前回调这里。前缀留到第一句真出声时才打 ——
                # 先打一个空的"[答] "挂在那儿等好几秒,看着像卡死了
                first = [True]

                def on_speak(sentence: str) -> None:
                    with self.print_lock:
                        head = self.prefix if first[0] else ""
                        first[0] = False
                        sys.stdout.write(f"{head}{sentence}")
                        sys.stdout.flush()

                self.speaker.on_speak = on_speak

            try:
                if not self.sync_text:
                    with self.print_lock:
                        sys.stdout.write(f"\r\033[2K{self.prefix}")
                        sys.stdout.flush()
                for chunk in self.assistant.reply_stream(text, extra_system, turn_tokens):
                    if t_first_token is None:
                        t_first_token = time.time()
                    n_chars += len(chunk)
                    if not self.sync_text:
                        with self.print_lock:
                            sys.stdout.write(chunk)
                            sys.stdout.flush()
                    if buf is not None:
                        for sentence in buf.feed(chunk):
                            if t_first_say is None:
                                t_first_say = time.time()
                            self.speaker.say(sentence)
                if buf is not None:
                    for sentence in buf.flush():  # 结尾那半句多半没有标点
                        if t_first_say is None:
                            t_first_say = time.time()
                        self.speaker.say(sentence)
                if self.sync_text:
                    # 换行要等最后一句念完,否则计时那几行会插到回答中间。
                    # 但**新问题一来就别等了** —— 合成慢的时候一段回答要念半分钟,
                    # 干等着的话,用户这期间问的下一句要等上一句念完才开始处理,
                    # 表现就是"越问越慢"。这里等的只是一个换行符,不值得挡住下一轮
                    self._wait_spoken(timeout=180.0)
                with self.print_lock:
                    sys.stdout.write("\n")
                    sys.stdout.flush()

                if self.timing:
                    done = time.time()
                    parts = [f"首字 {(t_first_token or done) - t0:.2f}s"]
                    if t_first_say is not None:
                        parts.append(f"首句交给TTS {t_first_say - t0:.2f}s")
                    parts.append(f"全文 {done - t0:.2f}s / {n_chars} 字")
                    if t_first_token and done > t_first_token and n_chars > 1:
                        # 粗算,按字不按 token;中文大致 1 字 ≈ 0.7 token
                        parts.append(f"{n_chars / (done - t_first_token):.1f} 字/秒")
                    with self.print_lock:
                        sys.stdout.write(f"\033[2m[计时] LLM {' · '.join(parts)}\033[0m\n")
                        sys.stdout.flush()
            except Exception as exc:  # 回答失败不该带崩识别
                with self.print_lock:
                    sys.stdout.write(f"\r\033[2K[回答失败] {exc}\n")
                    sys.stdout.flush()
            finally:
                if self.sync_text:
                    # 一定要摘掉:留着的话,报警那条路(直接 say(),不经过这里)
                    # 出声时也会被打上"[答] "的头
                    self.speaker.on_speak = None
                # 生成结束就撤掉闸门的"忙"标志 —— 剩下的交给 _pending 和 TAIL_GUARD。
                # 放 finally 里:回答失败也必须撤,否则麦克风就一直哑着了。
                with self._cv:
                    self._busy = False
