#!/usr/bin/env python3
"""
唤醒词误唤醒 / 漏检评测:拿同一批录音比阈值、比唤醒词。

"旁边说一堆无关的话也容易醒"这种问题,凭感觉换词、凭感觉调阈值都靠不住 ——
要先有数。两个指标:

    漏检率      真叫了它却没醒(正样本里没检出的比例)
    误唤醒      没叫它却醒了。报两种口径:
                  按条   负样本里有多少条触发过(「1000 句闲聊里醒了几次」)
                  按时长 每小时醒几次(长录音用这个,和现场体感直接对得上)

两者此消彼长:阈值调高误唤醒少了,漏检就多。这个脚本把一串阈值一次扫完,
按「每小时误唤醒不超过 --max-fa 次」挑出漏检最少的那个阈值,几个唤醒词各挑各的,
最后并排比 —— 「你好小智」和「你好马高」谁更好,就看这一张表。

数据目录约定:
    data/
      negative/              没叫它的录音。车间闲聊、机器噪声、念命令词
                             ("处理好了""全部解除")都往这里放,任何一次检出都算误唤醒
      positive/你好小智/      叫了它的录音。默认一个文件喊一次;
      positive/你好马高/      长录音在文件名里写 _x17 表示这段里喊了 17 次

评测喂的是**录下来的原始音频 + AGC**,和 realtime_asr.py 里唤醒词那一路一样
(不降噪,见那边 dn_stream 的注释)。所以录音别事先放大、别降噪,原样存。

用法(PowerShell):
    # 1. 录正样本:按提示喊 50 次,每次 2.5 秒一个文件。换着距离、音量、语速喊
    .venv\\Scripts\\python.exe wake_eval.py record --out data\\positive\\你好马高 --say 你好马高 --count 50

    # 2. 录负样本:麦克风放在现场,录一小时,每分钟切一个文件
    .venv\\Scripts\\python.exe wake_eval.py record --out data\\negative --minutes 60

    # 3. 评测:两个唤醒词、一串阈值一次扫完
    .venv\\Scripts\\python.exe wake_eval.py run --data data --keywords 你好小智 你好马高
    .venv\\Scripts\\python.exe wake_eval.py run --data data --keywords 你好小智 --thresholds 0.1 0.2 0.3 --csv result.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import gain
import wakeword
import winutil

HERE = Path(__file__).resolve().parent
SR = wakeword.MODEL_SAMPLE_RATE
BLOCK_SECONDS = 0.1  # 和 realtime_asr.BLOCK_SECONDS 一致:流式 KWS 的分块会影响检出

AUDIO_EXT = {".wav", ".flac", ".ogg"}
DEFAULT_THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4]

# 每个文件末尾补的静音。短片段里唤醒词紧贴着文件结尾,KWS 要再看几帧才敢报,
# 不补的话最后那一声会被算成漏检 —— 真麦克风上后面总是还有音频的
TAIL_PAD_SECONDS = 1.0

# 空文件 / 几乎没声音的文件不算数。录音设备选错、麦克风静音时录出来的就是这种,
# 算进正样本会把漏检率凭空抬高,而且看表完全看不出是数据坏了
MIN_FILE_SECONDS = 0.3
MIN_FILE_PEAK = 1e-3

# 文件名里的 _x17 = 这段录音里喊了 17 次
_COUNT_RE = re.compile(r"_x(\d+)(?=\D*$)")


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def audio_files(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in AUDIO_EXT)


def expected_count(path: Path) -> int:
    m = _COUNT_RE.search(path.stem)
    return int(m.group(1)) if m else 1


def load_blocks(path: Path, gain_mode: str) -> tuple[list[np.ndarray], float]:
    """读文件 → 16 kHz → 过 AGC → 切成 0.1 秒一块。返回 (块, 原始时长秒)。

    AGC 每个文件新开一个:现场唤醒词那一路前面就是它,不过的话电平低的录音
    会全被判成漏检,测出来的是麦克风电平而不是唤醒词本身。
    """
    from realtime_asr import load_audio_file

    audio = load_audio_file(str(path), SR)
    dur = len(audio) / SR
    if dur < MIN_FILE_SECONDS or not audio.size or float(np.abs(audio).max()) < MIN_FILE_PEAK:
        return [], dur
    audio = np.concatenate([audio, np.zeros(int(TAIL_PAD_SECONDS * SR), np.float32)])
    n = int(BLOCK_SECONDS * SR)
    if len(audio) % n:
        audio = np.pad(audio, (0, n - len(audio) % n))
    blocks = [audio[i : i + n] for i in range(0, len(audio), n)]
    agc = gain.parse_gain(gain_mode)
    if agc is not None:
        blocks = [agc(b) for b in blocks]
    return blocks, dur


# --------------------------------------------------------------------------- #
# 评测
# --------------------------------------------------------------------------- #
class Cell:
    """一个 (唤醒词, 阈值) 组合的检出器和计数。"""

    def __init__(self, model: str, keyword: str, threshold: float, score: float) -> None:
        self.keyword = keyword
        self.threshold = threshold
        self.ww = wakeword.WakeWord(
            model, keywords=[keyword], threshold=threshold, score=score,
            num_threads=1, block_seconds=BLOCK_SECONDS,
        )
        if not self.ww.keywords:
            raise ValueError(f"「{keyword}」编不进 KWS 词表:{'、'.join(self.ww.skipped)}")
        self.pos_expected = 0
        self.pos_detected = 0
        self.neg_hits = 0
        self.neg_files_hit = 0
        self.fa: list[tuple[Path, list[float]]] = []  # 误唤醒的文件和时刻,给人去听

    def run(self, blocks: list[np.ndarray]) -> list[float]:
        """整段喂一遍,返回每次检出的时刻(秒)。"""
        self.ww.reset()
        hits = []
        for i, b in enumerate(blocks):
            if self.ww.accept(b):
                hits.append(i * BLOCK_SECONDS)
        return hits

    # --- 指标 --- #
    @property
    def miss_rate(self) -> float | None:
        if not self.pos_expected:
            return None
        return 1.0 - self.pos_detected / self.pos_expected


def evaluate(args) -> int:
    data = Path(args.data)
    model = args.model or wakeword.find_kws_model(HERE / "models")
    if not model:
        print(wakeword.DOWNLOAD_HELP)
        return 1

    neg = audio_files(data / "negative")
    pos = {kw: audio_files(data / "positive" / kw) for kw in args.keywords}
    if not neg and not any(pos.values()):
        print(f"{data} 底下没有录音。目录约定见 wake_eval.py 开头,或者先用 record 子命令录")
        return 1

    print(f"KWS 模型: {Path(model).name}  加分 {args.score}  增益 {args.gain}")
    cells: list[Cell] = []
    for kw in args.keywords:
        for th in args.thresholds:
            try:
                cells.append(Cell(model, kw, th, args.score))
            except (ValueError, RuntimeError) as exc:
                print(f"[跳过] {exc}")
                break
    if not cells:
        return 1

    # 负样本喂给所有检出器;正样本只喂给它自己那个唤醒词的检出器。
    # 每个文件只读一次、AGC 只过一次,然后并行喂给各个检出器
    # (onnxruntime 解码时会放开 GIL,线程池就够用)
    jobs: list[tuple[Path, str | None]] = [(p, None) for p in neg]
    for kw, files in pos.items():
        jobs += [(p, kw) for p in files]

    neg_seconds = 0.0
    n_neg = 0  # 真正算进统计的负样本条数(空文件不算)
    t0 = time.time()
    audio_seconds = 0.0
    bad: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for n, (path, kw) in enumerate(jobs, 1):
            try:
                blocks, dur = load_blocks(path, args.gain)
            except Exception as exc:  # 坏文件跳过,别让一个文件拖垮一小时的评测
                print(f"\n[跳过] {path}: {exc}")
                blocks, dur = [], 0.0
            if not blocks:
                bad.append(path)
                continue
            audio_seconds += dur
            targets = [c for c in cells if kw is None or c.keyword == kw]
            results = list(pool.map(lambda c: c.run(blocks), targets))
            for c, hits in zip(targets, results):
                if kw is None:
                    c.neg_hits += len(hits)
                    if hits:
                        c.neg_files_hit += 1
                        c.fa.append((path, hits))
                else:
                    want = expected_count(path)
                    c.pos_expected += want
                    c.pos_detected += min(len(hits), want)
            if kw is None:
                neg_seconds += dur
                n_neg += 1
            if n % 20 == 0 or n == len(jobs):
                spent = time.time() - t0
                sys.stdout.write(f"\r已评测 {n}/{len(jobs)} 个文件,"
                                 f"音频 {audio_seconds / 60:.1f} 分钟,用时 {spent:.0f}s")
                sys.stdout.flush()
    print("\n")
    if bad:
        print(f"[!] {len(bad)} 个文件是空的 / 几乎没声音 / 读不了,没算进统计"
              f"(录音设备选错或者麦克风静音?):")
        for p in bad[:10]:
            print(f"      {p}")
        if len(bad) > 10:
            print(f"      ……还有 {len(bad) - 10} 个")
        print()

    report(args, cells, n_neg, neg_seconds)
    if args.csv:
        write_csv(args.csv, cells, n_neg, neg_seconds)
        print(f"\n明细已写到 {args.csv}")
    return 0


def per_hour(c: Cell, neg_seconds: float) -> float | None:
    return c.neg_hits / (neg_seconds / 3600) if neg_seconds else None


def pick(cells: list[Cell], neg_seconds: float, max_fa: float) -> Cell | None:
    """每小时误唤醒 ≤ max_fa 的阈值里,挑漏检最少的;同样少就挑阈值高的(更保守)。"""
    ok = [c for c in cells
          if (fa := per_hour(c, neg_seconds)) is not None and fa <= max_fa]
    if not ok:
        return None
    return min(ok, key=lambda c: (c.miss_rate if c.miss_rate is not None else 0.0,
                                  -c.threshold))


def pct(x: float | None) -> str:
    return "  -  " if x is None else f"{x:6.1%}"


def report(args, cells: list[Cell], n_neg: int, neg_seconds: float) -> None:
    if not n_neg:
        print("[!] 没有负样本(data/negative):只能看漏检,误唤醒量不出来")
    elif neg_seconds < 1800:
        # 半小时里 0 次误唤醒,只能说明每小时 ≲ 2 次;挑阈值会偏乐观
        print(f"[!] 负样本只有 {neg_seconds / 60:.0f} 分钟。每小时误唤醒这一列的分辨率是 "
              f"{3600 / neg_seconds:.1f} 次,想比出 1 次/小时的差别至少录一小时")

    best: dict[str, Cell | None] = {}
    for kw in dict.fromkeys(c.keyword for c in cells):
        mine = [c for c in cells if c.keyword == kw]
        n_pos = mine[0].pos_expected
        print(f"「{kw}」  正样本 {n_pos} 次 · 负样本 {n_neg} 条 共 {neg_seconds / 3600:.2f} 小时")
        print(f"   阈值    漏检率          误唤醒(按条)          每小时误唤醒")
        for c in mine:
            miss = (f"{pct(c.miss_rate)} ({c.pos_expected - c.pos_detected:>3})"
                    if c.pos_expected else "     -      ")
            fa_files = (f"{pct(c.neg_files_hit / n_neg)} ({c.neg_files_hit:>4}/{n_neg})"
                        if n_neg else "        -         ")
            fa_h = per_hour(c, neg_seconds)
            print(f"   {c.threshold:<6.3g}  {miss}   {fa_files}    "
                  f"{'-' if fa_h is None else f'{fa_h:.1f}'}")
        if not n_pos:
            print("   [!] 没有正样本(data/positive/%s),漏检量不出来" % kw)
        b = pick(mine, neg_seconds, args.max_fa) if n_neg else None
        best[kw] = b
        if b is not None:
            print(f"   → 推荐阈值 {b.threshold:g}:每小时误唤醒 ≤ {args.max_fa:g} 的阈值里漏检最少")
            if b.fa and args.show_fa:
                print(f"     这个阈值下误唤醒的录音(去听一下被什么话触发的):")
                for path, hits in b.fa[: args.show_fa]:
                    ts = "、".join(f"{t:.1f}s" for t in hits)
                    print(f"       {path}  @ {ts}")
                if len(b.fa) > args.show_fa:
                    print(f"       ……还有 {len(b.fa) - args.show_fa} 个")
        elif n_neg:
            print(f"   → 扫过的阈值里没有一个能把每小时误唤醒压到 {args.max_fa:g} 以下,"
                  f"往上加阈值再扫,或者换词")
        print()

    ranked = [(kw, b) for kw, b in best.items() if b is not None]
    if len(best) > 1 and ranked:
        print(f"各唤醒词在自己的推荐阈值上(每小时误唤醒 ≤ {args.max_fa:g}):")
        ranked.sort(key=lambda kb: (kb[1].miss_rate or 0.0, per_hour(kb[1], neg_seconds)))
        for kw, b in ranked:
            print(f"   「{kw}」 阈值 {b.threshold:g}  漏检 {pct(b.miss_rate).strip()}  "
                  f"每小时误唤醒 {per_hour(b, neg_seconds):.1f}")
        missing = [kw for kw, b in best.items() if b is None]
        if missing:
            print(f"   「{'」「'.join(missing)}」在扫过的阈值里达不到这个误唤醒上限")
        print(f"   按「先比漏检、漏检一样再比误唤醒」排,最好的是「{ranked[0][0]}」")


def write_csv(path: str, cells: list[Cell], n_neg: int, neg_seconds: float) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:  # 带 BOM,Excel 直接打开不乱码
        w = csv.writer(f)
        w.writerow(["keyword", "threshold", "pos_expected", "pos_detected", "miss_rate",
                    "neg_files", "neg_files_hit", "neg_hits", "neg_hours", "fa_per_hour"])
        for c in cells:
            fa_h = per_hour(c, neg_seconds)
            w.writerow([c.keyword, c.threshold, c.pos_expected, c.pos_detected,
                        "" if c.miss_rate is None else f"{c.miss_rate:.4f}",
                        n_neg, c.neg_files_hit, c.neg_hits, f"{neg_seconds / 3600:.3f}",
                        "" if fa_h is None else f"{fa_h:.2f}"])


# --------------------------------------------------------------------------- #
# 录音
# --------------------------------------------------------------------------- #
def record(args) -> int:
    import sounddevice as sd
    import soundfile as sf

    from realtime_asr import pick_input_samplerate

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    if dev is not None and str(dev).lstrip("-").isdigit():
        dev = int(dev)
    sr = pick_input_samplerate(dev, SR)
    print(f"输入设备: {sd.query_devices(dev, 'input')['name']}  {sr} Hz → {out.resolve()}")
    stamp = time.strftime("%Y%m%d_%H%M%S")

    if args.say:
        # 正样本:一次一个短片段。给人留出换姿势的空当,别连着机械地喊 ——
        # 现场叫它的时候有远有近、有大声有随口,样本要覆盖这些
        print(f"每次提示后说一声「{args.say}」,{args.clip:.1f} 秒一段,共 {args.count} 次。"
              f"换着距离、音量、语速说。Ctrl-C 提前结束\n")
        try:
            for i in range(1, args.count + 1):
                input(f"[{i}/{args.count}] 准备好按回车,然后说「{args.say}」 ")
                audio = sd.rec(int(args.clip * sr), samplerate=sr, channels=1,
                               dtype="float32", device=dev)
                sd.wait()
                path = out / f"{stamp}_{i:03d}.wav"
                sf.write(str(path), audio[:, 0], sr, subtype="PCM_16")
                peak = float(np.abs(audio).max())
                warn = "  ← 太小声,几乎没录到" if peak < 0.005 else ""
                print(f"      存为 {path.name}  峰值 {peak:.3f}{warn}")
        except KeyboardInterrupt:
            print("\n提前结束")
        return 0

    # 负样本:长时间连续录,按分钟切文件 —— 中途断了不至于全丢,
    # 误唤醒报出来时也能直接打开那一分钟去听
    total = args.minutes * 60
    chunk = int(args.chunk * sr)
    print(f"连续录 {args.minutes:g} 分钟,每 {args.chunk:.0f} 秒一个文件。这段时间里**别喊唤醒词**。"
          f"Ctrl-C 提前结束(已录的保留)\n")
    buf: list[np.ndarray] = []
    have = 0
    n_file = 0
    recorded = 0.0

    def dump() -> None:
        nonlocal buf, have, n_file
        if not buf:
            return
        n_file += 1
        path = out / f"{stamp}_{n_file:04d}.wav"
        sf.write(str(path), np.concatenate(buf), sr, subtype="PCM_16")
        buf, have = [], 0

    try:
        with sd.InputStream(device=dev, samplerate=sr, channels=1, dtype="float32",
                            blocksize=int(0.1 * sr)) as stream:
            while recorded < total:
                block, _ = stream.read(int(0.1 * sr))
                buf.append(block[:, 0].copy())
                have += len(block)
                recorded += len(block) / sr
                if have >= chunk:
                    dump()
                    sys.stdout.write(f"\r已录 {recorded / 60:.1f}/{args.minutes:g} 分钟,{n_file} 个文件")
                    sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    dump()
    print(f"\n共 {recorded / 60:.1f} 分钟,{n_file} 个文件")
    return 0


# --------------------------------------------------------------------------- #
def main() -> int:
    winutil.setup_console()
    p = argparse.ArgumentParser(description="唤醒词误唤醒 / 漏检评测")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="评测:扫阈值、比唤醒词")
    r.add_argument("--data", default="data", help="数据目录(约定见文件开头)")
    r.add_argument("--keywords", nargs="+", default=[wakeword.DEFAULT_KEYWORD],
                   help="要比的唤醒词,正样本放在 data/positive/<唤醒词>/ 下")
    r.add_argument("--thresholds", nargs="+", type=float, default=DEFAULT_THRESHOLDS)
    r.add_argument("--score", type=float, default=wakeword.DEFAULT_SCORE,
                   help="解码加分,和 realtime_asr.py --wake-score 同一个")
    r.add_argument("--gain", default="auto",
                   help="和 realtime_asr.py --gain 同一个。录音时用的什么这里就用什么")
    r.add_argument("--max-fa", type=float, default=1.0,
                   help="挑阈值时能接受的每小时误唤醒次数上限")
    r.add_argument("--show-fa", type=int, default=10,
                   help="推荐阈值下列出多少个误唤醒的文件(0 = 不列)")
    r.add_argument("--csv", default=None, help="把每个 (唤醒词, 阈值) 的明细写成 CSV")
    r.add_argument("--model", default=None, help="KWS 模型目录,默认在 models\\ 里找")
    r.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="并行几个检出器")

    c = sub.add_parser("record", help="录评测数据")
    c.add_argument("--out", required=True, help="存到哪个目录")
    c.add_argument("--say", default=None,
                   help="录正样本:要喊的唤醒词。不给就是录负样本(长时间连续录)")
    c.add_argument("--count", type=int, default=50, help="正样本录几次")
    c.add_argument("--clip", type=float, default=2.5, help="正样本每段几秒")
    c.add_argument("--minutes", type=float, default=60, help="负样本录多久")
    c.add_argument("--chunk", type=float, default=60, help="负样本每个文件几秒")
    c.add_argument("--device", default=None, help="输入设备编号")

    args = p.parse_args()
    return evaluate(args) if args.cmd == "run" else record(args)


if __name__ == "__main__":
    sys.exit(main())
