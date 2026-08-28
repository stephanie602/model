# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    Windows 上的一键环境准备:虚拟环境 + sherpa-onnx + 三个模型。

.DESCRIPTION
    ASR、降噪、TTS 三块全部由 sherpa-onnx 提供(一个库顶三个),LLM 是外部的
    OpenAI 兼容 server —— 那部分归 start_llm.ps1 管,这里不碰。

.EXAMPLE
    .\setup_windows.ps1              # 默认
    .\setup_windows.ps1 -CN          # 国内网络,pip 和 HuggingFace 都换镜像
    .\setup_windows.ps1 -Offline     # 模型已经从别处拷进 models\,只装 pip 包
    .\setup_windows.ps1 -Python "C:\Python311\python.exe"   # 指定用哪个 Python

.NOTES
    第一次跑之前可能要放开执行策略(只影响当前用户,不需要管理员):
        Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
    或者干脆绕过一次:
        powershell -ExecutionPolicy Bypass -File .\setup_windows.ps1
#>

[CmdletBinding()]
param(
    [switch]$CN,
    [switch]$Offline,
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
# 控制台按 UTF-8 输出,否则下面这些中文提示在默认 936 代码页下是乱码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# 老版 Win10 上 PowerShell 5.1 默认还在用 TLS 1.0,GitHub 和 PyPI 早就不收了,
# 表现是 Invoke-WebRequest 报"基础连接已关闭",看不出跟 TLS 有关
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Here   = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv   = Join-Path $Here ".venv"
$Models = Join-Path $Here "models"

function Say  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "[!] $m"   -ForegroundColor Yellow }
function Die  ($m) { Write-Host "[x] $m"   -ForegroundColor Red; exit 1 }

# 下载到指定路径;已经存在就跳过。返回 $true/$false,让调用方决定是致命还是可降级。
function Get-File($Url, $Out) {
    if ((Test-Path $Out) -and ((Get-Item $Out).Length -gt 0)) {
        Write-Host "  已存在,跳过: $(Split-Path -Leaf $Out)"
        return $true
    }
    if ($Offline) {
        Warn "  -Offline 但缺文件: $Out"
        return $false
    }
    Write-Host "  下载 $(Split-Path -Leaf $Out) ..."
    $part = "$Out.part"
    try {
        # curl.exe 是 Win10 1803+ 自带的,断点和进度都比 Invoke-WebRequest 好;
        # 更要紧的是 IWR 会把整个响应先读进内存,几百 MB 的模型很容易把它撑爆。
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            # Out-Null 挡住 stdout:这个函数靠返回值传成败,原生命令漏一个字符串
            # 出去就会和 $true/$false 一起进返回值数组。进度条走的是 stderr,照样看得到。
            & curl.exe -fSL --retry 3 --connect-timeout 20 --progress-bar -o $part $Url | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "curl 退出码 $LASTEXITCODE" }
        } else {
            $ProgressPreference = "SilentlyContinue"
            Invoke-WebRequest -Uri $Url -OutFile $part -UseBasicParsing
        }
        Move-Item -Force $part $Out
        return $true
    } catch {
        Remove-Item -Force -ErrorAction SilentlyContinue $part
        Warn "  下载失败: $_"
        return $false
    }
}

# 下载 .tar.bz2 并解开;目标目录已经在了就跳过
function Get-Tar($Url, $Name) {
    $dir = Join-Path $Models $Name
    if (Test-Path $dir) {
        Write-Host "  已存在,跳过: $Name"
        return $true
    }
    $tar = Join-Path $Models "$Name.tar.bz2"
    if (-not (Get-File $Url $tar)) { return $false }
    Write-Host "  解压 $Name ..."
    # Win10 1803+ 自带的 tar.exe 是 bsdtar,认 bz2。老系统上退回 Python 的 tarfile。
    if (Get-Command tar.exe -ErrorAction SilentlyContinue) {
        & tar.exe -xf $tar -C $Models | Out-Null
        if ($LASTEXITCODE -ne 0) { Warn "  tar 解压失败"; return $false }
    } else {
        & $script:PY -c "import tarfile,sys; tarfile.open(sys.argv[1]).extractall(sys.argv[2])" $tar $Models | Out-Null
        if ($LASTEXITCODE -ne 0) { Warn "  解压失败"; return $false }
    }
    Remove-Item -Force -ErrorAction SilentlyContinue $tar
    return $true
}


# --------------------------------------------------------------------------- #
Say "1/4  找 Python"
# --------------------------------------------------------------------------- #
if ($Python -eq "") {
    # py.exe(Python Launcher)比 python.exe 靠谱:后者在没装 Python 时会被
    # 系统那个"打开 Microsoft Store"的假 python.exe 抢走,报的错让人一头雾水
    if (Get-Command py.exe -ErrorAction SilentlyContinue) {
        $Python = "py.exe"
        $pyArgs = @("-3")
    } elseif (Get-Command python.exe -ErrorAction SilentlyContinue) {
        $Python = "python.exe"
        $pyArgs = @()
    } else {
        Die "找不到 Python。装 3.9-3.12 的 64 位版:https://www.python.org/downloads/windows/`n    安装时记得勾上 Add python.exe to PATH。"
    }
} else {
    $pyArgs = @()
}

$ver = & $Python @pyArgs -c "import sys,platform; print(f'{sys.version_info.major}.{sys.version_info.minor} {platform.machine()}')"
Write-Host "  $Python -> Python $ver"
if ($ver -notmatch "AMD64|ARM64") {
    Warn "  看起来是 32 位 Python。sherpa-onnx 只有 64 位 wheel,请换 64 位版。"
}


# --------------------------------------------------------------------------- #
Say "2/4  Python 虚拟环境"
# --------------------------------------------------------------------------- #
$script:PY = Join-Path $Venv "Scripts\python.exe"

# 从 Mac/Linux 拷过来的 .venv 在这儿一个字节都用不了(里面是 .so / bin 布局)。
# 只判断目录在不在是不够的,所以实际跑一下,跑不通就整个重建。
if ((Test-Path $Venv) -and -not (Test-Path $script:PY)) {
    Warn ".venv 存在但没有 Scripts\python.exe(多半是从 Mac/Linux 拷过来的),删掉重建"
    Remove-Item -Recurse -Force $Venv
}
if (-not (Test-Path $Venv)) {
    & $Python @pyArgs -m venv $Venv
    if ($LASTEXITCODE -ne 0) { Die "建 venv 失败" }
}

$pipArgs = @("--disable-pip-version-check")
if ($CN) { $pipArgs += @("-i", "https://pypi.tuna.tsinghua.edu.cn/simple") }

& $script:PY -m pip install -U @pipArgs pip wheel
if ($LASTEXITCODE -ne 0) { Die "pip 自升级失败,检查网络或加 -CN 换镜像" }

# sherpa-onnx 一个库同时提供 ASR / 降噪 / TTS / 唤醒词,Windows 官方 wheel 就是 CPU 后端。
# pypinyin 是给唤醒词用的:那个 KWS 模型按拼音建模,唤醒词要先从汉字转成声母韵母
# (纯 Python,几百 KB)。
& $script:PY -m pip install @pipArgs numpy scipy sounddevice soundfile requests sherpa-onnx pypinyin
if ($LASTEXITCODE -ne 0) { Die "依赖安装失败" }


# --------------------------------------------------------------------------- #
Say "3/4  模型文件"
# --------------------------------------------------------------------------- #
New-Item -ItemType Directory -Force -Path $Models | Out-Null
$REL = "https://github.com/k2-fsa/sherpa-onnx/releases/download"
if ($CN) {
    # GitHub release 在国内经常拉不动。这个前缀是常见的加速代理,失效了就换一个,
    # 或者直接从能上网的机器把 models\ 整个拷过来再跑 -Offline。
    $REL = "https://ghproxy.net/https://github.com/k2-fsa/sherpa-onnx/releases/download"
}

# --- 降噪:GTCRN,500 KB,真流式 ---
if (-not (Get-File "$REL/speech-enhancement-models/gtcrn_simple.onnx" (Join-Path $Models "gtcrn_simple.onnx"))) {
    Warn "GTCRN 下载失败,--denoise 会用不了"
}

# --- ASR:SenseVoice int8,中/英/日/韩/粤,约 250 MB ---
$AsrDir = ""
$asrName = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
if (Get-Tar "$REL/asr-models/$asrName.tar.bz2" $asrName) {
    $AsrDir = $asrName
} else {
    Warn "SenseVoice 下载失败,ASR 会用不了"
}

# --- VAD:Silero,640 KB ---
# 噪声环境下必须用它。纯能量 VAD 在底噪 RMS 0.02 的房间里会被钳死,一句话只切出一个字
if (-not (Get-File "$REL/asr-models/silero_vad.onnx" (Join-Path $Models "silero_vad.onnx"))) {
    Warn "Silero VAD 下载失败,只能退回 --vad energy(噪声环境下句子会被切碎)"
}

# --- 唤醒词:KWS zipformer,3.3 MB ---
# 常驻跑的就是它:说"你好小智"之前 SenseVoice 完全不动,待机 CPU 降一个数量级
$KwsDir = ""
$kwsName = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
if (Get-Tar "$REL/kws-models/$kwsName.tar.bz2" $kwsName) {
    $KwsDir = $kwsName
} else {
    Warn "唤醒词模型下载失败,--wake 会用不了(可以退回 --wake --wake-mode text)"
}

# --- TTS:Matcha 中文单音色,约 80 MB + 声码器 ---
# 为什么不用 Kokoro(音色多、听感更好):它在这类 CPU 上合成比实时还慢。
# 同一台 i7 上量的同一句"正常,":Kokoro 1.98 秒,Matcha 配 vocos 0.05 秒。
# 语音助手里合成慢是致命的 —— 句子越排越多,声音永远追不上文字。
$ttsName = "matcha-icefall-zh-baker"
$TtsDir = $ttsName
if (-not (Get-Tar "$REL/tts-models/$ttsName.tar.bz2" $ttsName)) {
    Warn "Matcha 下载失败,--speak 会用不了"
    $TtsDir = ""
}
# 声码器是单独一个文件,不在模型包里(多个 Matcha 模型共用它)
if (-not (Get-File "$REL/vocoder-models/vocos-22khz-univ.onnx" `
                   (Join-Path $Models "vocos-22khz-univ.onnx"))) {
    Warn "声码器 vocos 下载失败,Matcha 起不来(--speak 会用不了)"
}

# 想要 Kokoro 那 8 个中文音色(听感更好,但慢得多)就把下面这行的注释去掉,
# 再用 --tts-model models\kokoro-multi-lang-v1_0 指过去:
# Get-Tar "$REL/tts-models/kokoro-multi-lang-v1_0.tar.bz2" "kokoro-multi-lang-v1_0" | Out-Null


# --------------------------------------------------------------------------- #
Say "4/4  写配置"
# --------------------------------------------------------------------------- #
$envText = @"
# setup_windows.ps1 生成,realtime_asr.py 会读它当默认值。手改也行。
SHERPA_ASR_DIR=$AsrDir
SHERPA_DENOISE_MODEL=gtcrn_simple.onnx
SHERPA_TTS_DIR=$TtsDir
SHERPA_KWS_DIR=$KwsDir
SHERPA_VAD_MODEL=silero_vad.onnx
"@
# 一律写成不带 BOM 的 UTF-8:load_paths_env() 是按 utf-8 读的,PowerShell 5.1 的
# Out-File 默认会加 BOM,那三个字节会跟着第一行的 key 一起被读进去
[System.IO.File]::WriteAllText((Join-Path $Models "paths.env"), $envText, (New-Object System.Text.UTF8Encoding $false))
Write-Host "  已写 models\paths.env (asr=$(if ($AsrDir) { $AsrDir } else { '未安装' }))"


Write-Host @"

============================================================
装完了。LLM 还要单独起一个 server,见下面第 4 步。

按顺序自检,别一上来就跑整条链路 —— 出问题不好定位:

  1) 麦克风
     .venv\Scripts\python.exe realtime_asr.py --list-devices
     (同一个麦克风会出现好几次,优先挑名字里带 WASAPI 的)

  2) 降噪 + ASR(不联网、不用 LLM)
     .venv\Scripts\python.exe realtime_asr.py --denoise

  3) 唤醒词(对着麦克风说"你好小智")
     .venv\Scripts\python.exe wakeword.py

  4) TTS(单独念一句)
     .venv\Scripts\python.exe tts.py "你好,我是语音助手"

  5) LLM:另开一个 PowerShell 窗口起 server
     .\start_llm.ps1

     起好之后回到这个窗口:
     .venv\Scripts\python.exe realtime_asr.py --denoise --chat --speak

     想让它待机时不吃 CPU,加 --wake(说"你好小智"才开始识别):
     .venv\Scripts\python.exe realtime_asr.py --wake --denoise --chat --speak

嫌慢就把 server 换成 1.7B:.\start_llm.ps1 -Model 1.7b
============================================================
"@ -ForegroundColor Green
