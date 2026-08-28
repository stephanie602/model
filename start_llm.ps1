# -*- coding: utf-8 -*-
<#
.SYNOPSIS
    在 Windows 上起 LLM server。realtime_asr.py --chat 要连的就是它。

.DESCRIPTION
    语音那三块(ASR / TTS / 降噪)的模型在 models\ 里,由 setup_windows.ps1 管;
    LLM 是完全独立的一个进程,用自己的 GGUF 文件,放在 llm-models\。
    两边只通过 HTTP 说话 —— realtime_asr.py 从来不碰 GGUF 文件。

    没有 llama-server.exe 的话会自动去 GitHub 下官方预编译包(免编译,解压即用)。

.EXAMPLE
    .\start_llm.ps1                    # 4B,平衡
    .\start_llm.ps1 -Model 1.7b        # 1.7B,明显更跟手,质量差一点
    .\start_llm.ps1 -Backend vulkan    # 有独显/核显就用它,比纯 CPU 快好几倍
    .\start_llm.ps1 -DownloadOnly      # 只下东西,不启动
    .\start_llm.ps1 -Port 8081         # 换端口(realtime_asr.py 要跟着加 --chat-url)
    .\start_llm.ps1 -CN                # 国内网络,模型走 hf-mirror

.NOTES
    这个窗口要一直开着。想让它后台常驻,用脚本末尾打印的计划任务方案。
#>

[CmdletBinding()]
param(
    [ValidateSet("4b", "1.7b")] [string]$Model = "4b",
    [ValidateSet("cpu", "vulkan")] [string]$Backend = "cpu",
    [int]$Port = 8080,
    [int]$Threads = 0,
    [int]$Ctx = 4096,
    [switch]$DownloadOnly,
    [switch]$CN,
    [string]$Tag = ""
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# 老版 Win10 上 PowerShell 5.1 默认还在用 TLS 1.0,GitHub 和 HuggingFace 早就不收了,
# 表现是 Invoke-RestMethod 报"基础连接已关闭",看不出跟 TLS 有关
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Here     = Split-Path -Parent $MyInvocation.MyCommand.Path
$LlamaDir = Join-Path $Here "llama-bin"
$LlmDir   = Join-Path $Here "llm-models"

if ($Model -eq "1.7b") {
    $ModelRepo = "unsloth/Qwen3-1.7B-GGUF"
    $ModelFile = "Qwen3-1.7B-Q4_K_M.gguf"
} else {
    $ModelRepo = "unsloth/Qwen3-4B-Instruct-2507-GGUF"
    $ModelFile = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
}

function Say  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Warn ($m) { Write-Host "[!] $m"   -ForegroundColor Yellow }
function Die  ($m) { Write-Host "[x] $m"   -ForegroundColor Red; exit 1 }

function Download($Url, $Out) {
    Write-Host "  $Url"
    $part = "$Out.part"
    try {
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            # Out-Null 挡住 stdout:这个函数靠返回值传成败,原生命令漏一个字符串
            # 出去就会和 $true/$false 一起进返回值数组。进度条走的是 stderr,照样看得到。
            & curl.exe -fL --retry 3 --connect-timeout 20 --progress-bar -o $part $Url | Out-Null
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


# --------------------------------------------------------------------------- #
Say "1/4  找 llama-server.exe"
# --------------------------------------------------------------------------- #
# 预编译包解压出来是一层平铺的目录,自己用 CMake 编的在 build\bin\,
# 所以递归搜而不是写死路径
$LlamaBin = $null
if (Test-Path $LlamaDir) {
    $found = Get-ChildItem -Path $LlamaDir -Filter "llama-server.exe" -Recurse -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if ($found) { $LlamaBin = $found.FullName }
}

if (-not $LlamaBin) {
    Say "  没找到,去 GitHub 下官方预编译包($Backend)"
    New-Item -ItemType Directory -Force -Path $LlamaDir | Out-Null

    if (-not $Tag) {
        # 问一下最新的 release tag,免得把版本号写死在脚本里过几个月就失效
        try {
            $rel = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" `
                                     -UseBasicParsing -Headers @{ "User-Agent" = "start_llm.ps1" }
            $Tag = $rel.tag_name
        } catch {
            Warn "  问不到最新版本($_),用一个已知可用的版本"
            $Tag = "b6100"
        }
    }
    Write-Host "  版本: $Tag"

    # 资产命名:llama-<tag>-bin-win-<backend>-x64.zip
    #   cpu    —— 到处都能跑,Qwen3-4B 在 8 核桌面 CPU 上大约 8-15 tok/s
    #   vulkan —— 只要有能用的显卡(独显、核显都行)就比 cpu 快好几倍,
    #             而且不挑厂商,不用装 CUDA
    $arch = if ($env:PROCESSOR_ARCHITECTURE -eq "ARM64") { "arm64" } else { "x64" }
    $zipName = "llama-$Tag-bin-win-$Backend-$arch.zip"
    $zipPath = Join-Path $LlamaDir $zipName
    $url = "https://github.com/ggml-org/llama.cpp/releases/download/$Tag/$zipName"
    if ($CN) { $url = "https://ghproxy.net/$url" }

    if (-not (Download $url $zipPath)) {
        Die "预编译包下不动。手动去 https://github.com/ggml-org/llama.cpp/releases 下 $zipName,`n    解压到 $LlamaDir 里,再重跑这个脚本。"
    }
    Write-Host "  解压 ..."
    Expand-Archive -Path $zipPath -DestinationPath (Join-Path $LlamaDir $Tag) -Force
    Remove-Item -Force $zipPath

    $found = Get-ChildItem -Path $LlamaDir -Filter "llama-server.exe" -Recurse -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if (-not $found) { Die "解压完还是没找到 llama-server.exe,看看 $LlamaDir 里是什么" }
    $LlamaBin = $found.FullName
}
Write-Host "  $LlamaBin"

# llama.cpp 现在是拆成多个 DLL 的(ggml-base.dll / ggml-cpu.dll / llama.dll ...)。
# Windows 找 DLL 是先看 exe 所在目录,预编译包里它们本来就在一起,所以正常不用管;
# 真缺了会弹一个 "0xc000007b" 之类的框,那基本就是解压不完整或者 32/64 位混了。
#
# 这里必须临时把 EAP 放回 Continue:llama-server --version 是往 stderr 打的,
# 而 EAP=Stop 时 PowerShell 会把原生命令的 stderr 当成终止性错误抛出来,
# 于是"版本号打印正常"反而变成了脚本崩溃。
$prevEAP = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $LlamaBin --version *> $null
$rc = $LASTEXITCODE
$ErrorActionPreference = $prevEAP
if ($rc -ne 0) {
    Warn "llama-server.exe 跑不起来。常见原因:"
    Warn "  1. 解压不完整 —— .dll 必须和 llama-server.exe 在同一个目录"
    Warn "  2. 缺 VC++ 运行库 —— 装一下 https://aka.ms/vs/17/release/vc_redist.x64.exe"
    Warn "  3. -Backend vulkan 但显卡驱动太老 —— 换回 -Backend cpu 先跑通"
    exit 1
}
Write-Host "  可执行 OK"


# --------------------------------------------------------------------------- #
Say "2/4  模型 ($Model)"
# --------------------------------------------------------------------------- #
New-Item -ItemType Directory -Force -Path $LlmDir | Out-Null
$Gguf = Get-ChildItem -Path $LlmDir -Filter "*.gguf" -ErrorAction SilentlyContinue |
        Select-Object -First 1

if (-not $Gguf) {
    $size = if ($Model -eq "1.7b") { "约 1.1 GB" } else { "约 2.5 GB" }
    Write-Host "  还没有,开始下载($size)"
    # HuggingFace 的 resolve/main/<文件名> 是稳定直链,不需要 huggingface_hub,
    # 少一层依赖就少一个卡住的地方
    $host_ = if ($CN) { "https://hf-mirror.com" } else { "https://huggingface.co" }
    $url = "$host_/$ModelRepo/resolve/main/$ModelFile"
    if (-not (Download $url (Join-Path $LlmDir $ModelFile))) {
        Die "模型下不动。手动下载后放进 $LlmDir 再重跑:`n    $url"
    }
    $Gguf = Get-Item (Join-Path $LlmDir $ModelFile)
}
Write-Host "  $($Gguf.FullName)  ($([math]::Round($Gguf.Length / 1GB, 2)) GB)"

if ($DownloadOnly) {
    Say "-DownloadOnly,不启动"
    exit 0
}


# --------------------------------------------------------------------------- #
Say "3/4  检查端口"
# --------------------------------------------------------------------------- #
$busy = $false
try {
    $client = New-Object System.Net.Sockets.TcpClient
    # 用异步 + 超时,不然端口没人监听时在某些网络配置下会卡好几秒
    $ok = $client.ConnectAsync("127.0.0.1", $Port).Wait(500)
    if ($ok -and $client.Connected) { $busy = $true }
    $client.Close()
} catch { }

if ($busy) {
    Warn "端口 $Port 上已经有东西在监听了。"
    Warn "如果是之前起的 server,不用重复启动;要重启的话先:"
    Warn "  Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process"
    exit 0
}
Write-Host "  $Port 空着 OK"


# --------------------------------------------------------------------------- #
Say "4/4  启动"
# --------------------------------------------------------------------------- #
if ($Threads -le 0) {
    # 按物理核算,不按逻辑核 —— llama.cpp 在超线程上开满线程反而更慢。
    # 再留 2 个核给 ASR 和 TTS,它们和这个进程是抢同一批 CPU 的。
    $cores = 0
    try {
        $cores = (Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum
    } catch { }
    if (-not $cores -or $cores -le 0) { $cores = [int]$env:NUMBER_OF_PROCESSORS }
    $Threads = [Math]::Max(1, $cores - 2)
    Write-Host "  $cores 个物理核,用 $Threads 线程(留 2 个给 ASR/TTS)"
}

$chatUrlNote = if ($Port -ne 8080) { " --chat-url http://127.0.0.1:$Port/v1" } else { "" }
Write-Host @"

  起来之后另开一个 PowerShell 窗口验证:
      curl.exe http://127.0.0.1:$Port/v1/models

  然后接上语音链路:
      cd "$Here"
      .venv\Scripts\python.exe realtime_asr.py --denoise --chat --speak$chatUrlNote

  这个窗口保持开着。想做成开机自启(当前用户登录时后台运行):
      `$a = New-ScheduledTaskAction -Execute "$LlamaBin" ``
           -Argument '-m "$($Gguf.FullName)" -c $Ctx -t $Threads --host 127.0.0.1 --port $Port'
      `$t = New-ScheduledTaskTrigger -AtLogOn
      Register-ScheduledTask -TaskName llama-server -Action `$a -Trigger `$t

"@

& $LlamaBin `
    -m $Gguf.FullName `
    -c $Ctx `
    -t $Threads `
    --host 127.0.0.1 `
    --port $Port
