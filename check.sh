#!/usr/bin/env bash
# 上传到 /root 后执行: bash /root/check.sh
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

fail=0
ok() { echo "  [通过] $*"; }
bad() { echo "  [缺失] $*"; fail=1; }
note() { echo "  [注意] $*"; }

echo "===== YP-RVC 环境自检 ====="

if [ -d /root/YP-RVC ]; then
  ROOT=/root/YP-RVC
elif [ -f "$(cd "$(dirname "$0")" && pwd)/YP-RVC/webui.py" ]; then
  ROOT="$(cd "$(dirname "$0")" && pwd)/YP-RVC"
else
  ROOT=/root/YP-RVC
fi
echo "项目目录: $ROOT"

echo
echo "-- 安装进程 --"
if ps -ef | grep -E '[d]eploy.sh|[p]ip install' >/dev/null 2>&1; then
  note "部署还在运行。缺的包可能仍在下载，等它结束再查一次。"
  ps -ef | grep -E '[d]eploy.sh|[p]ip install' | awk '{print "    "$0}'
else
  ok "当前没有部署或 pip 安装进程"
fi

echo
echo "-- 显卡 --"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | while IFS= read -r line; do
    ok "GPU $line"
  done
else
  bad "nvidia-smi 不可用，实例里没有可用的 NVIDIA 驱动"
fi

echo
echo "-- Python --"
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  if conda env list | awk '{print $1}' | grep -qx yprvc; then
    conda activate yprvc
    ok "使用 conda 环境 yprvc"
  else
    conda activate base
    note "还没有 yprvc 环境，当前用的是 base"
  fi
else
  note "未找到 /root/miniconda3，使用系统 python"
fi

if ! command -v python >/dev/null 2>&1; then
  bad "找不到 python"
  echo
  echo "结论: 环境未就绪"
  exit 1
fi

PY="$(command -v python)"
ver="$("$PY" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "  解释器: $PY"
echo "  版本: $ver"
case "$ver" in
  3.10|3.11) ok "Python 版本适合编译 fairseq" ;;
  *) bad "Python $ver 不适合 fairseq，需要 3.10 或 3.11" ;;
esac

echo
echo "-- 系统命令 --"
if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg $({ ffmpeg -version 2>/dev/null || true; } | awk 'NR==1{print $3}')"
else
  bad "没有 ffmpeg"
fi

echo
echo "-- Python 包 --"
"$PY" - <<'PY' || fail=1
import importlib
import sys

def line(flag, text):
    print("  [%s] %s" % (flag, text))

missing = 0
try:
    import torch
except Exception as exc:
    line("缺失", "torch 未安装 (%s)" % exc.__class__.__name__)
    missing = 1
else:
    line("通过", "torch %s" % torch.__version__)
    if torch.cuda.is_available():
        line("通过", "CUDA 可用: %s" % torch.cuda.get_device_name(0))
    else:
        line("缺失", "torch 已安装，但 CUDA 不可用")
        missing = 1

for name in ("fairseq", "gradio", "faiss", "pyworld", "parselmouth", "librosa", "soundfile", "numpy"):
    try:
        importlib.import_module(name)
        line("通过", name)
    except Exception:
        line("缺失", name)
        missing = 1
sys.exit(missing)
PY

echo
echo "-- 项目文件 --"
if [ ! -d "$ROOT" ]; then
  bad "项目目录不存在: $ROOT"
else
  for f in \
    webui.py \
    requirements.txt \
    start.sh \
    assets/hubert/hubert_base.pt \
    assets/rmvpe/rmvpe.pt \
    assets/pretrained_v2/f0G40k.pth \
    assets/pretrained_v2/f0D40k.pth \
    logs/mute/1_16k_wavs/mute.wav
  do
    if [ -f "$ROOT/$f" ]; then
      ok "$f"
    else
      bad "$f"
    fi
  done
fi

echo
echo "-- 训练页端口 --"
if command -v ss >/dev/null 2>&1 && ss -ltn | grep -q ':7860'; then
  ok "7860 已在监听，可以用应用端口地址打开"
elif command -v netstat >/dev/null 2>&1 && netstat -ltn | grep -q ':7860'; then
  ok "7860 已在监听，可以用应用端口地址打开"
else
  note "7860 还没监听。依赖齐了之后执行 bash /root/YP-RVC/cloud-start.sh"
fi

echo
if [ "$fail" -eq 0 ]; then
  echo "结论: 环境就绪，可以启动训练页。"
  exit 0
fi
echo "结论: 还有缺失项。若部署脚本仍在下载，等它结束后再运行一次本脚本。"
exit 1
