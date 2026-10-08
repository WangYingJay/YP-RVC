#!/usr/bin/env bash
# 放在 YP-RVC-cloud.tar 旁边，在仙宫云终端执行: bash /root/deploy.sh
set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PYTHONIOENCODING=utf-8

trap 'echo; echo "部署中断。请把这段终端输出发回来。"; exit 1' ERR

BASE="$(cd "$(dirname "$0")" && pwd)"
TAR="$BASE/YP-RVC-cloud.tar"
ROOT="$BASE/YP-RVC"

echo "===== YP-RVC 一键部署 ====="
echo "脚本目录: $BASE"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
fi

ver="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
echo "当前 Python: $(command -v python) $ver"
case "$ver" in
  3.10|3.11) ;;
  *)
    echo "Python $ver 不适合编译 fairseq，正在创建 Python 3.11 环境 yprvc"
    export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
    if conda env list | awk '{print $1}' | grep -qx yprvc; then
      echo "环境 yprvc 已存在，直接使用。"
    else
      conda create -y -n yprvc python=3.11
    fi
    conda activate yprvc
    ;;
esac

PY="$(command -v python)"
echo "将使用: $PY"
"$PY" -V

echo "===== 1/4 解压 ====="
if [ -f "$ROOT/webui.py" ] && [ -f "$ROOT/assets/hubert/hubert_base.pt" ] && [ -f "$ROOT/assets/rmvpe/rmvpe.pt" ]; then
  echo "已经解压过，跳过。目录: $ROOT"
else
  if [ ! -f "$TAR" ]; then
    echo "找不到压缩包: $TAR"
    echo "请把 deploy.sh 和 YP-RVC-cloud.tar 放在同一个目录后再运行。"
    exit 1
  fi
  mkdir -p "$ROOT"
  echo "正在解压，大约几分钟，请不要关闭窗口。"
  tar -xf "$TAR" -C "$ROOT"
fi

for f in \
  webui.py \
  requirements.txt \
  tools/install_fairseq.sh \
  assets/hubert/hubert_base.pt \
  assets/rmvpe/rmvpe.pt \
  assets/pretrained_v2/f0G40k.pth \
  assets/pretrained_v2/f0D40k.pth \
  logs/mute/1_16k_wavs/mute.wav
do
  if [ ! -f "$ROOT/$f" ]; then
    echo "解压结果不完整，缺少 $f"
    exit 1
  fi
done

echo "===== 2/4 系统依赖 ====="
apt-get update || true
apt-get install -y ffmpeg libsndfile1 build-essential python3-dev

echo "===== 3/4 Python 依赖 ====="
# setuptools 82 起移除了 pkg_resources，pyworld 0.3.4 导入时还依赖它。
"$PY" -m pip install -U "pip==24.0" "setuptools>=65,<82" wheel

if "$PY" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "CUDA 版 PyTorch 已可用，跳过安装。"
else
  echo "正在安装 CUDA 版 PyTorch。cudnn 大约 700MB，请等待，不要中断。"
  "$PY" -m pip uninstall -y torch torchvision torchaudio || true
  "$PY" -m pip cache purge || true
  ok=0
  # 一次只用一个源。两个源混用时，文件和校验和会对不上，然后整包重下。
  for mirror in \
    "https://pypi.tuna.tsinghua.edu.cn/simple" \
    "https://mirrors.aliyun.com/pypi/simple" \
    "https://pypi.org/simple"
  do
    echo "从 ${mirror} 安装 PyTorch"
    if "$PY" -m pip install --timeout 600 --no-cache-dir torch==2.4.1 --index-url "$mirror"; then
      ok=1
      break
    fi
    echo "这个源校验失败，换下一个源。不会用同一个源再下三遍。"
    "$PY" -m pip cache purge || true
  done
  if [ "$ok" -eq 0 ]; then
    echo "三个源都没有装上 PyTorch。"
    exit 1
  fi
fi
"$PY" -m pip install "numpy>=1.23,<2"
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无可用 GPU')"
if ! "$PY" -c "import torch; assert torch.cuda.is_available()"; then
  echo "PyTorch 已安装，但 CUDA 不可用。不要继续用 CPU 训练。"
  exit 1
fi

"$PY" - <<'PY' || true
import re
import site
from pathlib import Path

pattern = re.compile(r"^(\s+)(\w+)\s*:\s*([A-Za-z_]\w*)\s*=\s*\3\(\)\s*$", re.M)

def repl(match):
    indent, name, typ = match.group(1), match.group(2), match.group(3)
    return f"{indent}{name}: {typ} = field(default_factory={typ})"

factory_pattern = re.compile(
    r"field\(\s*default\s*=\s*([A-Za-z_][\w.]*)\(\s*\)(\s*,)?"
)

def factory_repl(match):
    typ = match.group(1)
    comma = match.group(2) or ""
    return f"field(default_factory={typ}{comma}"

total = 0
for root in site.getsitepackages():
    base = Path(root)
    for pkg in ("fairseq", "hydra"):
        folder = base / pkg
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if not re.search(r"from dataclasses import [^\n]*\bfield\b", text):
                continue
            patched, count = pattern.subn(repl, text)
            patched, count2 = factory_pattern.subn(factory_repl, patched)
            count += count2
            if count:
                path.write_text(patched, encoding="utf-8")
                print("已修补", path.name, count, "处")
                total += count
init_py = None
for root in site.getsitepackages():
    path = Path(root) / "fairseq" / "dataclass" / "initialize.py"
    if path.is_file():
        init_py = path
        break
if init_py is not None:
    init_text = init_py.read_text(encoding="utf-8")
    found = re.search(
        r"^([ \t]*)v = FairseqConfig\.__dataclass_fields__\[k\]\.default\s*$",
        init_text,
        re.M,
    )
    if found and "_field.default_factory" not in init_text:
        indent = found.group(1)
        init_text = init_text.replace(
            found.group(0),
            "\n".join(
                [
                    indent + "_field = FairseqConfig.__dataclass_fields__[k]",
                    indent + "v = _field.default",
                    indent + "if v is MISSING and _field.default_factory is not MISSING:",
                    indent + "    v = _field.default_factory()",
                ]
            ),
            1,
        )
        if "from dataclasses import MISSING" not in init_text:
            init_text = init_text.replace(
                "import logging\n",
                "import logging\nfrom dataclasses import MISSING\n",
                1,
            )
        init_py.write_text(init_text, encoding="utf-8")
        print("已修补 fairseq 启动逻辑")
print("Python 3.11 兼容修补完成，共", total, "处")
PY

if "$PY" -c "import fairseq,gradio,faiss,pyworld,parselmouth,librosa,soundfile" 2>/dev/null; then
  echo "其余依赖已装好，跳过。"
else
  echo "正在编译 fairseq。官方包缺一个用不到的 C++ 文件，脚本会跳过它。"
  cd "$ROOT"
  "$PY" - <<'PY'
import pathlib
import shutil
import site

src = pathlib.Path("tools/rvc_py311_compat.py")
site_dir = pathlib.Path(site.getsitepackages()[0])
shutil.copy(src, site_dir / "rvc_py311_compat.py")
(site_dir / "rvc_py311_compat.pth").write_text("import rvc_py311_compat\n", encoding="utf-8")
print("installed collections compat into", site_dir)
PY
  "$PY" -m pip install --no-cache-dir "cython" "torchaudio==2.4.1" "numpy>=1.23,<2"
  rm -rf /tmp/fairseq-build
  mkdir -p /tmp/fairseq-build
  curl -fL --retry 3 -o /tmp/fairseq-build/fairseq-0.12.2.tar.gz \
    "https://pypi.tuna.tsinghua.edu.cn/packages/30/36/db42846570f479ac859be3b48d8b47f2ae9b0b9c77487a512f2f2ecbcb6b/fairseq-0.12.2.tar.gz" \
    || curl -fL --retry 3 -o /tmp/fairseq-build/fairseq-0.12.2.tar.gz \
    "https://files.pythonhosted.org/packages/30/36/db42846570f479ac859be3b48d8b47f2ae9b0b9c77487a512f2f2ecbcb6b/fairseq-0.12.2.tar.gz"
  tar -xzf /tmp/fairseq-build/fairseq-0.12.2.tar.gz -C /tmp/fairseq-build
  "$PY" - <<'PY'
from pathlib import Path

setup_py = Path("/tmp/fairseq-build/fairseq-0.12.2/setup.py")
text = setup_py.read_text(encoding="utf-8")
needle = "def do_setup(package_data):\n"
insert = """def do_setup(package_data):
    global extensions
    kept = []
    for ext in extensions:
        missing = [src for src in ext.sources if not os.path.exists(src)]
        if missing:
            print("skip fairseq extension", ext.name, missing)
            continue
        kept.append(ext)
    extensions = kept
"""
if needle not in text:
    raise SystemExit("无法修改 fairseq 的 setup.py")
setup_py.write_text(text.replace(needle, insert, 1), encoding="utf-8")
print("已跳过源码包里缺失的 C++ 扩展")
PY
  "$PY" -m pip install --no-cache-dir --no-build-isolation /tmp/fairseq-build/fairseq-0.12.2
  "$PY" - <<'PY'
import re
import site
from pathlib import Path

pattern = re.compile(r"^(\s+)(\w+)\s*:\s*([A-Za-z_]\w*)\s*=\s*\3\(\)\s*$", re.M)

def repl(match):
    indent, name, typ = match.group(1), match.group(2), match.group(3)
    return f"{indent}{name}: {typ} = field(default_factory={typ})"

factory_pattern = re.compile(
    r"field\(\s*default\s*=\s*([A-Za-z_][\w.]*)\(\s*\)(\s*,)?"
)

def factory_repl(match):
    typ = match.group(1)
    comma = match.group(2) or ""
    return f"field(default_factory={typ}{comma}"

total = 0
for root in site.getsitepackages():
    base = Path(root)
    for pkg in ("fairseq", "hydra"):
        folder = base / pkg
        if not folder.is_dir():
            continue
        for path in folder.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if not re.search(r"from dataclasses import [^\n]*\bfield\b", text):
                continue
            patched, count = pattern.subn(repl, text)
            patched, count2 = factory_pattern.subn(factory_repl, patched)
            count += count2
            if count:
                path.write_text(patched, encoding="utf-8")
                print("已修补", path.name, count, "处")
                total += count
init_py = None
for root in site.getsitepackages():
    path = Path(root) / "fairseq" / "dataclass" / "initialize.py"
    if path.is_file():
        init_py = path
        break
if init_py is not None:
    init_text = init_py.read_text(encoding="utf-8")
    found = re.search(
        r"^([ \t]*)v = FairseqConfig\.__dataclass_fields__\[k\]\.default\s*$",
        init_text,
        re.M,
    )
    if found and "_field.default_factory" not in init_text:
        indent = found.group(1)
        init_text = init_text.replace(
            found.group(0),
            "\n".join(
                [
                    indent + "_field = FairseqConfig.__dataclass_fields__[k]",
                    indent + "v = _field.default",
                    indent + "if v is MISSING and _field.default_factory is not MISSING:",
                    indent + "    v = _field.default_factory()",
                ]
            ),
            1,
        )
        if "from dataclasses import MISSING" not in init_text:
            init_text = init_text.replace(
                "import logging\n",
                "import logging\nfrom dataclasses import MISSING\n",
                1,
            )
        init_py.write_text(init_text, encoding="utf-8")
        print("已修补 fairseq 启动逻辑")
print("Python 3.11 兼容修补完成，共", total, "处")
PY
  grep -v '^fairseq==' requirements.txt > /tmp/requirements.nofairseq.txt
  "$PY" -m pip install -r /tmp/requirements.nofairseq.txt
fi
"$PY" -c "import fairseq,gradio,faiss,pyworld,parselmouth,librosa,soundfile; print('依赖检查通过')"

echo "===== 4/4 启动训练页 ====="
export PORT=7860
export RVC_DATASET_DIR="$ROOT/datasets"
export RVC_LOG_DIR="$ROOT/logs"
mkdir -p "$RVC_DATASET_DIR" "$RVC_LOG_DIR" "$RVC_LOG_DIR/weights" "$ROOT/assets/weights"

cat > "$ROOT/cloud-start.sh" << EOF
#!/usr/bin/env bash
cd "$ROOT"
export PORT=7860
export PYTHONIOENCODING=utf-8
export RVC_DATASET_DIR="$ROOT/datasets"
export RVC_LOG_DIR="$ROOT/logs"
exec "$PY" webui.py
EOF
chmod +x "$ROOT/cloud-start.sh"

echo "训练页监听 0.0.0.0:7860"
echo "请确认仙宫云开放端口里已经添加 7860，再用控制台给出的应用端口地址打开。"
echo "这个窗口要保持开着。下次只需执行: bash $ROOT/cloud-start.sh"
cd "$ROOT"
exec "$PY" webui.py
