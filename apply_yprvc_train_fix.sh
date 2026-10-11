#!/usr/bin/env bash
# 有品 WebUI 训练修复：解压补丁覆盖代码并尽量重启训练页。
#
# 用法（在云端）：
#   bash apply_yprvc_train_fix.sh /root/yprvc-train-ui-fix.zip
#   bash apply_yprvc_train_fix.sh
#
# 环境变量：
#   ROOT=/root/YP-RVC   项目根目录（默认）

set -euo pipefail

ROOT="${ROOT:-/root/YP-RVC}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ZIP_PATH="${1:-}"
if [[ -z "$ZIP_PATH" ]]; then
  if [[ -f "$SCRIPT_DIR/yprvc-train-ui-fix.zip" ]]; then
    ZIP_PATH="$SCRIPT_DIR/yprvc-train-ui-fix.zip"
  elif [[ -f "/root/yprvc-train-ui-fix.zip" ]]; then
    ZIP_PATH="/root/yprvc-train-ui-fix.zip"
  elif [[ -f "./yprvc-train-ui-fix.zip" ]]; then
    ZIP_PATH="./yprvc-train-ui-fix.zip"
  else
    echo "用法: bash apply_yprvc_train_fix.sh /path/to/yprvc-train-ui-fix.zip" >&2
    exit 1
  fi
fi

ZIP_PATH="$(readlink -f "$ZIP_PATH" 2>/dev/null || realpath "$ZIP_PATH" 2>/dev/null || echo "$ZIP_PATH")"

if [[ ! -f "$ZIP_PATH" ]]; then
  echo "找不到补丁包: $ZIP_PATH" >&2
  exit 1
fi
if [[ ! -d "$ROOT" ]]; then
  echo "项目目录不存在: $ROOT" >&2
  exit 1
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
BACKUP="$ROOT/logs/fix-backup-$STAMP"
mkdir -p "$BACKUP"

FILES=(
  "webui.py"
  "webui.css"
  "restart_webui.sh"
  "infer/modules/train/train.py"
  "infer/lib/train/process_ckpt.py"
)
# 可选附带脚本（在 zip 内则一并更新到 /root）
ROOT_SCRIPTS=(
  "apply_yprvc_train_fix.sh"
  "run_yprvc_fix.sh"
)

echo "==> 项目: $ROOT"
echo "==> 补丁: $ZIP_PATH"
echo "==> 备份: $BACKUP"

for rel in "${FILES[@]}"; do
  src="$ROOT/$rel"
  if [[ -f "$src" ]]; then
    mkdir -p "$BACKUP/$(dirname "$rel")"
    cp -a "$src" "$BACKUP/$rel"
    echo "  已备份 $rel"
  else
    echo "  警告: 原文件不存在，跳过备份 $rel"
  fi
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 用 Python 解压，兼容 Windows 反斜杠 zip；不要用会因 warning 失败的 unzip
python3 - "$ZIP_PATH" "$TMP" <<'PY'
import sys, zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
dest = Path(sys.argv[2])
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        name = info.filename.replace("\\", "/").lstrip("/")
        if not name or name.endswith("/"):
            continue
        target = dest / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(info) as src, open(target, "wb") as out:
            out.write(src.read())
        print("  解压", name)
PY

SRC_ROOT="$TMP"
if [[ ! -f "$TMP/webui.py" ]]; then
  for candidate in "$TMP"/*; do
    if [[ -f "$candidate/webui.py" ]]; then
      SRC_ROOT="$candidate"
      break
    fi
  done
fi

if [[ ! -f "$SRC_ROOT/webui.py" ]]; then
  echo "补丁包内容无效：缺少 webui.py" >&2
  echo "解压结果：" >&2
  find "$TMP" -type f >&2 || true
  exit 1
fi

for rel in "${FILES[@]}"; do
  if [[ ! -f "$SRC_ROOT/$rel" ]]; then
    echo "补丁包缺少: $rel" >&2
    find "$TMP" -type f >&2 || true
    exit 1
  fi
  mkdir -p "$ROOT/$(dirname "$rel")"
  cp -a "$SRC_ROOT/$rel" "$ROOT/$rel"
  echo "  已覆盖 $rel"
done

for rel in "${ROOT_SCRIPTS[@]}"; do
  if [[ -f "$SRC_ROOT/$rel" ]]; then
    cp -a "$SRC_ROOT/$rel" "/root/$rel"
    # 统一 LF，避免从 Windows 拷过来后 bash 异常
    python3 - "$rel" <<'PY' || true
import pathlib, sys
p = pathlib.Path("/root") / sys.argv[1]
if p.is_file():
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
PY
    chmod +x "/root/$rel" || true
    echo "  已同步 /root/$rel"
  fi
done

# 快速自检（失败则 exit 1，避免“以为更新了其实没有”）
if ! grep -q 'dataset-upload-ui?v=6' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 dataset-upload-ui?v=6，补丁可能未生效" >&2
  exit 1
fi
if ! grep -q 'train-status-ui?v=6' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 train-status-ui?v=6（任务跟随）" >&2
  exit 1
fi
if ! grep -q 'job-status' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 job-status（任务状态）" >&2
  exit 1
fi
if ! grep -q 'STRICT_PORT' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 STRICT_PORT（禁止飘到 7861）" >&2
  exit 1
fi
if ! grep -q 'restore_all_ui' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 restore_all_ui（全量回显）" >&2
  exit 1
fi
if ! grep -q 'ui-session' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 ui-session（模型名会话）" >&2
  exit 1
fi
if ! grep -q 'logs-ui' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 logs-ui（独立日志页）" >&2
  exit 1
fi
if ! grep -q 'dataset-upload-log' "$ROOT/webui.py"; then
  echo "自检失败：webui.py 里没有 dataset-upload-log（上传行为日志）" >&2
  exit 1
fi
if ! grep -qE 'port80_proxy|绕过 boot-start' "$ROOT/restart_webui.sh"; then
  echo "自检失败：restart_webui.sh 不是预期版本" >&2
  exit 1
fi
if ! grep -q 'STRICT_PORT' "$ROOT/restart_webui.sh"; then
  echo "自检失败：restart_webui.sh 没有 STRICT_PORT（防 7861）" >&2
  exit 1
fi
chmod +x "$ROOT/restart_webui.sh" || true
cp -a "$ROOT/restart_webui.sh" /root/restart_webui.sh
echo "  已同步 /root/restart_webui.sh"
echo "==> 自检通过：代码与重启脚本已写入"

mkdir -p "$ROOT/logs" "$ROOT/assets/weights" "$ROOT/logs/weights"
echo "==> 重启 WebUI"
bash "$ROOT/restart_webui.sh" || {
  echo "自动重启未成功，请手动执行: bash $ROOT/restart_webui.sh" >&2
}

echo
echo "修复完成。"
echo "  备份目录: $BACKUP"
echo "  以后只重启可执行: bash /root/restart_webui.sh"
echo "  请强制刷新浏览器（Ctrl+F5）验证。"
