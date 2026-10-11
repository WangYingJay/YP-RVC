#!/usr/bin/env bash
# 只把 yprvc-train-ui-fix.zip 拷到 /root 后，执行本脚本即可
#（会先从 zip 里抽出最新 apply 脚本，再覆盖代码并重启）
#
#   bash /root/run_yprvc_fix.sh
#   bash /root/run_yprvc_fix.sh /root/yprvc-train-ui-fix.zip
#
set -euo pipefail
ZIP="${1:-/root/yprvc-train-ui-fix.zip}"
if [[ ! -f "$ZIP" ]]; then
  echo "找不到补丁包: $ZIP" >&2
  exit 1
fi

python3 - "$ZIP" <<'PY'
import sys
import zipfile
from pathlib import Path

zpath = Path(sys.argv[1])
with zipfile.ZipFile(zpath) as zf:
    names = {n.replace("\\", "/") for n in zf.namelist()}
    if "apply_yprvc_train_fix.sh" not in names:
        raise SystemExit("zip 内缺少 apply_yprvc_train_fix.sh，请使用新版补丁包")
    data = zf.read("apply_yprvc_train_fix.sh")
    # zip 里可能是 CRLF，统一成 LF
    text = data.replace(b"\r\n", b"\n")
    Path("/root/apply_yprvc_train_fix.sh").write_bytes(text)
    if "run_yprvc_fix.sh" in names:
        Path("/root/run_yprvc_fix.sh").write_bytes(
            zf.read("run_yprvc_fix.sh").replace(b"\r\n", b"\n")
        )
print("已从 zip 更新 /root/apply_yprvc_train_fix.sh")
PY

chmod +x /root/apply_yprvc_train_fix.sh /root/run_yprvc_fix.sh 2>/dev/null || true
bash /root/apply_yprvc_train_fix.sh "$ZIP"
