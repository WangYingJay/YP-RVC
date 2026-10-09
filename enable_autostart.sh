#!/usr/bin/env bash
# 上传到 /root 后执行: bash /root/enable_autostart.sh
# 登记开机自启动，并立刻拉起训练页。重启实例后会自己再启动。
set -eo pipefail

ROOT="/root/YP-RVC"
if [ ! -f "$ROOT/webui.py" ] && [ -f "$(cd "$(dirname "$0")" && pwd)/YP-RVC/webui.py" ]; then
  ROOT="$(cd "$(dirname "$0")" && pwd)/YP-RVC"
fi
if [ ! -f "$ROOT/webui.py" ]; then
  echo "找不到训练页: $ROOT/webui.py"
  exit 1
fi

echo "===== 登记开机自启动 ====="
echo "脚本版本: 6"
echo "项目目录: $ROOT"
pid1="$(tr '\0' ' ' < /proc/1/cmdline || true)"
echo "当前 1 号进程: ${pid1:-未知}"

DATASET="${RVC_DATASET_DIR:-$ROOT/datasets}"
LOGDIR="${RVC_LOG_DIR:-$ROOT/logs}"
PORT="${PORT:-7860}"
PY=""

web_pid="$(ps -ww -eo pid=,args= | awk '/python/ && /webui\.py/ {print $1; exit}')"
if [ -n "$web_pid" ] && [ -r "/proc/$web_pid/exe" ]; then
  PY="$(readlink -f "/proc/$web_pid/exe" || true)"
fi
if [ -z "$PY" ] && [ -f "$ROOT/cloud-start.sh" ]; then
  PY="$(awk '/^exec / {print $2; exit}' "$ROOT/cloud-start.sh" || true)"
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  if [ -x /usr/local/miniconda3/envs/yprvc/bin/python ]; then
    PY="/usr/local/miniconda3/envs/yprvc/bin/python"
  elif [ -x /root/miniconda3/envs/yprvc/bin/python ]; then
    PY="/root/miniconda3/envs/yprvc/bin/python"
  else
    CONDA_SH=""
    for candidate in \
      /usr/local/miniconda3/etc/profile.d/conda.sh \
      /root/miniconda3/etc/profile.d/conda.sh \
      /opt/conda/etc/profile.d/conda.sh
    do
      if [ -f "$candidate" ]; then
        CONDA_SH="$candidate"
        break
      fi
    done
    if [ -n "$CONDA_SH" ]; then
      # shellcheck disable=SC1090
      source "$CONDA_SH"
      CONDA_ROOT="$(cd "$(dirname "$CONDA_SH")/../.." && pwd)"
      if [ -x "$CONDA_ROOT/envs/yprvc/bin/python" ]; then
        PY="$CONDA_ROOT/envs/yprvc/bin/python"
      else
        PY="$(command -v python || true)"
      fi
    else
      PY="$(command -v python || true)"
    fi
  fi
fi
if [ -z "$PY" ] || [ ! -x "$PY" ]; then
  echo "找不到可用的 Python。"
  exit 1
fi
echo "将使用: $PY"

if [ -n "$web_pid" ] && [ -r "/proc/$web_pid/environ" ]; then
  eval "$("$PY" - "$web_pid" <<'PY'
import sys
pid = sys.argv[1]
wanted = {"RVC_DATASET_DIR", "RVC_LOG_DIR", "PORT"}
raw = open("/proc/%s/environ" % pid, "rb").read().split(b"\0")
for item in raw:
    if b"=" not in item:
        continue
    key, val = item.split(b"=", 1)
    name = key.decode("utf-8", "replace")
    if name not in wanted:
        continue
    text = val.decode("utf-8", "replace").replace("'", "'\\''")
    print("%s='%s'" % (name, text))
PY
)"
  DATASET="${RVC_DATASET_DIR:-$DATASET}"
  LOGDIR="${RVC_LOG_DIR:-$LOGDIR}"
  PORT="${PORT:-7860}"
fi

mkdir -p "$DATASET" "$LOGDIR" "$LOGDIR/weights" "$ROOT/assets/weights" "$ROOT/logs"

cat > "$ROOT/boot-start.sh" << EOF
#!/bin/bash
PY='$PY'
ROOT='$ROOT'
DATASET='$DATASET'
LOGDIR='$LOGDIR'
PORT='${PORT}'
LOG="\$ROOT/logs/boot.log"
APPLOG="\$ROOT/logs/webui.log"
mkdir -p "\$DATASET" "\$LOGDIR" "\$LOGDIR/weights" "\$ROOT/assets/weights" "\$ROOT/logs"

refresh_train_link() {
  export PORT="\$PORT"
  export RVC_ROOT="\$ROOT"
  if [ -f "\$ROOT/tools/refresh_train_link.py" ]; then
    "\$PY" "\$ROOT/tools/refresh_train_link.py" >> "\$LOG" 2>&1 || \\
      python3 "\$ROOT/tools/refresh_train_link.py" >> "\$LOG" 2>&1 || true
  fi
}

refresh_train_link

say() {
  local line
  line="\$(date '+%F %T') \$*"
  echo "\$line"
  if [ -t 1 ]; then
    echo "\$line" >> "\$LOG"
  fi
}

follow_log() {
  say "\$1"
  say "日志文件: \$APPLOG"
  if [ -f "\$APPLOG" ]; then
    tail -n 40 -f "\$APPLOG"
  fi
  sleep infinity
}

listening() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn | grep -q ":\$PORT "
    return
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -ltn | grep -q ":\$PORT "
    return
  fi
  return 1
}

start_server() {
  export PORT="\$PORT"
  export PYTHONUNBUFFERED=1
  export PYTHONIOENCODING=utf-8
  export TZ="\${TZ:-Asia/Shanghai}"
  export RVC_DATASET_DIR="\$DATASET"
  export RVC_LOG_DIR="\$LOGDIR"
  cd "\$ROOT"
  if [ ! -x "\$PY" ]; then
    say "找不到 Python: \$PY"
    exit 1
  fi
  say "启动训练页 \$PY  端口 \$PORT"
  if [ "\${1:-}" = "foreground" ]; then
    "\$PY" webui.py 2>&1 | tee -a "\$APPLOG"
    status=\${PIPESTATUS[0]}
    say "训练页退出，代码 \$status"
    exit "\$status"
  fi
  exec "\$PY" webui.py >> "\$APPLOG" 2>&1
}

lock_or_wait() {
  if ! command -v flock >/dev/null 2>&1; then
    return 0
  fi
  exec 9>/tmp/yprvc-start.lock
  if ! flock -n 9; then
    follow_log "已有启动过程，跟随现有日志"
  fi
}

start_web_proxy() {
  if ss -ltn 2>/dev/null | grep -q ':80 '; then
    return 0
  fi
  if [ ! -f "\$ROOT/port80_proxy.py" ]; then
    return 0
  fi
  nohup python3 "\$ROOT/port80_proxy.py" >> "\$ROOT/logs/port80.log" 2>&1 &
}

start_web_proxy

if [ "\${1:-}" = "foreground" ]; then
  say "前台启动，输出会显示在这个终端，并写入 \$APPLOG"
  lock_or_wait
  if listening; then
    follow_log "训练页已在 0.0.0.0:\$PORT 监听"
  fi
  start_server foreground
fi

if listening; then
  exit 0
fi
(
  if command -v flock >/dev/null 2>&1; then
    exec 9>/tmp/yprvc-start.lock
    flock -n 9 || exit 0
  fi
  if listening; then
    exit 0
  fi
  ready=0
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    if "\$PY" -c "import torch" >/dev/null 2>&1; then
      ready=1
      break
    fi
    sleep 2
  done
  if [ "\$ready" -ne 1 ]; then
    echo "\$(date '+%F %T') Python 还不能导入 torch，仍尝试启动" >> "\$LOG"
  fi
  if listening; then
    exit 0
  fi
  start_server
) >> "\$LOG" 2>&1 &
exit 0
EOF
chmod +x "$ROOT/boot-start.sh"
echo "已写入 $ROOT/boot-start.sh"

cat > "$ROOT/port80_proxy.py" << 'EOF'
# -*- coding: utf-8 -*-
import socket
import threading

LISTEN = ("0.0.0.0", 80)
TARGET = ("127.0.0.1", 7860)


def pipe(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        for sock in (src, dst):
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                sock.close()
            except Exception:
                pass


def handle(client):
    try:
        upstream = socket.create_connection(TARGET, 10)
    except Exception:
        client.close()
        return
    threading.Thread(target=pipe, args=(client, upstream), daemon=True).start()
    pipe(upstream, client)


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(LISTEN)
    server.listen(128)
    while True:
        client, _addr = server.accept()
        threading.Thread(target=handle, args=(client,), daemon=True).start()


if __name__ == "__main__":
    main()
EOF

if [ -d /.xgcos/desktop ]; then
  mkdir -p /.xgcos/desktop/train.app
  if [ ! -f /.xgcos/desktop/train.app/train.png ] && [ -f /.xgcos/desktop/home.app/home.png ]; then
    cp /.xgcos/desktop/home.app/home.png /.xgcos/desktop/train.app/train.png || true
  fi
  cat > /.xgcos/desktop/train.app/info.yaml << 'EOF'
name: 打开训练页
title: 打开训练页
icon: train.png
type: browser
props:
  port: 7860
  protocol: https
  path: /
EOF
  echo "已写入桌面快捷方式 /.xgcos/desktop/train.app"
fi

write_over() {
  local file="$1"
  local tmp="$2"
  local mode="$3"
  local err dir
  err="$(mktemp)"
  if cp "$tmp" "$file" 2>"$err"; then
    chmod "$mode" "$file" 2>/dev/null || chmod 755 "$file" || true
    rm -f "$err"
    return 0
  fi
  chattr -i "$file" 2>/dev/null || true
  chmod u+w "$file" 2>/dev/null || true
  if cp "$tmp" "$file" 2>"$err"; then
    chmod "$mode" "$file" 2>/dev/null || chmod 755 "$file" || true
    rm -f "$err"
    return 0
  fi
  dir="$(dirname "$file")"
  chmod u+w "$dir" 2>/dev/null || true
  if [ -w "$dir" ]; then
    if [ ! -e "$file.yprvc-bak" ]; then
      cp -a "$file" "$file.yprvc-bak" 2>/dev/null || true
    fi
    if rm -f "$file" 2>"$err" && cp "$tmp" "$file" 2>"$err"; then
      chmod "$mode" "$file" 2>/dev/null || chmod 755 "$file" || true
      rm -f "$err"
      return 0
    fi
  fi
  echo "不能写入: $file"
  ls -l "$file" 2>/dev/null || true
  lsattr -d "$file" 2>/dev/null || true
  findmnt -T "$file" -o TARGET,SOURCE,FSTYPE,OPTIONS 2>/dev/null || true
  sed -n '1,8p' "$err" 2>/dev/null || true
  rm -f "$err"
  return 1
}

patch_script() {
  local file="$1"
  [ -n "$file" ] || return 0
  [ -f "$file" ] || return 0
  case "$file" in
    /bin/*|/sbin/*|/usr/bin/*|/usr/sbin/*)
      case "$file" in
        *.sh) ;;
        *) return 0 ;;
      esac
      ;;
  esac
  case "$file" in
    "$ROOT"|"$ROOT"/*) return 0 ;;
  esac
  if [ "$(stat -c%s "$file" 2>/dev/null || echo 99999999)" -gt 1048576 ]; then
    echo "跳过，文件太大: $file"
    return 0
  fi
  if ! "$PY" -c 'import pathlib,sys; data=pathlib.Path(sys.argv[1]).read_bytes()[:512]; raise SystemExit(1 if b"\x00" in data else 0)' "$file"; then
    echo "跳过二进制: $file"
    return 0
  fi
  first="$(head -n 1 "$file" 2>/dev/null || true)"
  case "$first" in
    \#!*python*|\#!*perl*|\#!*node*)
      echo "跳过，不是 shell: $file"
      return 0
      ;;
  esac
  case "$file" in
    *.sh) ;;
    *)
      case "$first" in
        \#!*bash*|\#!*sh*) ;;
        *)
          echo "跳过，不是 shell 脚本: $file"
          return 0
          ;;
      esac
      ;;
  esac
  if grep -q "yprvc-autostart" "$file"; then
    echo "启动脚本里已经有训练页: $file"
    return 0
  fi
  local tmp mode
  mode="$(stat -c%a "$file" 2>/dev/null || echo 755)"
  tmp="$(mktemp)"
  if head -n 1 "$file" | grep -q '^#!'; then
    head -n 1 "$file" > "$tmp"
    cat >> "$tmp" << 'EOF'
# yprvc-autostart
if [ -x /root/YP-RVC/boot-start.sh ]; then
  bash /root/YP-RVC/boot-start.sh || true
fi
EOF
    tail -n +2 "$file" >> "$tmp"
  else
    cat > "$tmp" << 'EOF'
# yprvc-autostart
if [ -x /root/YP-RVC/boot-start.sh ]; then
  bash /root/YP-RVC/boot-start.sh || true
fi
EOF
    cat "$file" >> "$tmp"
  fi
  if write_over "$file" "$tmp" "$mode"; then
    rm -f "$tmp"
    echo "已挂到开机脚本: $file"
  else
    rm -f "$tmp"
  fi
}

boot_hooked=0
supervisor_init=0
if printf '%s' "$pid1" | grep -q supervisord; then
  supervisor_init=1
fi
if [ "$supervisor_init" -eq 0 ] && pgrep -x supervisord >/dev/null 2>&1; then
  supervisor_init=1
fi

if [ "$supervisor_init" -eq 1 ]; then
  conf_dir="/etc/supervisor/conf.d"
  if [ -d /etc/supervisord.d ]; then
    conf_dir="/etc/supervisord.d"
  fi
  mkdir -p "$conf_dir" "$ROOT/logs"
  cat > "$conf_dir/yprvc.conf" << EOF
[program:yprvc]
command=/bin/bash $ROOT/boot-start.sh foreground
directory=$ROOT
autostart=true
autorestart=true
startsecs=20
startretries=30
stopasgroup=true
killasgroup=true
stdout_logfile=$ROOT/logs/webui.log
stderr_logfile=$ROOT/logs/webui.err.log
user=root
EOF
  echo "已写入 $conf_dir/yprvc.conf"
  boot_hooked=1
  if command -v supervisorctl >/dev/null 2>&1; then
    supervisorctl reread || true
    supervisorctl update || true
    supervisorctl restart yprvc || supervisorctl start yprvc || true
  fi
else
  scripts=""
  add_script() {
    local file="$1"
    [ -n "$file" ] || return 0
    case " $scripts " in
      *" $file "*) return 0 ;;
    esac
    scripts="$scripts $file"
  }
  collect_pid() {
    local pid="$1"
    [ -r "/proc/$pid/cmdline" ] || return 0
    local arg
    while IFS= read -r arg; do
      case "$arg" in
        ""|-*) continue ;;
      esac
      if [ -f "$arg" ]; then
        add_script "$arg"
      fi
    done < <(tr '\0' '\n' < "/proc/$pid/cmdline")
  }
  collect_pid 1
  for status in /proc/[0-9]*/status; do
    ppid="$(awk '/^PPid:/ {print $2}' "$status" 2>/dev/null || true)"
    if [ "$ppid" = "1" ]; then
      collect_pid "$(basename "$(dirname "$status")")"
    fi
  done
  for file in \
    /init \
    /init.sh \
    /entrypoint.sh \
    /entry.sh \
    /run.sh \
    /opt/nvidia/nvidia_entrypoint.sh \
    /usr/local/bin/start-notebook.sh \
    /root/startup.sh \
    /scripts/start.sh
  do
    add_script "$file"
  done
  hooked=0
  for file in $scripts; do
    before="$(grep -c yprvc-autostart "$file" 2>/dev/null || true)"
    patch_script "$file"
    after="$(grep -c yprvc-autostart "$file" 2>/dev/null || true)"
    if [ "${after:-0}" != "${before:-0}" ]; then
      hooked=1
    fi
    if grep -q yprvc-autostart "$file" 2>/dev/null; then
      hooked=1
    fi
  done

  mkdir -p /start.d
  cat > /start.d/yprvc.sh << EOF
#!/bin/bash
if [ -x "$ROOT/boot-start.sh" ]; then
  bash "$ROOT/boot-start.sh" || true
fi
EOF
  chmod +x /start.d/yprvc.sh
  echo "已写入 /start.d/yprvc.sh"

  mkdir -p /root/.config/autostart /etc/xdg/autostart
  for desktop in /root/.config/autostart/yprvc.desktop /etc/xdg/autostart/yprvc.desktop; do
    cat > "$desktop" << EOF
[Desktop Entry]
Type=Application
Name=YP-RVC
Comment=start training page
Exec=/bin/bash $ROOT/boot-start.sh
X-GNOME-Autostart-enabled=true
NoDisplay=true
EOF
  done
  echo "已写入桌面自启动项"

  if command -v crontab >/dev/null 2>&1; then
    current="$(crontab -l 2>/dev/null || true)"
    if ! printf '%s\n' "$current" | grep -q 'YP-RVC/boot-start.sh'; then
      printf '%s\n@reboot /bin/bash %s/boot-start.sh\n' "$current" "$ROOT" | crontab -
      echo "已写入 crontab @reboot"
    fi
  fi

  boot_file="/scripts/start.sh"
  if [ ! -f "$boot_file" ]; then
    boot_file=""
  fi
  if [ -n "$boot_file" ] && grep -q '/scripts/start.d' "$boot_file" 2>/dev/null; then
    if mkdir -p /scripts/start.d && cat > /scripts/start.d/yprvc.sh << EOF
#!/bin/bash
if [ -x "$ROOT/boot-start.sh" ]; then
  bash "$ROOT/boot-start.sh" || true
fi
EOF
    then
      chmod +x /scripts/start.d/yprvc.sh
      boot_hooked=1
      echo "已写入 /scripts/start.d/yprvc.sh"
      echo "开机脚本会执行 /scripts/start.d，重启后会自启动。"
    else
      echo "不能写入 /scripts/start.d/yprvc.sh"
    fi
  elif [ -n "$boot_file" ] && grep -q yprvc-autostart "$boot_file" 2>/dev/null; then
    boot_hooked=1
    echo "开机脚本已包含训练页: $boot_file"
  elif [ -n "$boot_file" ] && [ -r "$boot_file" ]; then
    for file in $scripts /start.d/yprvc.sh /opt/nvidia/nvidia_entrypoint.sh; do
      [ -f "$file" ] || continue
      grep -q yprvc-autostart "$file" 2>/dev/null || continue
      if grep -q "$file" "$boot_file" || grep -q "$(basename "$file")" "$boot_file"; then
        boot_hooked=1
        echo "开机脚本 $boot_file 会调用 $file，重启后会自启动。"
        break
      fi
    done
    if [ "$boot_hooked" -eq 0 ] && grep -v '/scripts/start.d' "$boot_file" | grep -q '/start.d'; then
      boot_hooked=1
      echo "开机脚本会执行 /start.d，重启后会自启动。"
    fi
  fi
  if [ "$boot_hooked" -eq 0 ]; then
    echo "真正的开机脚本还没挂上: ${boot_file:-未知}"
    echo "发布镜像时，把启动命令设成: bash $ROOT/boot-start.sh foreground"
    if [ -n "$boot_file" ] && [ -r "$boot_file" ]; then
      echo "----- $boot_file 里和启动有关的行 -----"
      grep -nE 'startup|start\.d|entrypoint|nvidia_|jupyter|bash |\.sh' "$boot_file" | head -n 30 || true
    fi
  fi
fi

echo "===== Jupyter 默认打开 quick_setup.ipynb ====="
mkdir -p /root/.jupyter
cat > /root/.jupyter/jupyter_server_config.py << 'EOF'
c = get_config()
c.ServerApp.root_dir = "/root"
c.FileContentsManager.root_dir = "/root"
c.LabApp.default_url = "/lab/tree/YP-RVC/quick_setup.ipynb"
c.ServerApp.default_url = "/lab/tree/YP-RVC/quick_setup.ipynb"
EOF
echo "已写入 /root/.jupyter/jupyter_server_config.py"

echo "===== 刷新训练页链接 ====="
export PORT="$PORT"
export RVC_ROOT="$ROOT"
if [ -f "$ROOT/tools/refresh_train_link.py" ]; then
  "$PY" "$ROOT/tools/refresh_train_link.py" || python3 "$ROOT/tools/refresh_train_link.py" || true
else
  echo "缺少 $ROOT/tools/refresh_train_link.py，请先上传该文件"
fi

echo "===== 现在启动训练页 ====="
if [ "$supervisor_init" -eq 1 ]; then
  echo "由 supervisor 拉起。"
else
  bash "$ROOT/boot-start.sh"
fi

ready=0
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
  if ss -ltn 2>/dev/null | grep -q ":${PORT} " || netstat -ltn 2>/dev/null | grep -q ":${PORT} "; then
    ready=1
    break
  fi
  sleep 1
done

if [ "$ready" -eq 1 ]; then
  echo "训练页已在 0.0.0.0:${PORT} 监听。"
  if [ "$boot_hooked" -eq 1 ]; then
    echo "下次重启实例会自动再启动。日志: $ROOT/logs/webui.log"
  else
    echo "这次是手动拉起的。开机脚本没改成功，重启后不会自动出现。"
    echo "日志: $ROOT/logs/webui.log"
  fi
else
  echo "端口暂时还没监听到。请查看: $ROOT/logs/boot.log 和 $ROOT/logs/webui.log"
fi
