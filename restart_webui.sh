#!/usr/bin/env bash
# 强制重启有品训练 WebUI（7860）+ 80 端口代理（自定义服务常用）
#
# 用法：
#   bash /root/YP-RVC/restart_webui.sh
#   bash /root/restart_webui.sh
#   ROOT=/root/YP-RVC bash restart_webui.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

resolve_root() {
  if [[ -n "${ROOT:-}" && -f "${ROOT}/webui.py" ]]; then
    printf '%s\n' "$(cd "$ROOT" && pwd)"
    return 0
  fi
  local candidate
  for candidate in \
    "$SCRIPT_DIR" \
    "$SCRIPT_DIR/YP-RVC" \
    "/root/YP-RVC" \
    "$(pwd)/YP-RVC" \
    "$(pwd)"
  do
    if [[ -f "$candidate/webui.py" ]]; then
      printf '%s\n' "$(cd "$candidate" && pwd)"
      return 0
    fi
  done
  return 1
}

if ! ROOT="$(resolve_root)"; then
  echo "找不到 YP-RVC 项目（需要目录里有 webui.py）。" >&2
  echo "请指定：ROOT=/root/YP-RVC bash restart_webui.sh" >&2
  exit 1
fi

PORT="${PORT:-7860}"
APPLOG="${APPLOG:-$ROOT/logs/webui.log}"
PIDFILE="${PIDFILE:-$ROOT/logs/webui.pid}"
PROXY_LOG="${PROXY_LOG:-$ROOT/logs/port80.log}"
LOCKFILE="${LOCKFILE:-/tmp/yprvc-start.lock}"

cd "$ROOT"
mkdir -p "$ROOT/logs" "$ROOT/datasets" "$ROOT/logs/weights" "$ROOT/assets/weights"
export PORT
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"
export TZ="${TZ:-Asia/Shanghai}"
export RVC_DATASET_DIR="${RVC_DATASET_DIR:-$ROOT/datasets}"
export RVC_LOG_DIR="${RVC_LOG_DIR:-$ROOT/logs}"

echo "==> 项目: $ROOT"
echo "==> 训练端口: $PORT"
echo "==> 日志: $APPLOG"

kill_pattern() {
  local pattern="$1"
  if ! command -v pgrep >/dev/null 2>&1; then
    return 0
  fi
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -z "$pids" ]]; then
    return 0
  fi
  echo "==> 停止: $pattern"
  # shellcheck disable=SC2086
  kill $pids 2>/dev/null || true
  sleep 1
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  fi
}

free_port() {
  local p="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${p}/tcp" 2>/dev/null || true
  fi
  if command -v lsof >/dev/null 2>&1; then
    local pids
    pids="$(lsof -t -iTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      echo "==> 释放端口 $p: $pids"
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
      sleep 1
      # shellcheck disable=SC2086
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

port_listening() {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lnt 2>/dev/null | grep -qE ":${p}[[:space:]]"
    return $?
  fi
  if command -v netstat >/dev/null 2>&1; then
    netstat -lnt 2>/dev/null | grep -qE ":${p}[[:space:]]"
    return $?
  fi
  return 1
}

pick_python() {
  local c
  for c in \
    /root/miniconda3/envs/yprvc/bin/python3.11 \
    /root/miniconda3/envs/yprvc/bin/python \
    /usr/local/miniconda3/envs/yprvc/bin/python3.11 \
    /usr/local/miniconda3/envs/yprvc/bin/python \
    "$ROOT/.venv/bin/python"
  do
    if [[ -x "$c" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  done
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  command -v python3
}

force_stop_all() {
  echo "==> 强制停止旧服务"
  if command -v supervisorctl >/dev/null 2>&1; then
    if supervisorctl status yprvc >/dev/null 2>&1; then
      echo "==> supervisorctl stop yprvc"
      supervisorctl stop yprvc || true
    fi
  fi
  kill_pattern "$ROOT/webui.py"
  kill_pattern "python.*webui\\.py"
  kill_pattern "$ROOT/port80_proxy.py"
  kill_pattern "port80_proxy.py"
  kill_pattern "$ROOT/boot-start.sh"
  kill_pattern "YP-RVC/boot-start.sh"
  kill_pattern "$ROOT/start.sh"
  free_port "$PORT"
  free_port 80
  if [[ -f "$LOCKFILE" ]]; then
    echo "==> 清除启动锁: $LOCKFILE"
    if command -v lsof >/dev/null 2>&1; then
      local lpids
      lpids="$(lsof -t "$LOCKFILE" 2>/dev/null || true)"
      if [[ -n "$lpids" ]]; then
        # shellcheck disable=SC2086
        kill $lpids 2>/dev/null || true
        sleep 1
        # shellcheck disable=SC2086
        kill -9 $lpids 2>/dev/null || true
      fi
    fi
    rm -f "$LOCKFILE"
  fi
  rm -f "$PIDFILE"
  sleep 1
}

start_port80_proxy() {
  if [[ ! -f "$ROOT/port80_proxy.py" ]]; then
    echo "==> 无 port80_proxy.py，跳过 80 代理（自定义服务若走 80 将不可用）"
    return 0
  fi
  if port_listening 80; then
    echo "==> 80 已在监听，跳过代理启动"
    return 0
  fi
  echo "==> 启动 80→${PORT} 代理: port80_proxy.py"
  nohup python3 "$ROOT/port80_proxy.py" >>"$PROXY_LOG" 2>&1 &
  echo $! >"$ROOT/logs/port80.pid"
  sleep 1
  if port_listening 80; then
    echo "==> 80 代理已监听"
  else
    echo "==> 警告: 80 代理可能未起来，日志: $PROXY_LOG" >&2
  fi
}

start_webui() {
  local py
  py="$(pick_python)"
  echo "==> Python: $py"
  echo "==> 启动 webui.py"
  {
    echo ""
    echo "======== $(date '+%F %T') restart_webui.sh ========"
  } >>"$APPLOG"
  nohup "$py" "$ROOT/webui.py" >>"$APPLOG" 2>&1 &
  local pid=$!
  echo "$pid" >"$PIDFILE"
  echo "==> webui PID $pid"
}

force_stop_all

echo "==> 确认端口 $PORT 已释放 ..."
for i in $(seq 1 25); do
  if ! port_listening "$PORT"; then
    echo "  端口 $PORT 已空闲"
    break
  fi
  echo "  端口 $PORT 仍被占用，继续清理 (${i}/25)"
  free_port "$PORT"
  kill_pattern "$ROOT/webui.py"
  kill_pattern "python.*webui\\.py"
  sleep 1
done
if port_listening "$PORT"; then
  echo "错误: 无法释放端口 $PORT，拒绝改用 7861（避免 80 代理指错）。" >&2
  ss -lntp 2>/dev/null | grep -E ":${PORT}\\b" || true
  lsof -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true
  exit 1
fi

export STRICT_PORT=1
export PORT
start_webui

echo "==> 等待端口 $PORT ..."
ok=0
for i in $(seq 1 60); do
  if port_listening "$PORT"; then
    ok=1
    break
  fi
  if [[ -f "$PIDFILE" ]]; then
    pid="$(cat "$PIDFILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "==> 启动进程已退出 (PID $pid)" >&2
      tail -n 30 "$APPLOG" 2>/dev/null || true
      break
    fi
  fi
  sleep 1
  if (( i % 10 == 0 )); then
    echo "  ... ${i}s"
  fi
done

if [[ "$ok" -ne 1 ]]; then
  echo "端口 $PORT 暂时未监听到，请查看日志: $APPLOG" >&2
  tail -n 80 "$APPLOG" 2>/dev/null || true
  exit 1
fi

echo "训练页已监听 0.0.0.0:$PORT"
# 镜像自定义服务通常走 80，必须再拉起代理
start_port80_proxy

echo "重启完成。"
echo "  训练页直连: http://<IP>:$PORT"
echo "  自定义服务(若走80): 依赖 port80_proxy → $PORT"
echo "  日志: $APPLOG"
if [[ -f "$ROOT/port80_proxy.py" ]]; then
  echo "  代理日志: $PROXY_LOG"
fi
exit 0
