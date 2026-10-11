import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import argparse
import html
import json
import logging
import shutil
import socket
import threading
import traceback
from random import shuffle
import subprocess
from subprocess import Popen
import datetime
from time import sleep
from time import time as ttime

import faiss
import gradio as gr
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

from i18n.i18n import I18nAuto

LOG_DIR = Path(os.environ.get("RVC_LOG_DIR", ROOT / "logs")).resolve()
DATASET_DIR = Path(os.environ.get("RVC_DATASET_DIR", ROOT / "datasets")).resolve()
MUTE_DIR = ROOT / "logs" / "mute"
WEIGHT_DIR = LOG_DIR / "weights"
INDEX_DIR = LOG_DIR / "indices"
os.environ["RVC_LOG_DIR"] = str(LOG_DIR)
os.environ["RVC_WEIGHT_DIR"] = str(WEIGHT_DIR)
os.environ["RVC_DATASET_DIR"] = str(DATASET_DIR)

for path in (LOG_DIR, DATASET_DIR, WEIGHT_DIR, INDEX_DIR, ROOT / "assets" / "weights"):
    path.mkdir(parents=True, exist_ok=True)

def relative_to_root(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return Path(os.path.relpath(path, ROOT)).as_posix()


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("numba").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("yprvc")

i18n = I18nAuto("zh_CN")
SR_DICT = {"32k": 32000, "40k": 40000, "48k": 48000}
CONFIG_KEYS = [
    "v1/32k.json",
    "v1/40k.json",
    "v1/48k.json",
    "v2/32k.json",
    "v2/40k.json",
    "v2/48k.json",
]


def load_config_json():
    loaded = {}
    for key in CONFIG_KEYS:
        with open(ROOT / "configs" / key, "r", encoding="utf-8") as handle:
            loaded[key] = json.load(handle)
    return loaded


JSON_CONFIG = load_config_json()


def config_key(version, sr):
    # v2/40k.json 与上游 40k 超参一致。上游没有单独的 v2/40k 文件，40k 时改读 v1/40k.json。
    if version == "v1":
        return "v1/%s.json" % sr
    return "v2/%s.json" % sr


def gpu_snapshot():
    names = []
    mem = []
    if not torch.cuda.is_available():
        return names, mem
    for index in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(index)
        if any(
            token in name.upper()
            for token in [
                "10",
                "16",
                "20",
                "30",
                "40",
                "50",
                "A2",
                "A3",
                "A4",
                "P4",
                "A50",
                "500",
                "A60",
                "70",
                "80",
                "90",
                "M4",
                "T4",
                "TITAN",
                "4060",
                "L",
                "6000",
            ]
        ):
            names.append((str(index), name))
            mem.append(
                int(torch.cuda.get_device_properties(index).total_memory / 1024**3 + 0.4)
            )
    return names, mem


GPU_INFOS, GPU_MEM = gpu_snapshot()
if GPU_INFOS:
    GPU_INFO = "\n".join("%s\t%s" % item for item in GPU_INFOS)
    DEFAULT_BATCH = max(min(GPU_MEM) // 2, 1)
    GPUS = "-".join(item[0] for item in GPU_INFOS)
    IS_HALF = True
    gpu_name = GPU_INFOS[0][1]
    if (
        ("16" in gpu_name and "V100" not in gpu_name.upper())
        or any(token in gpu_name.upper() for token in ("P40", "P10"))
        or any(token in gpu_name for token in ("1060", "1070", "1080"))
    ):
        IS_HALF = False
    if GPU_MEM and min(GPU_MEM) <= 4:
        PREPROCESS_PER = 3.0
    else:
        PREPROCESS_PER = 3.7 if IS_HALF else 3.0
else:
    GPU_INFO = i18n("很遗憾您这没有能用的显卡来支持您训练")
    DEFAULT_BATCH = 1
    GPUS = "0"
    IS_HALF = False
    PREPROCESS_PER = 3.0

N_CPU = os.cpu_count() or 1
PYTHON = sys.executable


def child_env():
    env = os.environ.copy()
    env["RVC_LOG_DIR"] = str(LOG_DIR)
    env["RVC_WEIGHT_DIR"] = str(WEIGHT_DIR)
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def safe_exp_name(name):
    name = (name or "").strip()
    if not name:
        raise ValueError("请输入模型名称")
    if name in {".", "..", "mute"} or any(token in name for token in ("/", "\\", ":")):
        raise ValueError("模型名称只能是一层目录名，不能包含路径分隔符")
    return name


def exp_path(name):
    return LOG_DIR / safe_exp_name(name)


AUDIO_SUFFIXES = {
    ".wav",
    ".mp3",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".aiff",
    ".aif",
}


def dataset_folder(name, create):
    folder_name = safe_exp_name(name)
    root = DATASET_DIR.resolve()
    folder = (root / folder_name).resolve()
    if folder == root or root not in folder.parents:
        raise ValueError("音频只能放在模型自己的目录里")
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def audio_files_in(folder):
    found = []
    if not folder.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(folder, followlinks=False):
        current = Path(dirpath)
        for filename in filenames:
            path = current / filename
            if path.is_symlink() or not path.is_file():
                continue
            if path.suffix.lower() in AUDIO_SUFFIXES:
                found.append(path)
        dirnames[:] = [item for item in dirnames if not (current / item).is_symlink()]
    found.sort(key=lambda path: path.name.lower())
    return found


def describe_audio_dir(folder):
    files = audio_files_in(folder)
    if not files:
        return "目录 %s\n还没有音频。可以一次选择多个文件上传。" % folder
    names = [path.name for path in files]
    preview = names[:40]
    extra = ""
    if len(names) > len(preview):
        extra = "\n… 另有 %s 个" % (len(names) - len(preview))
    return "目录 %s\n共 %s 个音频:\n%s%s" % (folder, len(files), "\n".join(preview), extra)


def preview_dataset_dir(name):
    name = (name or "").strip()
    if not name:
        return "", "请输入模型名称"
    try:
        folder = dataset_folder(name, create=False)
    except ValueError as exc:
        return "", str(exc)
    if folder.is_dir():
        summary = describe_audio_dir(folder)
        write_ui_session(name, note="preview")
        return str(folder), summary
    write_ui_session(name, note="preview_new")
    return str(folder), "将新建目录 %s\n可以一次选择多个音频上传。" % folder


def upload_sources(files):
    if not files:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]
    found = []
    for item in files:
        if item is None:
            continue
        orig = None
        src = None
        if isinstance(item, dict):
            src = item.get("name") or item.get("path")
            orig = item.get("orig_name")
        else:
            orig = getattr(item, "orig_name", None)
            named = getattr(item, "name", None)
            if isinstance(item, (str, Path)):
                src = item
            elif named:
                src = named
        if not src:
            continue
        src_path = Path(src)
        found.append((src_path, orig or src_path.name))
    return found


def safe_audio_filename(original):
    name = Path(str(original).replace("\\", "/")).name.strip()
    if not name or name in {".", ".."} or "\x00" in name:
        raise ValueError("文件名无效")
    if Path(name).suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(name)
    return name


def summarize_saved_audio(folder, saved, skipped):
    lines = ["已保存 %s 个文件到 %s" % (len(saved), folder)]
    if saved:
        shown = saved[:30]
        lines.append("本次写入: " + "、".join(shown))
        if len(saved) > len(shown):
            lines.append("另有 %s 个文件" % (len(saved) - len(shown)))
    if skipped:
        lines.append("已跳过 %s 个非音频或无效文件" % len(skipped))
    lines.append(describe_audio_dir(folder))
    return "\n".join(lines)


def upload_dataset_audio(name, files):
    # file_count="multiple"
    sources = upload_sources(files)
    if not (name or "").strip():
        if not sources:
            return gr.update(), gr.update(), gr.update(interactive=False, label="请输入模型名称")
        return gr.update(), "请输入模型名称", gr.update(value=None, interactive=False, label="请输入模型名称")
    if not sources:
        return gr.update(), gr.update(), gr.update()
    try:
        folder = dataset_folder(name, create=True)
    except ValueError as exc:
        return gr.update(), str(exc), gr.update(value=None, interactive=False, label="请输入模型名称")
    saved = []
    skipped = []
    seen = set()
    for src, orig in sources:
        if not src.is_file():
            skipped.append(str(orig))
            continue
        try:
            dest_name = safe_audio_filename(orig)
        except ValueError:
            skipped.append(str(orig))
            continue
        shutil.copy2(src, folder / dest_name)
        if dest_name not in seen:
            saved.append(dest_name)
            seen.add(dest_name)
    return str(folder), summarize_saved_audio(folder, saved, skipped), gr.update()


def purge_directory(folder):
    removed = 0
    for dirpath, dirnames, filenames in os.walk(folder, topdown=False, followlinks=False):
        current = Path(dirpath)
        for filename in filenames:
            (current / filename).unlink()
            removed += 1
        for dirname in dirnames:
            path = current / dirname
            if path.is_symlink():
                path.unlink()
                removed += 1
        if current != folder:
            try:
                current.rmdir()
            except OSError:
                pass
    return removed


def clear_dataset_audio(name):
    if train_process() is not None:
        return gr.update(), "训练进行中，先停止后再清空音频"
    try:
        folder = dataset_folder(name, create=False)
    except ValueError as exc:
        return gr.update(), str(exc)
    root = DATASET_DIR.resolve()
    if not folder.is_dir():
        return str(folder), "还没有音频可删。将使用目录 %s" % folder
    resolved = folder.resolve()
    if resolved == root or root not in resolved.parents:
        return gr.update(), "只能清空某个模型自己的音频目录"
    removed = purge_directory(resolved)
    write_ui_session(name, note="clear_audio")
    return str(resolved), "已彻底删除 %s 个文件。\n%s" % (removed, describe_audio_dir(resolved))


def finished_model_path(name):
    model_name = safe_exp_name(name)
    filename = "%s.pth" % model_name
    for folder in (WEIGHT_DIR, ROOT / "assets" / "weights"):
        path = folder / filename
        if path.is_file():
            return path
    return None


def finished_index_path(name):
    model_name = safe_exp_name(name)
    filename = "%s.index" % model_name
    candidates = [INDEX_DIR / filename]
    try:
        candidates.append(exp_path(model_name) / filename)
    except ValueError:
        pass
    for path in candidates:
        if path.is_file():
            return path
    return None


def expose_download(path, dest_name):
    if path is None:
        return None
    out_dir = ROOT / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / dest_name
    source = Path(path).resolve()
    if dest.exists() or dest.is_symlink():
        if dest.resolve() != source:
            dest.unlink()
        else:
            return str(dest)
    try:
        os.link(source, dest)
    except OSError:
        shutil.copy2(source, dest)
    return str(dest)


def refresh_downloads(exp_name):
    try:
        name = safe_exp_name(exp_name)
        model = finished_model_path(name)
        index = finished_index_path(name)
    except ValueError as exc:
        return None, None, str(exc)
    lines = []
    model_out = None
    index_out = None
    if model is None:
        lines.append("还没有完整模型 %s.pth。训练正常结束后会生成，中途按轮次另存的分段权重不会出现在这里。" % name)
    else:
        model_out = expose_download(model, "%s.pth" % name)
        lines.append("完整模型 %s" % model)
    if index is None:
        lines.append("还没有索引 %s.index。特征索引训练完成后会生成。" % name)
    else:
        index_out = expose_download(index, "%s.index" % name)
        lines.append("索引 %s" % index)
    return model_out, index_out, "\n".join(lines)


def pick_port(start):
    """优先占用指定端口。STRICT_PORT=1（默认）时不自动改到 7861，避免代理仍指向旧端口。"""
    strict = str(os.environ.get("STRICT_PORT", "1")).strip() not in {"0", "false", "False"}
    last_err = None
    for attempt in range(15 if strict else 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", int(start)))
                return int(start)
            except OSError as exc:
                last_err = exc
        if not strict:
            break
        sleep(1)
    if strict:
        raise RuntimeError(
            "端口 %s 仍被占用（已重试）。请先执行 restart_webui.sh 或释放该端口。原错误: %s"
            % (start, last_err)
        )
    for port in range(int(start), min(int(start) + 100, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("7860 附近没有可用端口")


def spawn(args):
    logger.info("Execute: %s", " ".join(args))
    # 独立会话：刷新/重启 WebUI 时不要连带杀掉预处理/提特征子进程
    return Popen(
        args,
        cwd=str(ROOT),
        env=child_env(),
        start_new_session=True,
    )


PAUSE_FILE = "train.pause"
STOP_FILE = "train.stop"
TRAIN_LOCK = threading.Lock()
TRAIN_STATE = {
    "process": None,
    "folder": None,
    "exp_name": None,
    "stop_requested": False,
}
WEBUI_LOG_PATH = ROOT / "logs" / "webui.log"
UPLOAD_LOG_PATH = ROOT / "logs" / "upload.log"
TRAIN_SESSION_PATH = ROOT / "logs" / ".train_ui_session.json"
UI_SESSION_PATH = ROOT / "logs" / ".ui_session.json"
JOB_SESSION_PATH = ROOT / "logs" / ".job_session.json"
TRAIN_POLL_SECONDS = 5
JOB_LOCK = threading.Lock()
JOB_STATE = {
    "stage": "idle",
    "exp_name": "",
    "pids": [],
    "log_path": "",
    "label": "",
    "done": True,
    "message": "",
}


def log_upload(message, *args, level="info"):
    """上传相关行为写入 webui logger + logs/upload.log。"""
    text = message % args if args else str(message)
    log_fn = getattr(logger, level, logger.info)
    log_fn("[upload] %s", text)
    try:
        UPLOAD_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with UPLOAD_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write("%s %s\n" % (stamp, text))
            handle.flush()
    except OSError:
        pass


class ExternalProcess:
    """WebUI 重启后按 PID 接管仍在跑的 train.py，不拥有真正的 Popen。"""

    def __init__(self, pid):
        self.pid = int(pid)

    def poll(self):
        return None if pid_is_alive(self.pid) else 0

    def wait(self, timeout=None):
        start = ttime() if timeout is not None else None
        while self.poll() is None:
            if timeout is not None and (ttime() - start) >= timeout:
                raise subprocess.TimeoutExpired(cmd="external-train", timeout=timeout)
            sleep(0.5)
        return 0


def pid_is_alive(pid):
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def train_process():
    process = TRAIN_STATE["process"]
    if process is None:
        return None
    try:
        if process.poll() is not None:
            return None
    except Exception:
        return None
    return process


def clear_train_flags(folder):
    if folder is None:
        return
    for name in (PAUSE_FILE, STOP_FILE):
        path = folder / name
        if path.is_file():
            path.unlink()


def write_train_session(exp_name, folder, phase, pid=None):
    payload = {
        "exp_name": exp_name or "",
        "folder": str(folder) if folder else "",
        "phase": phase,
        "pid": int(pid) if pid else None,
    }
    TRAIN_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_SESSION_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if exp_name:
        write_ui_session(exp_name, note="train", train_phase=phase)


def read_train_session():
    if not TRAIN_SESSION_PATH.is_file():
        return None
    try:
        return json.loads(TRAIN_SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_ui_session():
    if not UI_SESSION_PATH.is_file():
        return None
    try:
        data = json.loads(UI_SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def write_ui_session(exp_name, note="", **extra):
    """从输入模型名称起记录工作台状态，供刷新/重开时询问回显。"""
    name = (exp_name or "").strip()
    if not name:
        return None
    try:
        safe = safe_exp_name(name)
    except ValueError:
        return None
    prev = read_ui_session() or {}
    payload = dict(prev)
    # config / 输出字段做合并，避免只写名称时冲掉已有配置
    prev_config = prev.get("config") if isinstance(prev.get("config"), dict) else {}
    new_config = extra.pop("config", None)
    if isinstance(new_config, dict):
        merged = dict(prev_config)
        merged.update(new_config)
        payload["config"] = merged
    elif "config" not in payload and prev_config:
        payload["config"] = prev_config
    for key in ("info1", "info2", "info3"):
        if key in extra and extra[key] is not None:
            payload[key] = extra.pop(key)
    payload.update(extra or {})
    payload["exp_name"] = safe
    payload["note"] = note or payload.get("note") or ""
    payload["updated_at"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        folder = dataset_folder(safe, create=False)
        payload["dataset_dir"] = str(folder)
        if folder.is_dir():
            files = audio_files_in(folder)
            payload["audio_count"] = len(files)
            payload["audio_names"] = [path.name for path in files[:40]]
            payload["audio_summary"] = describe_audio_dir(folder)
        else:
            payload["audio_count"] = 0
            payload["audio_names"] = []
            payload["audio_summary"] = "将新建目录 %s\n可以一次选择多个音频上传。" % folder
    except ValueError as exc:
        payload["dataset_dir"] = ""
        payload["audio_count"] = 0
        payload["audio_names"] = []
        payload["audio_summary"] = str(exc)
    try:
        UI_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        UI_SESSION_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.exception("写入 UI 会话失败")
        return None
    return payload


def save_full_ui_session(
    exp_name,
    version19,
    sr2,
    if_f0_3,
    np7,
    trainset_dir4,
    spk_id5,
    gpus6,
    f0method8,
    gpus_rmvpe,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    if_cache_gpu17,
    if_save_every_weights18,
    pretrained_g14,
    pretrained_d15,
    gpus16,
    info1="",
    info2="",
    info3="",
):
    """保存模型名称 + 全部配置项 + 输出区文案。"""
    name = (exp_name or "").strip()
    if not name:
        return
    config = {
        "version19": version19,
        "sr2": sr2,
        "if_f0_3": if_f0_3,
        "np7": np7,
        "trainset_dir4": trainset_dir4,
        "spk_id5": spk_id5,
        "gpus6": gpus6,
        "f0method8": f0method8,
        "gpus_rmvpe": gpus_rmvpe,
        "save_epoch10": save_epoch10,
        "total_epoch11": total_epoch11,
        "batch_size12": batch_size12,
        "if_save_latest13": if_save_latest13,
        "if_cache_gpu17": if_cache_gpu17,
        "if_save_every_weights18": if_save_every_weights18,
        "pretrained_g14": pretrained_g14,
        "pretrained_d15": pretrained_d15,
        "gpus16": gpus16,
    }
    write_ui_session(
        name,
        note="form",
        config=config,
        info1=info1 or "",
        info2=info2 or "",
        info3=info3 or "",
    )


def _cfg_bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "是"}:
        return True
    if text in {"0", "false", "no", "n", "否"}:
        return False
    return default


def _cfg_number(value, default, cast=float):
    try:
        if value is None or value == "":
            return default
        return cast(value)
    except (TypeError, ValueError):
        return default


def restore_all_ui():
    """Gradio 原生回填：模型名、音频摘要、全部配置、输出与训练按钮。"""
    session = read_ui_session() or {}
    exp_name = (session.get("exp_name") or "").strip()
    train_session = read_train_session() or {}
    if not exp_name:
        exp_name = (train_session.get("exp_name") or "").strip()
    cfg = session.get("config") if isinstance(session.get("config"), dict) else {}
    yes = i18n("是")
    no = i18n("否")
    default_np = max(int(np.ceil(N_CPU / 1.5)), 1)
    default_rmvpe = "%s-%s" % (GPUS, GPUS)

    if not exp_name:
        path, summary = "", "请输入模型名称"
        phase = "hidden"
        message = ""
        info1 = info2 = ""
        model_dl = index_dl = None
        dl_info = "训练或索引完成后会自动出现在这里。也可以稍后点刷新下载。"
        version = "v2"
        sr = "40k"
        if_f0 = True
        np_val = default_np
        spk = 0
        gpus_feat = GPUS
        f0method = "rmvpe_gpu"
        gpus_rmvpe_val = default_rmvpe
        save_ep = 5
        total_ep = 20
        batch = DEFAULT_BATCH
        save_latest = no
        cache_gpu = no
        save_every = yes
        pre_g = "assets/pretrained_v2/f0G40k.pth"
        pre_d = "assets/pretrained_v2/f0D40k.pth"
        gpus_train = GPUS
    else:
        path, summary = preview_dataset_dir(exp_name)
        train = get_train_status_payload()
        job = get_job_status_payload()
        phase = train.get("phase") or "hidden"
        message = train.get("message") or job.get("info3") or session.get("info3") or ""
        info1 = job.get("info1") or session.get("info1") or ""
        info2 = job.get("info2") or session.get("info2") or ""
        model_dl, index_dl, dl_info = refresh_downloads(exp_name)
        version = cfg.get("version19") or "v2"
        sr = cfg.get("sr2") or "40k"
        if_f0 = _cfg_bool(cfg.get("if_f0_3"), True)
        np_val = int(_cfg_number(cfg.get("np7"), default_np, float))
        spk = int(_cfg_number(cfg.get("spk_id5"), 0, float))
        gpus_feat = cfg.get("gpus6") if cfg.get("gpus6") is not None else GPUS
        f0method = cfg.get("f0method8") or "rmvpe_gpu"
        gpus_rmvpe_val = cfg.get("gpus_rmvpe") or default_rmvpe
        save_ep = int(_cfg_number(cfg.get("save_epoch10"), 5, float))
        total_ep = int(_cfg_number(cfg.get("total_epoch11"), 20, float))
        batch = int(_cfg_number(cfg.get("batch_size12"), DEFAULT_BATCH, float))
        save_latest = cfg.get("if_save_latest13") or no
        cache_gpu = cfg.get("if_cache_gpu17") or no
        save_every = cfg.get("if_save_every_weights18") or yes
        pre_g = cfg.get("pretrained_g14") or "assets/pretrained_v2/f0G40k.pth"
        pre_d = cfg.get("pretrained_d15") or "assets/pretrained_v2/f0D40k.pth"
        gpus_train = cfg.get("gpus16") if cfg.get("gpus16") is not None else GPUS
        if cfg.get("trainset_dir4") and not path:
            path = cfg.get("trainset_dir4")

    pause, resume, stop = control_updates(phase)
    return (
        exp_name,
        version,
        sr,
        if_f0,
        np_val,
        path,
        spk,
        summary,
        info1,
        gpus_feat,
        f0method,
        gpus_rmvpe_val,
        info2,
        save_ep,
        total_ep,
        batch,
        save_latest,
        cache_gpu,
        save_every,
        pre_g,
        pre_d,
        gpus_train,
        message,
        pause,
        resume,
        stop,
        model_dl,
        index_dl,
        dl_info,
    )


def build_ui_session_payload():
    session = read_ui_session() or {}
    exp_name = (session.get("exp_name") or "").strip()
    train_session = read_train_session() or {}
    if not exp_name:
        exp_name = (train_session.get("exp_name") or "").strip()
    if exp_name:
        session = write_ui_session(exp_name, note=session.get("note") or "refresh") or session
    train = get_train_status_payload()
    try:
        job = get_job_status_payload()
    except Exception:
        job = {"ok": False, "running": False}
    has_session = bool(exp_name)
    audio_count = int(session.get("audio_count") or 0)
    cfg = session.get("config") if isinstance(session.get("config"), dict) else {}
    prompt_lines = [
        "检测到上次工作记录，是否回显到当前页面？",
        "将恢复：模型名称、音频列表、全部配置，以及任务/训练日志。",
        "",
        "模型名称：%s" % (exp_name or "（空）"),
        "已上传音频：%s 个" % audio_count,
        "音频目录：%s" % (session.get("dataset_dir") or "（尚未创建）"),
        "版本/采样率：%s / %s" % (cfg.get("version19") or "—", cfg.get("sr2") or "—"),
        "记录时间：%s" % (session.get("updated_at") or "—"),
    ]
    if job.get("running"):
        prompt_lines.append(
            "进行中任务：%s" % (job.get("label") or job.get("stage") or "未知")
        )
    elif train.get("running"):
        prompt_lines.append("训练状态：进行中")
    else:
        prompt_lines.append("训练状态：%s" % (train.get("phase") or "无"))
    return {
        "ok": True,
        "has_session": has_session,
        "exp_name": exp_name,
        "dataset_dir": session.get("dataset_dir") or "",
        "audio_count": audio_count,
        "audio_names": session.get("audio_names") or [],
        "audio_summary": session.get("audio_summary") or "",
        "updated_at": session.get("updated_at") or "",
        "config": cfg,
        "train": train,
        "job": job,
        "prompt_lines": prompt_lines,
    }


def mark_train_session_done(message=""):
    session = read_train_session() or {}
    session["phase"] = "done"
    session["message"] = message
    session["pid"] = None
    TRAIN_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRAIN_SESSION_PATH.write_text(
        json.dumps(session, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def find_pid_for_exp(exp_name):
    if not exp_name:
        return None
    needle = str(exp_name)
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid,args"],
            text=True,
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if "infer/modules/train/train.py" not in line:
            continue
        if needle not in line:
            continue
        parts = line.strip().split(None, 1)
        if not parts:
            continue
        try:
            return int(parts[0])
        except ValueError:
            continue
    return None


def discover_active_train_folder():
    if not LOG_DIR.is_dir():
        return None
    best = None
    best_mtime = 0.0
    now = ttime()
    for folder in LOG_DIR.iterdir():
        if not folder.is_dir():
            continue
        progress = folder / "train_progress.txt"
        if not progress.is_file():
            continue
        try:
            mtime = progress.stat().st_mtime
        except OSError:
            continue
        if now - mtime > 45 * 60:
            continue
        if mtime >= best_mtime:
            best = folder
            best_mtime = mtime
    return best


def attach_train_state(exp_name, folder, pid, stop_requested=False):
    process = ExternalProcess(pid)
    with TRAIN_LOCK:
        TRAIN_STATE["process"] = process
        TRAIN_STATE["folder"] = Path(folder)
        TRAIN_STATE["exp_name"] = exp_name or Path(folder).name
        TRAIN_STATE["stop_requested"] = bool(stop_requested)
    phase = train_phase_for_folder(Path(folder))
    write_train_session(TRAIN_STATE["exp_name"], Path(folder), phase, pid)
    return process


def ensure_train_attached():
    """刷新或 WebUI 重启后，尽量把仍在跑的 train.py 接回 TRAIN_STATE。"""
    current = train_process()
    if current is not None:
        return current
    session = read_train_session() or {}
    folder = Path(session["folder"]) if session.get("folder") else None
    exp_name = session.get("exp_name") or (folder.name if folder else "")
    pid = session.get("pid")
    if folder and folder.is_dir() and pid_is_alive(pid):
        return attach_train_state(exp_name, folder, pid)
    if exp_name:
        found = find_pid_for_exp(exp_name)
        if found and folder and folder.is_dir():
            return attach_train_state(exp_name, folder, found)
    folder = discover_active_train_folder()
    if folder is None:
        return None
    found = find_pid_for_exp(folder.name)
    if not found:
        return None
    return attach_train_state(folder.name, folder, found)


def kill_process_tree(pid):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            capture_output=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return
    try:
        os.kill(pid, 15)
    except OSError:
        return


def force_stop_later(process, pid, delay=20):
    try:
        process.wait(timeout=delay)
    except subprocess.TimeoutExpired:
        if process.poll() is None and process.pid == pid:
            kill_process_tree(pid)


def release_train_process(process, folder, final_message=""):
    with TRAIN_LOCK:
        if TRAIN_STATE["process"] is process:
            TRAIN_STATE["process"] = None
            TRAIN_STATE["folder"] = None
            TRAIN_STATE["exp_name"] = None
    clear_train_flags(folder)
    mark_train_session_done(final_message)


def wait_process(process):
    done = [False]

    def watch():
        process.wait()
        done[0] = True
        finish_job_if_pids([getattr(process, "pid", None)])

    threading.Thread(target=watch, daemon=True).start()
    return done


def wait_processes(processes):
    done = [False]
    pids = [getattr(proc, "pid", None) for proc in processes]

    def watch():
        for process in processes:
            process.wait()
        done[0] = True
        finish_job_if_pids(pids)

    threading.Thread(target=watch, daemon=True).start()
    return done


def read_log(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def write_job_session(stage, exp_name, pids, log_path, label="", done=False, message=""):
    clean_pids = []
    for pid in pids or []:
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            continue
        if pid > 0:
            clean_pids.append(pid)
    payload = {
        "stage": stage or "idle",
        "exp_name": (exp_name or "").strip(),
        "pids": clean_pids,
        "log_path": str(log_path or ""),
        "label": label or "",
        "done": bool(done),
        "message": message or "",
        "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with JOB_LOCK:
        JOB_STATE.update(payload)
    try:
        JOB_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        JOB_SESSION_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.exception("写入任务会话失败")
    if exp_name:
        write_ui_session(exp_name, note="job:%s" % stage)
    return payload


def read_job_session():
    if not JOB_SESSION_PATH.is_file():
        return None
    try:
        data = json.loads(JOB_SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def finish_job_if_pids(pids):
    session = read_job_session() or {}
    tracked = set(int(x) for x in (session.get("pids") or []) if str(x).isdigit())
    finished = set()
    for pid in pids or []:
        try:
            finished.add(int(pid))
        except (TypeError, ValueError):
            pass
    if tracked and not tracked.issubset(finished):
        # 只结束当前这批里仍登记的 pid；若还有别的存活 pid，不标完成
        alive = [pid for pid in tracked if pid_is_alive(pid)]
        if alive:
            write_job_session(
                session.get("stage") or "idle",
                session.get("exp_name") or "",
                alive,
                session.get("log_path") or "",
                label=session.get("label") or "",
                done=False,
                message=session.get("message") or "",
            )
            return
    log_text = ""
    log_path = session.get("log_path") or ""
    if log_path:
        log_text = read_log(Path(log_path))
    label = session.get("label") or "任务"
    message = "%s已结束\n%s" % (label, log_text[-4000:] if log_text else "")
    write_job_session(
        session.get("stage") or "idle",
        session.get("exp_name") or "",
        [],
        log_path,
        label=label,
        done=True,
        message=message.strip(),
    )
    exp_name = session.get("exp_name") or ""
    if exp_name:
        stage = session.get("stage") or ""
        extra = {}
        if stage == "preprocess":
            extra["info1"] = message.strip()
        elif stage in {"extract_f0", "extract_feature", "extract"}:
            extra["info2"] = message.strip()
        if extra:
            write_ui_session(exp_name, note="job_done", **extra)


def list_process_cmdline():
    try:
        out = subprocess.check_output(
            ["ps", "-eo", "pid=,args="],
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    rows = []
    for line in out.splitlines():
        text = line.strip()
        if not text:
            continue
        parts = text.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        rows.append((pid, parts[1]))
    return rows


def discover_pipeline_jobs(exp_name=None):
    """从进程表发现仍在跑的预处理/提特征任务（刷新或重启 WebUI 后可接管）。"""
    needle = None
    if exp_name:
        try:
            needle = str(exp_path(exp_name))
        except ValueError:
            needle = None
    found = []
    rules = [
        ("preprocess", "infer/modules/train/preprocess.py", "preprocess.log", "数据预处理"),
        ("extract_f0", "extract_f0_rmvpe.py", "extract_f0_feature.log", "音高提取"),
        ("extract_f0", "extract_f0_print.py", "extract_f0_feature.log", "音高提取"),
        ("extract_feature", "extract_feature_print.py", "extract_f0_feature.log", "特征提取"),
    ]
    for pid, args in list_process_cmdline():
        for stage, token, log_name, label in rules:
            if token not in args:
                continue
            if needle and needle not in args:
                # 允许用模型名目录片段匹配
                if not exp_name or ("/%s" % exp_name) not in args.replace("\\", "/"):
                    continue
            # 从命令行猜实验目录（优先 logs/ 下的实验目录，避免误判 datasets/）
            folder = None
            chunks = args.split()
            candidates = []
            for item in chunks:
                path = Path(item)
                if path.name in {
                    "preprocess.py",
                    "extract_f0_rmvpe.py",
                    "extract_f0_print.py",
                    "extract_feature_print.py",
                }:
                    continue
                norm = item.replace("\\", "/")
                if not path.is_dir():
                    continue
                candidates.append(path)
                if "/logs/" in norm or norm.rstrip("/").endswith("/logs/" + path.name):
                    folder = path
                    break
            if folder is None:
                for path in candidates:
                    try:
                        if LOG_DIR.resolve() in path.resolve().parents or path.resolve().parent == LOG_DIR.resolve():
                            folder = path
                            break
                    except OSError:
                        continue
            if folder is None and needle:
                folder = Path(needle)
            if folder is None:
                continue
            found.append(
                {
                    "stage": stage,
                    "pid": pid,
                    "exp_name": folder.name,
                    "log_path": str(folder / log_name),
                    "label": label,
                }
            )
            break
    return found


def begin_job(stage, exp_name, processes, log_path, label):
    pids = []
    for proc in processes if isinstance(processes, (list, tuple)) else [processes]:
        pid = getattr(proc, "pid", None)
        if pid:
            pids.append(pid)
    message = "%s进行中…" % label
    write_job_session(
        stage,
        exp_name,
        pids,
        log_path,
        label=label,
        done=False,
        message=message,
    )
    return message


def get_job_status_payload():
    session = read_job_session() or {}
    exp_name = (session.get("exp_name") or "").strip()
    ui = read_ui_session() or {}
    if not exp_name:
        exp_name = (ui.get("exp_name") or "").strip()
    discovered = discover_pipeline_jobs(exp_name or None)
    if discovered and exp_name:
        discovered = [item for item in discovered if item.get("exp_name") == exp_name] or discovered
    alive_pids = []
    stage = session.get("stage") or "idle"
    label = session.get("label") or ""
    log_path = session.get("log_path") or ""
    if discovered:
        alive_pids = [item["pid"] for item in discovered]
        # 优先特征提取 > 音高 > 预处理
        order = {"extract_feature": 3, "extract_f0": 2, "preprocess": 1}
        top = sorted(discovered, key=lambda item: order.get(item["stage"], 0), reverse=True)[0]
        stage = top["stage"]
        label = top["label"]
        log_path = top["log_path"]
        exp_name = top.get("exp_name") or exp_name
        write_job_session(
            stage,
            exp_name,
            alive_pids,
            log_path,
            label=label,
            done=False,
            message="%s进行中（已自动接管）…" % label,
        )
    else:
        for pid in session.get("pids") or []:
            if pid_is_alive(pid):
                alive_pids.append(int(pid))
    running = bool(alive_pids)
    log_text = read_log(Path(log_path)) if log_path else ""
    if running:
        header = "【%s进行中】模型 %s\n" % (label or stage, exp_name or "")
        body = header + (log_text[-6000:] if log_text else "日志尚未写出，任务仍在运行…")
        done = False
    else:
        # 进程已不在但会话未收尾：按结束处理，避免刷新后“假死”
        if (session.get("pids") or []) and not session.get("done"):
            message = ("%s已结束（进程已退出）\n%s" % (label or "任务", log_text[-4000:] if log_text else "")).strip()
            write_job_session(
                stage or "idle",
                exp_name,
                [],
                log_path,
                label=label,
                done=True,
                message=message,
            )
            session = read_job_session() or session
        done = True if session.get("done") else bool(log_text)
        if session.get("message"):
            body = session.get("message")
        elif log_text:
            body = "【%s】模型 %s\n%s" % (label or stage or "任务", exp_name or "", log_text[-6000:])
        else:
            body = ""
    info1 = ui.get("info1") or ""
    info2 = ui.get("info2") or ""
    info3 = ui.get("info3") or ""
    if body:
        if stage == "preprocess":
            info1 = body
        elif stage in {"extract_f0", "extract_feature", "extract"}:
            info2 = body
    # 训练中时 info3 仍由 train-status 负责；这里附带最近训练进度摘要
    train = None
    try:
        train = get_train_status_payload()
        if train.get("message"):
            info3 = train.get("message")
    except Exception:
        train = {"ok": False}
    if exp_name and (info1 or info2):
        write_ui_session(exp_name, note="job_status", info1=info1, info2=info2, info3=info3 or ui.get("info3") or "")
    return {
        "ok": True,
        "running": running,
        "done": done,
        "stage": stage,
        "label": label,
        "exp_name": exp_name,
        "pids": alive_pids,
        "log_path": log_path,
        "info1": info1,
        "info2": info2,
        "info3": info3,
        "train": train,
    }


def dedupe_consecutive_lines(lines):
    out = []
    last = None
    for line in lines:
        if line == last:
            continue
        out.append(line)
        last = line
    return out


def tail_text_lines(path, limit=80):
    if not path or not Path(path).is_file():
        return []
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return dedupe_consecutive_lines(text.splitlines())[-limit:]


def train_phase_for_folder(folder):
    if folder is None:
        return "hidden"
    if (folder / STOP_FILE).is_file():
        return "running"
    if (folder / PAUSE_FILE).is_file():
        return "paused"
    return "running"


def format_train_message(folder, prefix="训练进行中"):
    progress_lines = []
    if folder is not None:
        progress_lines = tail_text_lines(folder / "train_progress.txt", 30)
    log_lines = tail_text_lines(WEBUI_LOG_PATH, 40)
    blocks = []
    if progress_lines:
        blocks.append("【训练进度】\n" + "\n".join(progress_lines))
    elif prefix:
        blocks.append(prefix + "\n正在加载模型")
    if log_lines:
        blocks.append("【webui.log】\n" + "\n".join(log_lines))
    if not blocks:
        return prefix or "暂无训练输出"
    if prefix and progress_lines:
        return prefix + "\n" + "\n\n".join(blocks)
    return "\n\n".join(blocks)


def follow_train_output(process, folder, prefix="训练进行中"):
    while process.poll() is None:
        phase = train_phase_for_folder(folder)
        yield format_train_message(folder, prefix=prefix), phase
        write_train_session(
            TRAIN_STATE.get("exp_name") or (folder.name if folder else ""),
            folder,
            phase,
            process.pid,
        )
        sleep(TRAIN_POLL_SECONDS)
    text = ""
    if folder is not None and (folder / "train_progress.txt").is_file():
        text = "\n".join(tail_text_lines(folder / "train_progress.txt", 30)).strip()
    with TRAIN_LOCK:
        stopped = TRAIN_STATE["stop_requested"]
    code = process.wait()
    if stopped:
        message = "训练已终止。再次训练会从最近一次保存的检查点继续\n" + text
        release_train_process(process, folder, message)
        yield message, "hidden"
        return
    if code not in (0, None):
        message = "训练异常退出，退出码 %s\n%s" % (code, text)
        release_train_process(process, folder, message)
        yield message, "hidden"
        return
    message = "训练结束\n" + text
    release_train_process(process, folder, message)
    yield message, "hidden"


def read_webui_log_view(limit=200):
    lines = tail_text_lines(WEBUI_LOG_PATH, limit)
    if not lines:
        return "还没有 webui.log，或文件为空：%s" % WEBUI_LOG_PATH
    return "【webui.log 最近 %s 行】\n%s" % (len(lines), "\n".join(lines))


def resolve_log_source(source):
    key = (source or "webui").strip().lower()
    if key in {"upload", "upload.log", "uploads"}:
        return "upload", UPLOAD_LOG_PATH
    if key in {"all", "both"}:
        return "all", None
    return "webui", WEBUI_LOG_PATH


def read_log_payload(source="webui", limit=500):
    try:
        limit = max(20, min(int(limit or 500), 5000))
    except (TypeError, ValueError):
        limit = 500
    key, path = resolve_log_source(source)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if key == "all":
        webui_lines = tail_text_lines(WEBUI_LOG_PATH, limit)
        upload_lines = tail_text_lines(UPLOAD_LOG_PATH, limit)
        parts = []
        parts.append("===== webui.log (%s 行) =====" % len(webui_lines))
        parts.extend(webui_lines or ["(空)"])
        parts.append("")
        parts.append("===== upload.log (%s 行) =====" % len(upload_lines))
        parts.extend(upload_lines or ["(空)"])
        text = "\n".join(parts)
        return {
            "ok": True,
            "source": "all",
            "path": "%s | %s" % (WEBUI_LOG_PATH, UPLOAD_LOG_PATH),
            "limit": limit,
            "count": len(webui_lines) + len(upload_lines),
            "updated_at": stamp,
            "lines": parts,
            "text": text,
        }
    lines = tail_text_lines(path, limit)
    text = "\n".join(lines) if lines else "(空日志) %s" % path
    return {
        "ok": True,
        "source": key,
        "path": str(path),
        "limit": limit,
        "count": len(lines),
        "updated_at": stamp,
        "lines": lines,
        "text": text,
        "exists": path.is_file(),
        "size": path.stat().st_size if path.is_file() else 0,
    }


def follow_log(log_path, done):
    while True:
        yield read_log(log_path)
        sleep(1)
        if done[0]:
            break
    yield read_log(log_path)


def preprocess_dataset(trainset_dir, exp_name, sr, n_p):
    try:
        folder = exp_path(exp_name)
    except ValueError as exc:
        yield str(exc)
        return
    trainset_text = str(trainset_dir or "").strip()
    if not trainset_text:
        yield "请输入模型名称"
        return
    trainset = Path(trainset_text)
    try:
        resolved = trainset.resolve()
    except OSError:
        yield "训练文件夹不存在: %s" % trainset
        return
    if resolved == DATASET_DIR.resolve():
        yield "请使用模型自己的音频目录，不能直接处理数据集根目录"
        return
    if not resolved.is_dir():
        yield "训练文件夹不存在: %s" % resolved
        return
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "preprocess.log"
    log_path.write_text("", encoding="utf-8")
    process = spawn(
        [
            PYTHON,
            "infer/modules/train/preprocess.py",
            str(resolved),
            str(SR_DICT[sr]),
            str(int(n_p)),
            str(folder),
            "False",
            "%.1f" % PREPROCESS_PER,
        ]
    )
    begin_job("preprocess", exp_name, process, log_path, "数据预处理")
    yield from follow_log(log_path, wait_process(process))
    final_text = read_log(log_path)
    write_job_session(
        "preprocess",
        exp_name,
        [],
        log_path,
        label="数据预处理",
        done=True,
        message=("数据预处理已结束\n" + final_text).strip(),
    )
    write_ui_session(exp_name, note="preprocess_done", info1=("数据预处理已结束\n" + final_text).strip())


def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_name, version, gpus_rmvpe):
    try:
        folder = exp_path(exp_name)
    except ValueError as exc:
        yield str(exc)
        return
    folder.mkdir(parents=True, exist_ok=True)
    log_path = folder / "extract_f0_feature.log"
    log_path.write_text("", encoding="utf-8")
    gpu_list = [item for item in str(gpus).split("-") if item != ""]
    if not gpu_list:
        gpu_list = ["0"]

    if_f0 = as_bool(if_f0)
    if if_f0:
        if f0method != "rmvpe_gpu":
            process = spawn(
                [
                    PYTHON,
                    "infer/modules/train/extract/extract_f0_print.py",
                    str(folder),
                    str(int(n_p)),
                    f0method,
                ]
            )
            begin_job("extract_f0", exp_name, process, log_path, "音高提取")
            done = wait_process(process)
        else:
            rmvpe_gpus = [item for item in str(gpus_rmvpe).split("-") if item not in {"", "-"}]
            if not rmvpe_gpus:
                rmvpe_gpus = gpu_list
            processes = []
            for index, gpu in enumerate(rmvpe_gpus):
                processes.append(
                    spawn(
                        [
                            PYTHON,
                            "infer/modules/train/extract/extract_f0_rmvpe.py",
                            str(len(rmvpe_gpus)),
                            str(index),
                            gpu,
                            str(folder),
                            "True" if IS_HALF else "False",
                        ]
                    )
                )
            begin_job("extract_f0", exp_name, processes, log_path, "音高提取")
            done = wait_processes(processes)
        yield from follow_log(log_path, done)
        log_path.write_text(read_log(log_path), encoding="utf-8")
        write_job_session(
            "extract_f0",
            exp_name,
            [],
            log_path,
            label="音高提取",
            done=True,
            message=("音高提取已结束\n" + read_log(log_path)).strip(),
        )

    processes = []
    for index, gpu in enumerate(gpu_list):
        processes.append(
            spawn(
                [
                    PYTHON,
                    "infer/modules/train/extract_feature_print.py",
                    "cuda:0" if torch.cuda.is_available() else "cpu",
                    str(len(gpu_list)),
                    str(index),
                    gpu,
                    str(folder),
                    version,
                    "True" if IS_HALF else "False",
                ]
            )
        )
    begin_job("extract_feature", exp_name, processes, log_path, "特征提取")
    yield from follow_log(log_path, wait_processes(processes))
    final_text = read_log(log_path)
    write_job_session(
        "extract_feature",
        exp_name,
        [],
        log_path,
        label="特征提取",
        done=True,
        message=("特征提取已结束\n" + final_text).strip(),
    )
    write_ui_session(exp_name, note="extract_done", info2=("特征提取已结束\n" + final_text).strip())


def get_pretrained_models(path_str, f0_str, sr):
    generator = ROOT / ("assets/pretrained%s/%sG%s.pth" % (path_str, f0_str, sr))
    discriminator = ROOT / ("assets/pretrained%s/%sD%s.pth" % (path_str, f0_str, sr))
    return (
        generator.relative_to(ROOT).as_posix() if generator.is_file() else "",
        discriminator.relative_to(ROOT).as_posix() if discriminator.is_file() else "",
    )


def change_sr2(sr, if_f0, version):
    path_str = "" if version == "v1" else "_v2"
    f0_str = "f0" if as_bool(if_f0) else ""
    return get_pretrained_models(path_str, f0_str, sr)


def change_version19(sr, if_f0, version):
    path_str = "" if version == "v1" else "_v2"
    f0_str = "f0" if as_bool(if_f0) else ""
    if sr == "32k" and version == "v1":
        sr = "40k"
    sr_choices = ["40k", "48k"] if version == "v1" else ["40k", "48k", "32k"]
    g_path, d_path = get_pretrained_models(path_str, f0_str, sr)
    return (
        g_path,
        d_path,
        gr.Radio.update(choices=sr_choices, value=sr),
    )


def change_f0(if_f0, sr, version):
    path_str = "" if version == "v1" else "_v2"
    is_f0 = as_bool(if_f0)
    g_path, d_path = get_pretrained_models(path_str, "f0" if is_f0 else "", sr)
    return (
        gr.Radio.update(visible=is_f0),
        gr.Textbox.update(visible=is_f0),
        g_path,
        d_path,
    )


def change_f0_method(f0method):
    return gr.Textbox.update(visible=f0method == "rmvpe_gpu")


def click_train(
    exp_name,
    sr,
    if_f0,
    spk_id,
    save_epoch,
    total_epoch,
    batch_size,
    if_save_latest,
    pretrained_g,
    pretrained_d,
    gpus,
    if_cache_gpu,
    if_save_every_weights,
    version,
):
    if train_process() is not None:
        yield "已有训练在进行。可以暂停，或终止后再开始新的训练", "running"
        return
    try:
        folder = exp_path(exp_name)
    except ValueError as exc:
        yield str(exc), "hidden"
        return
    if_f0 = as_bool(if_f0)
    gt_dir = folder / "0_gt_wavs"
    feature_dir = folder / ("3_feature256" if version == "v1" else "3_feature768")
    if not gt_dir.is_dir() or not feature_dir.is_dir():
        yield "请先完成数据处理和特征提取", "hidden"
        return
    if if_f0:
        f0_dir = folder / "2a_f0"
        f0nsf_dir = folder / "2b-f0nsf"
        if not f0_dir.is_dir() or not f0nsf_dir.is_dir():
            yield "请先提取音高", "hidden"
            return
        names = (
            {path.stem.split(".")[0] for path in gt_dir.iterdir()}
            & {path.stem.split(".")[0] for path in feature_dir.iterdir()}
            & {path.name.split(".")[0] for path in f0_dir.iterdir()}
            & {path.name.split(".")[0] for path in f0nsf_dir.iterdir()}
        )
    else:
        names = {path.stem.split(".")[0] for path in gt_dir.iterdir()} & {
            path.stem.split(".")[0] for path in feature_dir.iterdir()
        }
    if not names:
        yield "实验目录里没有可对齐的训练样本", "hidden"
        return
    lines = []
    for name in names:
        if if_f0:
            lines.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_dir.as_posix(),
                    name,
                    feature_dir.as_posix(),
                    name,
                    (folder / "2a_f0").as_posix(),
                    name,
                    (folder / "2b-f0nsf").as_posix(),
                    name,
                    int(spk_id),
                )
            )
        else:
            lines.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (gt_dir.as_posix(), name, feature_dir.as_posix(), name, int(spk_id))
            )
    fea_dim = 256 if version == "v1" else 768
    mute_wav = MUTE_DIR / "0_gt_wavs" / ("mute%s.wav" % sr)
    mute_fea = MUTE_DIR / ("3_feature%s" % fea_dim) / "mute.npy"
    if not mute_wav.is_file() or not mute_fea.is_file():
        yield "缺少静音样本 %s" % MUTE_DIR, "hidden"
        return
    for _ in range(2):
        if if_f0:
            lines.append(
                "%s|%s|%s|%s|%s"
                % (
                    mute_wav.as_posix(),
                    mute_fea.as_posix(),
                    (MUTE_DIR / "2a_f0" / "mute.wav.npy").as_posix(),
                    (MUTE_DIR / "2b-f0nsf" / "mute.wav.npy").as_posix(),
                    int(spk_id),
                )
            )
        else:
            lines.append("%s|%s|%s" % (mute_wav.as_posix(), mute_fea.as_posix(), int(spk_id)))
    shuffle(lines)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")

    config_save = folder / "config.json"
    if not config_save.exists():
        payload = json.loads(json.dumps(JSON_CONFIG[config_key(version, sr)]))
        if not IS_HALF:
            payload["train"]["fp16_run"] = False
        config_save.write_text(
            json.dumps(payload, ensure_ascii=False, indent=4, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    cmd = [
        PYTHON,
        "infer/modules/train/train.py",
        "-e",
        safe_exp_name(exp_name),
        "-sr",
        sr,
        "-f0",
        "1" if if_f0 else "0",
        "-bs",
        str(int(batch_size)),
        "-g",
        gpus.strip() or "0",
        "-te",
        str(int(total_epoch)),
        "-se",
        str(int(save_epoch)),
        "-l",
        "1" if if_save_latest == i18n("是") else "0",
        "-c",
        "1" if if_cache_gpu == i18n("是") else "0",
        "-sw",
        "1" if if_save_every_weights == i18n("是") else "0",
        "-v",
        version,
    ]
    if pretrained_g:
        cmd.extend(["-pg", pretrained_g])
    if pretrained_d:
        cmd.extend(["-pd", pretrained_d])
    with TRAIN_LOCK:
        if train_process() is not None:
            yield "已有训练在进行。可以暂停，或终止后再开始新的训练", "running"
            return
        clear_train_flags(folder)
        progress_path = folder / "train_progress.txt"
        progress_path.write_text("", encoding="utf-8")
        process = spawn(cmd)
        TRAIN_STATE["process"] = process
        TRAIN_STATE["folder"] = folder
        TRAIN_STATE["exp_name"] = safe_exp_name(exp_name)
        TRAIN_STATE["stop_requested"] = False
        write_train_session(TRAIN_STATE["exp_name"], folder, "running", process.pid)
    for message, phase in follow_train_output(process, folder):
        yield message, phase


def control_updates(phase):
    return (
        {"__type__": "update", "visible": phase == "running"},
        {"__type__": "update", "visible": phase == "paused"},
        {"__type__": "update", "visible": phase in {"running", "paused"}},
    )


def pack_status(message, phase):
    return (message,) + control_updates(phase)


def pack_restore(exp_name, message, phase):
    return (exp_name or "", message) + control_updates(phase)


def click_train_button(*args):
    for message, phase in click_train(*args):
        yield pack_status(message, phase)


def click_pause_train():
    with TRAIN_LOCK:
        process = train_process()
        folder = TRAIN_STATE["folder"]
        if process is None or folder is None:
            return pack_status("当前没有正在进行的训练", "hidden")
        if (folder / STOP_FILE).is_file():
            return pack_status("训练正在终止", "running")
        (folder / PAUSE_FILE).write_text("pause\n", encoding="utf-8")
        write_train_session(
            TRAIN_STATE.get("exp_name") or folder.name, folder, "paused", process.pid
        )
        message = (
            "训练已暂停。当前这一步跑完后会停住，点「继续训练」恢复\n\n"
            + format_train_message(folder, prefix="训练已暂停")
        )
        return pack_status(message, "paused")


def click_resume_train():
    with TRAIN_LOCK:
        process = train_process()
        folder = TRAIN_STATE["folder"]
        if process is None or folder is None:
            yield pack_status("当前没有正在进行的训练", "hidden")
            return
        pause_path = folder / PAUSE_FILE
        if pause_path.is_file():
            pause_path.unlink()
        write_train_session(
            TRAIN_STATE.get("exp_name") or folder.name, folder, "running", process.pid
        )
    yield pack_status(
        "训练已继续\n\n" + format_train_message(folder, prefix="训练已继续"),
        "running",
    )
    for message, phase in follow_train_output(process, folder, prefix="训练进行中"):
        yield pack_status(message, phase)


def click_stop_train():
    with TRAIN_LOCK:
        process = train_process()
        folder = TRAIN_STATE["folder"]
        if process is None or folder is None:
            return pack_status("当前没有正在进行的训练", "hidden")
        pause_path = folder / PAUSE_FILE
        if pause_path.is_file():
            pause_path.unlink()
        (folder / STOP_FILE).write_text("stop\n", encoding="utf-8")
        TRAIN_STATE["stop_requested"] = True
        pid = process.pid
        write_train_session(
            TRAIN_STATE.get("exp_name") or folder.name, folder, "running", pid
        )
    threading.Thread(
        target=force_stop_later, args=(process, pid, 60), daemon=True
    ).start()
    return pack_status(
        "已发送终止指令。当前这一步结束后会保存权重并退出；若一分钟内还没退出，会强制结束\n\n"
        + format_train_message(folder, prefix="正在终止训练"),
        "hidden",
    )


def get_train_status_payload():
    ensure_train_attached()
    process = train_process()
    folder = TRAIN_STATE.get("folder")
    exp_name = TRAIN_STATE.get("exp_name") or ""
    if process is not None and folder is not None:
        if not exp_name:
            exp_name = folder.name
        phase = train_phase_for_folder(folder)
        prefix = "训练已暂停" if phase == "paused" else "训练进行中"
        write_train_session(exp_name, folder, phase, getattr(process, "pid", None))
        return {
            "ok": True,
            "running": True,
            "exp_name": exp_name,
            "phase": phase,
            "message": format_train_message(folder, prefix=prefix + "（已自动回显）"),
            "folder": str(folder),
            "pid": getattr(process, "pid", None),
        }
    session = read_train_session() or {}
    folder = Path(session["folder"]) if session.get("folder") else discover_active_train_folder()
    exp_name = session.get("exp_name") or (folder.name if folder else "")
    progress = ""
    if folder is not None and (folder / "train_progress.txt").is_file():
        progress = "\n".join(tail_text_lines(folder / "train_progress.txt", 30))
    if folder is not None and progress:
        done_msg = session.get("message") or ""
        if session.get("phase") == "done":
            message = done_msg or ("上次训练已结束\n" + progress)
            phase = "hidden"
        else:
            message = (
                "页面已刷新。未接管到运行中的训练进程，下面是磁盘上的最近进度：\n"
                + progress
            )
            phase = "hidden"
        if WEBUI_LOG_PATH.is_file():
            message += "\n\n" + read_webui_log_view(60)
        return {
            "ok": True,
            "running": False,
            "exp_name": exp_name,
            "phase": phase,
            "message": message.strip(),
            "folder": str(folder) if folder else "",
            "pid": None,
        }
    return {
        "ok": True,
        "running": False,
        "exp_name": exp_name or "",
        "phase": "hidden",
        "message": "",
        "folder": "",
        "pid": None,
    }


def restore_train_ui():
    # 不静默回填。刷新/重开由前端横幅询问后点隐藏按钮走 restore_all_ui。
    return pack_restore("", "", "hidden")


def on_model_name_change(
    name,
    version19,
    sr2,
    if_f0_3,
    np7,
    trainset_dir4,
    spk_id5,
    gpus6,
    f0method8,
    gpus_rmvpe,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    if_cache_gpu17,
    if_save_every_weights18,
    pretrained_g14,
    pretrained_d15,
    gpus16,
    info1,
    info2,
    info3,
):
    path, summary = preview_dataset_dir(name)
    save_full_ui_session(
        name,
        version19,
        sr2,
        if_f0_3,
        np7,
        path or trainset_dir4,
        spk_id5,
        gpus6,
        f0method8,
        gpus_rmvpe,
        save_epoch10,
        total_epoch11,
        batch_size12,
        if_save_latest13,
        if_cache_gpu17,
        if_save_every_weights18,
        pretrained_g14,
        pretrained_d15,
        gpus16,
        info1,
        info2,
        info3,
    )
    return path, summary


def train_index(exp_name, version):
    try:
        folder = exp_path(exp_name)
    except ValueError as exc:
        yield str(exc)
        return
    feature_dir = folder / ("3_feature256" if version == "v1" else "3_feature768")
    if not feature_dir.is_dir() or not any(feature_dir.iterdir()):
        yield "请先进行特征提取!"
        return
    infos = []
    arrays = [np.load(path) for path in sorted(feature_dir.iterdir()) if path.suffix == ".npy"]
    if not arrays:
        yield "请先进行特征提取!"
        return
    big_npy = np.concatenate(arrays, 0)
    order = np.arange(big_npy.shape[0])
    np.random.shuffle(order)
    big_npy = big_npy[order]
    if big_npy.shape[0] > 2e5:
        infos.append("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        yield "\n".join(infos)
        try:
            big_npy = (
                MiniBatchKMeans(
                    n_clusters=10000,
                    verbose=True,
                    batch_size=256 * N_CPU,
                    compute_labels=False,
                    init="random",
                )
                .fit(big_npy)
                .cluster_centers_
            )
        except Exception:
            infos.append(traceback.format_exc())
            yield "\n".join(infos)
    np.save(folder / "total_fea.npy", big_npy)
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), max(big_npy.shape[0] // 39, 1))
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    dim = 256 if version == "v1" else 768
    index = faiss.index_factory(dim, "IVF%s,Flat" % n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    infos.append("adding")
    yield "\n".join(infos)
    for start in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[start : start + 8192])
    index_name = "%s.index" % safe_exp_name(exp_name)
    index_path = folder / index_name
    faiss.write_index(index, str(index_path))
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    index_copy = INDEX_DIR / index_name
    shutil.copy2(index_path, index_copy)
    infos.append("成功构建索引 %s" % index_name)
    infos.append("索引副本: %s" % index_copy)
    yield "\n".join(infos)


def train1key(
    exp_name,
    sr,
    if_f0,
    trainset_dir,
    spk_id,
    n_p,
    f0method,
    save_epoch,
    total_epoch,
    batch_size,
    if_save_latest,
    pretrained_g,
    pretrained_d,
    gpus,
    if_cache_gpu,
    if_save_every_weights,
    version,
    gpus_rmvpe,
):
    yield pack_status(i18n("step1:正在处理数据"), "hidden")
    for chunk in preprocess_dataset(trainset_dir, exp_name, sr, n_p):
        yield pack_status(i18n("step1:正在处理数据") + "\n" + chunk, "hidden")
    yield pack_status(i18n("step2:正在提取音高&正在提取特征"), "hidden")
    for chunk in extract_f0_feature(gpus, n_p, f0method, if_f0, exp_name, version, gpus_rmvpe):
        yield pack_status(i18n("step2:正在提取音高&正在提取特征") + "\n" + chunk, "hidden")
    for message, phase in click_train(
        exp_name,
        sr,
        if_f0,
        spk_id,
        save_epoch,
        total_epoch,
        batch_size,
        if_save_latest,
        pretrained_g,
        pretrained_d,
        gpus,
        if_cache_gpu,
        if_save_every_weights,
        version,
    ):
        yield pack_status(i18n("step3a:正在训练模型") + "\n" + message, phase)
    for chunk in train_index(exp_name, version):
        yield pack_status(chunk, "hidden")
    yield pack_status(
        "%s\n完整模型 %s.pth\n索引 %s.index\n可在页面下方下载"
        % (
            i18n("全流程结束！"),
            safe_exp_name(exp_name),
            safe_exp_name(exp_name),
        ),
        "hidden",
    )


def asset_status():
    required = [
        ROOT / "assets" / "hubert" / "hubert_base.pt",
        ROOT / "assets" / "rmvpe" / "rmvpe.pt",
        MUTE_DIR / "0_gt_wavs" / "mute40k.wav",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        return "缺少训练资源: " + ", ".join(missing)
    return "训练资源已在当前目录内。数据集目录 %s ，实验输出 %s 。" % (DATASET_DIR, LOG_DIR)


def build_theme():
    slate = "#1e293b"
    border = "#e2e8f0"
    page = "#f8fafc"
    ink = "#0f172a"
    return gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        font=["Inter", "Segoe UI", "PingFang SC", "Microsoft YaHei", "sans-serif"],
    ).set(
        body_background_fill=page,
        body_background_fill_dark=page,
        body_text_color=ink,
        body_text_color_dark=ink,
        body_text_color_subdued="#64748b",
        body_text_color_subdued_dark="#64748b",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary=page,
        background_fill_secondary_dark=page,
        border_color_primary=border,
        border_color_primary_dark=border,
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color=border,
        block_border_color_dark=border,
        block_label_background_fill="#ffffff",
        block_label_background_fill_dark="#ffffff",
        block_label_text_color=ink,
        block_label_text_color_dark=ink,
        block_title_text_color=ink,
        block_title_text_color_dark=ink,
        block_radius="10px",
        button_primary_background_fill=slate,
        button_primary_background_fill_dark=slate,
        button_primary_background_fill_hover="#334155",
        button_primary_background_fill_hover_dark="#334155",
        button_primary_text_color="#f8fafc",
        button_primary_text_color_dark="#f8fafc",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_text_color=ink,
        button_secondary_text_color_dark=ink,
        button_secondary_border_color=border,
        button_secondary_border_color_dark=border,
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color=border,
        input_border_color_dark=border,
        input_border_color_focus="#94a3b8",
        input_border_color_focus_dark="#94a3b8",
        slider_color=slate,
        slider_color_dark=slate,
        color_accent=slate,
        color_accent_soft="#f1f5f9",
        color_accent_soft_dark="#f1f5f9",
    )


def build_app():
    css = (ROOT / "webui.css").read_text(encoding="utf-8")
    with gr.Blocks(title="有品音色训练", theme=build_theme(), css=css) as app:
        gr.HTML(
            """
            <div class="yp-head">
              <div class="yp-brand">
                <div>
                  <div class="yp-brand-title">
                    <h1>有品音色训练</h1>
                    <span class="yp-brand-badge">RVC v2 / v1</span>
                  </div>
                  <p class="yp-head-sub">基于检索增强变分自编码器的轻量音色训练工作台</p>
                </div>
              </div>
              <div class="yp-status">
                <span class="yp-status-icon">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                  </svg>
                </span>
                <span class="yp-status-text">%s</span>
              </div>
            </div>

            <div class="yp-disclaimer-card">
              <div class="yp-disclaimer-header">
                <div class="yp-disclaimer-title">
                  <span class="yp-disclaimer-icon">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                      <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                      <line x1="12" y1="8" x2="12" y2="12"/>
                      <line x1="12" y1="16" x2="12.01" y2="16"/>
                    </svg>
                  </span>
                  <span class="yp-disclaimer-badge">免责声明</span>
                  <span class="yp-disclaimer-sub">Open Source License & Terms of Use</span>
                </div>
                <a href="#license" class="yp-disclaimer-link">LICENSE</a>
              </div>
              <div class="yp-disclaimer-content">
                本软件以MIT协议开源, 作者不对软件具备任何控制力, 使用软件者、传播软件导出的声音者自负全责。<br>
                如不认可该条款, 则不能使用或引用软件包内任何代码和文件。详见根目录 <code>LICENSE</code>。
              </div>
            </div>
            """
            % html.escape(asset_status())
        )

        with gr.Box(elem_classes=["yp-card"]):
            gr.HTML(
                """
                <div class="yp-step-header">
                  <div class="yp-step-badge"><span>1</span></div>
                  <div class="yp-step-title-wrap">
                    <div class="yp-step-title-line">
                      <span class="yp-step-icon">
                        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <line x1="4" x2="20" y1="21" y2="21"/>
                          <line x1="4" x2="20" y1="3" y2="3"/>
                          <line x1="4" x2="20" y1="12" y2="12"/>
                          <circle cx="8" cy="12" r="3"/>
                          <circle cx="16" cy="3" r="3"/>
                          <circle cx="12" cy="21" r="3"/>
                        </svg>
                      </span>
                      <span class="yp-step-title">全局基础配置</span>
                    </div>
                    <div class="yp-step-desc">先填写模型名称，再上传音频。版本、采样率和音高也在这里设定</div>
                  </div>
                </div>
                """
            )
            with gr.Row():
                exp_dir1 = gr.Textbox(
                    label="模型名称",
                    value="",
                    info="默认为空。每个音色单独存放，避免不同音频进同一个目录",
                    placeholder="请输入模型名称",
                    elem_id="model-name",
                    scale=2,
                )
                version19 = gr.Radio(
                    label=i18n("版本"),
                    choices=["v1", "v2"],
                    value="v2",
                    info="推荐 v2，音质表现更细腻",
                    interactive=True,
                    scale=1,
                )
                sr2 = gr.Radio(
                    label=i18n("目标采样率"),
                    choices=["40k", "48k"],
                    value="40k",
                    info="通常推荐 40k 或 48k 高保真",
                    interactive=True,
                    scale=1,
                )
                if_f0_3 = gr.Radio(
                    label=i18n("模型是否带音高指导"),
                    choices=[True, False],
                    value=True,
                    info="唱歌模型必开；纯配音或语聊可按需关闭",
                    interactive=True,
                    scale=2,
                )
                np7 = gr.Slider(
                    minimum=1,
                    maximum=N_CPU,
                    step=1,
                    label=i18n("CPU 进程数"),
                    info="提取音高与数据处理所占用的多进程数",
                    value=max(int(np.ceil(N_CPU / 1.5)), 1),
                    interactive=True,
                    scale=2,
                )

        with gr.Box(elem_classes=["yp-card"]):
            gr.HTML(
                """
                <div class="yp-step-header">
                  <div class="yp-step-badge"><span>2</span></div>
                  <div class="yp-step-title-wrap">
                    <div class="yp-step-title-line">
                      <span class="yp-step-icon">
                        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <path d="M3 10v4"/>
                          <path d="M7 6v12"/>
                          <path d="M11 3v18"/>
                          <path d="M15 8v8"/>
                          <path d="M19 11v2"/>
                        </svg>
                      </span>
                      <span class="yp-step-title">数据预处理</span>
                    </div>
                    <div class="yp-step-desc">按模型名称建立目录，可一次上传多个音频，再切片、归一化并重采样</div>
                  </div>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML(
                        '<iframe class="yp-upload-frame" src="dataset-upload-ui?v=6" title="上传音频"></iframe>',
                        elem_id="dataset-upload",
                        show_label=False,
                    )
                    refresh_audio_dir = gr.Button("刷新音频目录", elem_id="refresh-audio-dir")
                    but_clear_audio = gr.Button(
                        "清空音频目录",
                        variant="secondary",
                        elem_classes=["yp-ctrl-btn", "yp-btn-clear"],
                    )
                    trainset_dir4 = gr.Textbox(
                        label=i18n("输入训练文件夹路径"),
                        value="",
                        info="由模型名称生成。上传和清空都作用在这个目录",
                        placeholder="填写模型名称后自动出现",
                        elem_id="trainset-dir",
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("请指定说话人id"),
                        info="单人音色训练默认填 0 即可",
                        value=0,
                        interactive=True,
                    )
                    but1 = gr.Button(i18n("处理数据"), variant="primary", elem_classes=["yp-action-btn", "yp-btn-preprocess"])
                with gr.Column(scale=2):
                    upload_info = gr.Textbox(
                        label="音频目录",
                        value="请输入模型名称",
                        lines=8,
                        max_lines=16,
                        interactive=False,
                        elem_id="upload-info",
                    )
                    info1 = gr.Textbox(
                        label=i18n("输出信息"),
                        value="",
                        lines=6,
                        max_lines=12,
                        elem_id="preprocess-output",
                    )
            refresh_audio_dir.click(preview_dataset_dir, [exp_dir1], [trainset_dir4, upload_info])
            but_clear_audio.click(clear_dataset_audio, [exp_dir1], [trainset_dir4, upload_info])
            but1.click(
                preprocess_dataset,
                [trainset_dir4, exp_dir1, sr2, np7],
                [info1],
            )

        with gr.Box(elem_classes=["yp-card"]):
            gr.HTML(
                """
                <div class="yp-step-header">
                  <div class="yp-step-badge"><span>3</span></div>
                  <div class="yp-step-title-wrap">
                    <div class="yp-step-title-line">
                      <span class="yp-step-icon">
                        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <circle cx="6" cy="6" r="3"/>
                          <circle cx="6" cy="18" r="3"/>
                          <circle cx="18" cy="12" r="3"/>
                          <line x1="8.5" y1="7.5" x2="15.5" y2="10.5"/>
                          <line x1="8.5" y1="16.5" x2="15.5" y2="13.5"/>
                        </svg>
                      </span>
                      <span class="yp-step-title">音高与特征提取</span>
                    </div>
                    <div class="yp-step-desc">提取基频 F0 音高特征与 Hubert 语义高维特征向量，支持多进程与 GPU 加速</div>
                  </div>
                </div>
                """
            )
            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        gpus6 = gr.Textbox(
                            label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value=GPUS,
                            info="提取特征使用的 GPU 卡号",
                            interactive=True,
                            scale=1,
                        )
                        f0method8 = gr.Radio(
                            label=i18n("音高提取算法"),
                            choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                            value="rmvpe_gpu",
                            info="歌声推荐 pm/rmvpe_gpu，语聊可 dio，高质量选 harvest",
                            interactive=True,
                            scale=2,
                        )
                    with gr.Row():
                        gpu_info_box = gr.Textbox(
                            label=i18n("显卡信息"),
                            value=GPU_INFO,
                            lines=2,
                            info="当前系统检测到的可用显卡硬件状态",
                            interactive=False,
                            scale=1,
                        )
                        gpus_rmvpe = gr.Textbox(
                            label=i18n("RMVPE 卡号配置"),
                            info="例如 0-0-1 代表卡0分配2个进程、卡1分配1个进程",
                            value="%s-%s" % (GPUS, GPUS),
                            interactive=True,
                            scale=2,
                        )
                    but2 = gr.Button(i18n("特征提取"), variant="primary", elem_classes=["yp-action-btn", "yp-btn-extract"])
                with gr.Column(scale=2):
                    info2 = gr.Textbox(
                        label=i18n("输出信息"),
                        value="",
                        lines=7,
                        max_lines=14,
                        elem_id="extract-output",
                    )

            f0method8.change(change_f0_method, [f0method8], [gpus_rmvpe])
            but2.click(
                extract_f0_feature,
                [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, gpus_rmvpe],
                [info2],
            )

        with gr.Box(elem_classes=["yp-card"]):
            gr.HTML(
                """
                <div class="yp-step-header">
                  <div class="yp-step-badge"><span>4</span></div>
                  <div class="yp-step-title-wrap">
                    <div class="yp-step-title-line">
                      <span class="yp-step-icon">
                        <svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                          <rect width="16" height="16" x="4" y="4" rx="2"/>
                          <rect width="6" height="6" x="9" y="9" rx="1"/>
                          <path d="M15 2v2"/>
                          <path d="M15 20v2"/>
                          <path d="M2 15h2"/>
                          <path d="M2 9h2"/>
                          <path d="M20 15h2"/>
                          <path d="M20 9h2"/>
                          <path d="M9 2v2"/>
                          <path d="M9 20v2"/>
                        </svg>
                      </span>
                      <span class="yp-step-title">模型与索引训练</span>
                    </div>
                    <div class="yp-step-desc">配置超参数进行生成对抗网络迭代，并构建用于音色相似检索的 IVF 特征索引</div>
                  </div>
                </div>
                """
            )
            with gr.Row():
                save_epoch10 = gr.Slider(
                    minimum=1,
                    maximum=50,
                    step=1,
                    label=i18n("保存频率 (Epoch)"),
                    info="每隔指定轮数自动保存一次权重模型",
                    value=5,
                    interactive=True,
                )
                total_epoch11 = gr.Slider(
                    minimum=2,
                    maximum=1000,
                    step=1,
                    label=i18n("总训练轮数 (Epoch)"),
                    info="整个数据集完整迭代学习的总次数",
                    value=20,
                    interactive=True,
                )
                batch_size12 = gr.Slider(
                    minimum=1,
                    maximum=40,
                    step=1,
                    label=i18n("每张显卡的 batch_size"),
                    info="单卡批大小，根据显存容量合理设置",
                    value=DEFAULT_BATCH,
                    interactive=True,
                )
            with gr.Row():
                if_save_latest13 = gr.Radio(
                    label=i18n("仅保存最新检查点"),
                    choices=[i18n("是"), i18n("否")],
                    value=i18n("否"),
                    info="是否仅保存最新的 ckpt 文件以显著节省硬盘空间",
                    interactive=True,
                )
                if_cache_gpu17 = gr.Radio(
                    label=i18n("训练集缓存至显存"),
                    choices=[i18n("是"), i18n("否")],
                    value=i18n("否"),
                    info="10分钟以下小音频建议开启加速；大数据易爆显存",
                    interactive=True,
                )
                if_save_every_weights18 = gr.Radio(
                    label=i18n("保存中间提取模型"),
                    choices=[i18n("是"), i18n("否")],
                    value=i18n("是"),
                    info="在每次保存点将最终推理小模型同步保存到 weights 与 assets/weights",
                    interactive=True,
                )
            with gr.Row():
                pretrained_g14 = gr.Textbox(
                    label=i18n("加载预训练底模G路径"),
                    value="assets/pretrained_v2/f0G40k.pth",
                    info="生成器 Generator 预训练权重路径",
                    interactive=True,
                )
                pretrained_d15 = gr.Textbox(
                    label=i18n("加载预训练底模D路径"),
                    value="assets/pretrained_v2/f0D40k.pth",
                    info="判别器 Discriminator 预训练权重路径",
                    interactive=True,
                )
                gpus16 = gr.Textbox(
                    label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                    value=GPUS,
                    info="执行对抗训练使用的显卡卡号",
                    interactive=True,
                )

            sr2.change(change_sr2, [sr2, if_f0_3, version19], [pretrained_g14, pretrained_d15])
            version19.change(
                change_version19,
                [sr2, if_f0_3, version19],
                [pretrained_g14, pretrained_d15, sr2],
            )
            if_f0_3.change(
                change_f0,
                [if_f0_3, sr2, version19],
                [f0method8, gpus_rmvpe, pretrained_g14, pretrained_d15],
            )

            with gr.Row():
                with gr.Column(scale=3):
                    with gr.Row():
                        but3 = gr.Button(i18n("训练模型"), variant="primary", elem_classes=["yp-action-btn", "yp-btn-train"])
                        but4 = gr.Button(i18n("训练特征索引"), variant="primary", elem_classes=["yp-action-btn", "yp-btn-index"])
                    with gr.Row():
                        but_pause = gr.Button(
                            "暂停训练",
                            variant="secondary",
                            visible=False,
                            elem_id="btn-train-pause",
                            elem_classes=["yp-ctrl-btn", "yp-btn-pause"],
                        )
                        but_resume = gr.Button(
                            "继续训练",
                            variant="secondary",
                            visible=False,
                            elem_id="btn-train-resume",
                            elem_classes=["yp-ctrl-btn", "yp-btn-resume"],
                        )
                        but_stop = gr.Button(
                            "终止训练",
                            variant="secondary",
                            visible=False,
                            elem_id="btn-train-stop",
                            elem_classes=["yp-ctrl-btn", "yp-btn-stop"],
                        )
                        but_view_log = gr.Button(
                            "查看日志",
                            variant="secondary",
                            elem_id="btn-train-log",
                            elem_classes=["yp-ctrl-btn", "yp-btn-log"],
                        )
                    with gr.Row():
                        but5 = gr.Button(i18n("一键训练"), variant="primary", elem_classes=["yp-primary-accent-btn", "yp-btn-train-all"])
                with gr.Column(scale=2):
                    info3 = gr.Textbox(
                        label=i18n("输出信息"),
                        value="",
                        lines=12,
                        max_lines=24,
                        elem_id="train-output",
                    )

            gr.HTML(
                """
                <div class="yp-download-head">训练结束后下载</div>
                <div class="yp-step-desc">完整模型是训练结束时保存的模型名.pth，不是中途按轮次另存的分段权重。索引文件保存为模型名.index。</div>
                """
            )
            with gr.Row():
                model_download = gr.File(label="完整模型", interactive=False)
                index_download = gr.File(label="索引文件", interactive=False)
            with gr.Row():
                but_download = gr.Button(
                    "刷新下载",
                    variant="secondary",
                    elem_classes=["yp-ctrl-btn", "yp-btn-download"],
                )
                download_info = gr.Textbox(
                    label="下载说明",
                    value="训练或索引完成后会自动出现在这里。也可以稍后点刷新下载。",
                    lines=3,
                    max_lines=8,
                    interactive=False,
                )
            but_download.click(
                refresh_downloads,
                [exp_dir1],
                [model_download, index_download, download_info],
            )

            but3.click(
                click_train_button,
                [
                    exp_dir1,
                    sr2,
                    if_f0_3,
                    spk_id5,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    if_save_latest13,
                    pretrained_g14,
                    pretrained_d15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                ],
                [info3, but_pause, but_resume, but_stop],
            ).then(refresh_downloads, [exp_dir1], [model_download, index_download, download_info])
            but_pause.click(click_pause_train, [], [info3, but_pause, but_resume, but_stop])
            but_resume.click(click_resume_train, [], [info3, but_pause, but_resume, but_stop])
            but_stop.click(click_stop_train, [], [info3, but_pause, but_resume, but_stop])
            # 「查看日志」由 train-status-ui 拦截，新开 /logs-ui 页面
            but4.click(train_index, [exp_dir1, version19], info3).then(
                refresh_downloads, [exp_dir1], [model_download, index_download, download_info]
            )
            but5.click(
                train1key,
                [
                    exp_dir1,
                    sr2,
                    if_f0_3,
                    trainset_dir4,
                    spk_id5,
                    np7,
                    f0method8,
                    save_epoch10,
                    total_epoch11,
                    batch_size12,
                    if_save_latest13,
                    pretrained_g14,
                    pretrained_d15,
                    gpus16,
                    if_cache_gpu17,
                    if_save_every_weights18,
                    version19,
                    gpus_rmvpe,
                ],
                [info3, but_pause, but_resume, but_stop],
            ).then(refresh_downloads, [exp_dir1], [model_download, index_download, download_info])
            session_inputs = [
                exp_dir1,
                version19,
                sr2,
                if_f0_3,
                np7,
                trainset_dir4,
                spk_id5,
                gpus6,
                f0method8,
                gpus_rmvpe,
                save_epoch10,
                total_epoch11,
                batch_size12,
                if_save_latest13,
                if_cache_gpu17,
                if_save_every_weights18,
                pretrained_g14,
                pretrained_d15,
                gpus16,
                info1,
                info2,
                info3,
            ]
            session_outputs = [
                exp_dir1,
                version19,
                sr2,
                if_f0_3,
                np7,
                trainset_dir4,
                spk_id5,
                upload_info,
                info1,
                gpus6,
                f0method8,
                gpus_rmvpe,
                info2,
                save_epoch10,
                total_epoch11,
                batch_size12,
                if_save_latest13,
                if_cache_gpu17,
                if_save_every_weights18,
                pretrained_g14,
                pretrained_d15,
                gpus16,
                info3,
                but_pause,
                but_resume,
                but_stop,
                model_download,
                index_download,
                download_info,
            ]

            exp_dir1.change(
                on_model_name_change,
                session_inputs,
                [trainset_dir4, upload_info],
            )
            # 不能用 visible=False：Gradio 可能不渲染进 DOM，前端就点不到回显。
            btn_save_session = gr.Button(
                "保存会话",
                elem_id="btn-save-session",
                elem_classes=["yp-hidden-session-btn"],
            )
            btn_restore_session = gr.Button(
                "恢复会话",
                elem_id="btn-restore-session",
                elem_classes=["yp-hidden-session-btn"],
            )
            btn_save_session.click(save_full_ui_session, session_inputs, [])
            btn_restore_session.click(restore_all_ui, [], session_outputs)
            for _comp in (
                version19,
                sr2,
                if_f0_3,
                np7,
                spk_id5,
                gpus6,
                f0method8,
                gpus_rmvpe,
                save_epoch10,
                total_epoch11,
                batch_size12,
                if_save_latest13,
                if_cache_gpu17,
                if_save_every_weights18,
                pretrained_g14,
                pretrained_d15,
                gpus16,
            ):
                _comp.change(save_full_ui_session, session_inputs, [])

            app.load(
                restore_train_ui,
                [],
                [exp_dir1, info3, but_pause, but_resume, but_stop],
            )
            # Gradio 会剥离 gr.HTML 内的 script，用隐藏 iframe 注入轮询（与上传区同思路）
            gr.HTML(
                '<iframe src="train-status-ui?v=6" title="train-status" '
                'style="position:absolute;width:0;height:0;border:0;overflow:hidden"></iframe>'
            )
    return app


TRAIN_STATUS_UI_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>train-status</title></head>
<body>
<script>
(function () {
  var restoreState = { asked: false, accepted: null, asking: false };

  function fieldOf(sel) {
    try {
      var root = parent.document.querySelector(sel);
      if (!root) return null;
      return root.querySelector("textarea, input") || root;
    } catch (err) {
      return null;
    }
  }

  function setField(sel, value) {
    var field = fieldOf(sel);
    if (!field) return;
    var next = value == null ? "" : String(value);
    if (field.value === next) return;
    field.value = next;
    try {
      field.dispatchEvent(new Event("input", { bubbles: true }));
      field.dispatchEvent(new Event("change", { bubbles: true }));
    } catch (err) {}
  }

  function setBlockVisible(sel, visible) {
    try {
      var root = parent.document.querySelector(sel);
      if (!root) return;
      root.style.setProperty("display", visible ? "" : "none", "important");
    } catch (err) {}
  }

  function clickById(sel) {
    try {
      var root = parent.document.querySelector(sel);
      if (!root) return false;
      var button = root.matches("button") ? root : root.querySelector("button");
      if (!button) return false;
      button.click();
      return true;
    } catch (err) {
      return false;
    }
  }

  function removeBanner() {
    try {
      var old = parent.document.getElementById("yp-restore-banner");
      if (old && old.parentNode) old.parentNode.removeChild(old);
    } catch (err) {}
  }

  function showRestoreBanner(data) {
    removeBanner();
    var wrap = parent.document.createElement("div");
    wrap.id = "yp-restore-banner";
    wrap.setAttribute("style", [
      "position:fixed", "left:0", "right:0", "top:0", "z-index:2147483646",
      "background:#0f172a", "color:#f8fafc", "padding:14px 18px",
      "box-shadow:0 8px 24px rgba(15,23,42,.28)",
      "font:13px/1.5 Segoe UI,PingFang SC,Microsoft YaHei,sans-serif"
    ].join(";"));
    var box = parent.document.createElement("div");
    box.setAttribute("style", "max-width:1100px;margin:0 auto;display:flex;gap:16px;flex-wrap:wrap;align-items:center;justify-content:space-between");
    var left = parent.document.createElement("div");
    left.setAttribute("style", "min-width:240px;flex:1");
    var title = parent.document.createElement("div");
    title.setAttribute("style", "font-weight:700;margin-bottom:4px");
    title.textContent = "检测到上次工作记录";
    var detail = parent.document.createElement("div");
    detail.setAttribute("style", "opacity:.92;font-size:12px;white-space:pre-line");
    detail.textContent = (data.prompt_lines || []).join(String.fromCharCode(10));
    left.appendChild(title);
    left.appendChild(detail);
    var right = parent.document.createElement("div");
    right.setAttribute("style", "display:flex;gap:8px;flex:0 0 auto");
    var yesBtn = parent.document.createElement("button");
    yesBtn.type = "button";
    yesBtn.setAttribute("data-yp", "yes");
    yesBtn.setAttribute("style", "border:0;border-radius:8px;padding:8px 14px;background:#2563eb;color:#fff;font-weight:700;cursor:pointer");
    yesBtn.textContent = "回显全部";
    var noBtn = parent.document.createElement("button");
    noBtn.type = "button";
    noBtn.setAttribute("data-yp", "no");
    noBtn.setAttribute("style", "border:1px solid #64748b;border-radius:8px;padding:8px 14px;background:transparent;color:#e2e8f0;cursor:pointer");
    noBtn.textContent = "不回显";
    right.appendChild(yesBtn);
    right.appendChild(noBtn);
    box.appendChild(left);
    box.appendChild(right);
    wrap.appendChild(box);
    parent.document.body.appendChild(wrap);
    try { parent.document.body.style.paddingTop = "88px"; } catch (err) {}
    wrap.querySelector("[data-yp='yes']").addEventListener("click", function () {
      restoreState.accepted = true;
      removeBanner();
      try { parent.document.body.style.paddingTop = ""; } catch (err) {}
      if (!clickById("#btn-restore-session")) {
        setField("#model-name", data.exp_name || "");
        clickById("#refresh-audio-dir");
      }
      setTimeout(function () { clickById("#btn-restore-session"); }, 400);
      setTimeout(function () { clickById("#btn-restore-session"); }, 1200);
    });
    wrap.querySelector("[data-yp='no']").addEventListener("click", function () {
      restoreState.accepted = false;
      removeBanner();
      try { parent.document.body.style.paddingTop = ""; } catch (err) {}
    });
  }

  function maybeAskRestore() {
    if (restoreState.asked || restoreState.asking) return;
    if (!parent.document.querySelector("#model-name")) return;
    if (!parent.document.querySelector("#btn-restore-session")) return;
    restoreState.asking = true;
    fetch("ui-session", { cache: "no-store" })
      .then(function (resp) { return resp.ok ? resp.json() : null; })
      .then(function (data) {
        restoreState.asked = true;
        restoreState.asking = false;
        if (!data || !data.has_session || !data.exp_name) {
          restoreState.accepted = false;
          return;
        }
        showRestoreBanner(data);
      })
      .catch(function () {
        restoreState.asking = false;
        restoreState.asked = true;
        restoreState.accepted = false;
      });
  }

  function openLogsPage(event) {
    try {
      if (event) {
        event.preventDefault();
        event.stopPropagation();
        if (event.stopImmediatePropagation) event.stopImmediatePropagation();
      }
      var url = new URL("logs-ui", window.location.href);
      parent.window.open(url.href, "_blank", "noopener,noreferrer");
    } catch (err) {
      try { window.open("logs-ui", "_blank"); } catch (e2) {}
    }
    return false;
  }

  function bindLogButton() {
    try {
      var root = parent.document.querySelector("#btn-train-log");
      if (!root) return;
      var targets = [root];
      var btn = root.querySelector("button");
      if (btn) targets.push(btn);
      for (var i = 0; i < targets.length; i++) {
        var el = targets[i];
        if (el.dataset && el.dataset.ypLogBound === "1") continue;
        if (el.dataset) el.dataset.ypLogBound = "1";
        el.addEventListener("click", openLogsPage, true);
      }
    } catch (err) {}
  }

  function bindAutoSave() {
    try {
      var field = fieldOf("#model-name");
      if (!field || field.dataset.ypPersistBound === "1") return;
      field.dataset.ypPersistBound = "1";
      var timer = null;
      function saveNow() {
        if (!(field.value || "").trim()) return;
        clickById("#btn-save-session");
      }
      function schedule() {
        if (timer) clearTimeout(timer);
        timer = setTimeout(saveNow, 500);
      }
      field.addEventListener("input", schedule);
      field.addEventListener("change", saveNow);
      field.addEventListener("blur", saveNow);
    } catch (err) {}
  }

  function applyJobStatus(job) {
    if (!job || !job.ok) return;
    if (job.info1) setField("#preprocess-output", job.info1);
    if (job.info2) setField("#extract-output", job.info2);
    if (job.info3) setField("#train-output", job.info3);
  }

  function tick() {
    bindLogButton();
    bindAutoSave();
    maybeAskRestore();
    fetch("job-status", { cache: "no-store" })
      .then(function (resp) { return resp.ok ? resp.json() : null; })
      .then(function (job) {
        // 有任务在跑时始终回填日志；已回显后也持续跟随
        if (!job || !job.ok) return;
        if (restoreState.accepted === true || job.running || job.info1 || job.info2) {
          applyJobStatus(job);
        }
      })
      .catch(function () {});
    fetch("train-status", { cache: "no-store" })
      .then(function (resp) { return resp.ok ? resp.json() : null; })
      .then(function (data) {
        if (!data || !data.ok) return;
        var phase = data.phase || "hidden";
        if (restoreState.accepted === true || data.running) {
          if (data.message) setField("#train-output", data.message);
          setBlockVisible("#btn-train-pause", phase === "running");
          setBlockVisible("#btn-train-resume", phase === "paused");
          setBlockVisible("#btn-train-stop", phase === "running" || phase === "paused");
        }
      })
      .catch(function () {});
  }

  setInterval(function () {
    if (restoreState.accepted !== false) clickById("#btn-save-session");
  }, 15000);
  setInterval(tick, 3000);
  tick();
  setTimeout(tick, 600);
  setTimeout(tick, 1500);
  setTimeout(tick, 3000);
})();
</script>
</body>
</html>
"""


LOG_VIEWER_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>运行日志 · 有品音色训练</title>
<style>
  :root {
    --bg: #f1f5f9;
    --panel: #ffffff;
    --border: #e2e8f0;
    --text: #0f172a;
    --muted: #64748b;
    --accent: #2563eb;
    --accent-soft: #eff6ff;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; height: 100%; }
  body {
    display: flex; flex-direction: column;
    font: 13px/1.5 "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
    color: var(--text); background: var(--bg);
  }
  header {
    display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between;
    gap: 12px; padding: 14px 18px; background: var(--panel);
    border-bottom: 1px solid var(--border);
  }
  h1 { margin: 0; font-size: 16px; font-weight: 700; }
  .sub { color: var(--muted); font-size: 12px; margin-top: 2px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .tabs { display: flex; gap: 6px; }
  .tabs button, .toolbar button, .toolbar select, .toolbar label.auto {
    border: 1px solid var(--border); background: #fff; color: var(--text);
    border-radius: 8px; padding: 6px 11px; font: inherit; cursor: pointer;
  }
  .tabs button.active { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 650; }
  .toolbar button.primary { background: var(--accent); border-color: var(--accent); color: #fff; font-weight: 650; }
  .toolbar button:disabled { opacity: .45; cursor: not-allowed; }
  .toolbar label.auto { display: inline-flex; align-items: center; gap: 6px; user-select: none; }
  .toolbar select { padding-right: 8px; }
  main {
    flex: 1; min-height: 0; margin: 14px 18px 10px;
    background: var(--panel); border: 1px solid var(--border); border-radius: 12px;
    display: flex; flex-direction: column; overflow: hidden;
  }
  #log {
    flex: 1; margin: 0; padding: 14px 16px; overflow: auto;
    white-space: pre-wrap; word-break: break-word;
    font: 12px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
    color: #0f172a; background: #fcfcfd;
  }
  footer {
    padding: 8px 18px 14px; color: var(--muted); font-size: 12px;
    display: flex; flex-wrap: wrap; gap: 10px 16px; align-items: center;
  }
  #toast {
    position: fixed; right: 18px; bottom: 18px; background: #0f172a; color: #f8fafc;
    padding: 8px 12px; border-radius: 8px; font-size: 12px; opacity: 0;
    pointer-events: none; transition: opacity .2s;
  }
  #toast.show { opacity: 1; }
</style>
</head>
<body>
  <header>
    <div>
      <h1>运行日志</h1>
      <div class="sub" id="path">加载中…</div>
    </div>
    <div class="toolbar">
      <div class="tabs" id="tabs">
        <button type="button" data-source="webui" class="active">webui.log</button>
        <button type="button" data-source="upload">upload.log</button>
        <button type="button" data-source="all">全部</button>
      </div>
      <select id="limit" title="显示行数">
        <option value="200">最近 200 行</option>
        <option value="500" selected>最近 500 行</option>
        <option value="1000">最近 1000 行</option>
        <option value="2000">最近 2000 行</option>
        <option value="5000">最近 5000 行</option>
      </select>
      <label class="auto"><input type="checkbox" id="auto" checked> 自动刷新</label>
      <button type="button" id="btn-refresh">刷新</button>
      <button type="button" id="btn-copy" class="primary">复制</button>
      <button type="button" id="btn-export">导出</button>
    </div>
  </header>
  <main><pre id="log">正在读取日志…</pre></main>
  <footer>
    <span id="meta">—</span>
    <span>提示：可复制全文，或导出为 .txt 文件</span>
  </footer>
  <div id="toast"></div>
<script>
(function () {
  var source = "webui";
  var currentText = "";
  var timer = null;
  var logEl = document.getElementById("log");
  var pathEl = document.getElementById("path");
  var metaEl = document.getElementById("meta");
  var limitEl = document.getElementById("limit");
  var autoEl = document.getElementById("auto");
  var toastEl = document.getElementById("toast");
  var stickBottom = true;

  function toast(msg) {
    toastEl.textContent = msg;
    toastEl.className = "show";
    setTimeout(function () { toastEl.className = ""; }, 1600);
  }

  function setSource(next) {
    source = next;
    var buttons = document.querySelectorAll("#tabs button");
    for (var i = 0; i < buttons.length; i++) {
      buttons[i].className = buttons[i].getAttribute("data-source") === source ? "active" : "";
    }
    loadLog(true);
  }

  function loadLog(forceScroll) {
    var limit = limitEl.value || "500";
    fetch("logs-api?source=" + encodeURIComponent(source) + "&limit=" + encodeURIComponent(limit), {
      cache: "no-store"
    })
      .then(function (resp) { return resp.ok ? resp.json() : Promise.reject(new Error("HTTP " + resp.status)); })
      .then(function (data) {
        if (!data || !data.ok) throw new Error("bad payload");
        currentText = data.text || "";
        var nearBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 80;
        logEl.textContent = currentText || "(空)";
        pathEl.textContent = data.path || "";
        metaEl.textContent = "更新于 " + (data.updated_at || "-")
          + " · " + (data.count || 0) + " 行"
          + (data.size != null ? (" · " + Math.max(0, data.size) + " bytes") : "");
        if (forceScroll || (autoEl.checked && (stickBottom || nearBottom))) {
          logEl.scrollTop = logEl.scrollHeight;
        }
      })
      .catch(function (err) {
        logEl.textContent = "读取日志失败：" + (err && err.message ? err.message : err);
        metaEl.textContent = "读取失败";
      });
  }

  function copyLog() {
    var text = currentText || logEl.textContent || "";
    if (!text) {
      toast("没有可复制的内容");
      return;
    }
    function ok() { toast("已复制到剪贴板"); }
    function fail() {
      try {
        var range = document.createRange();
        range.selectNodeContents(logEl);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);
        toast("请按 Ctrl+C 复制");
      } catch (err) {
        toast("复制失败");
      }
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(ok).catch(fail);
    } else {
      fail();
    }
  }

  function exportLog() {
    var text = currentText || logEl.textContent || "";
    if (!text) {
      toast("没有可导出的内容");
      return;
    }
    var stamp = new Date();
    var name = "yprvc-" + source + "-"
      + stamp.getFullYear()
      + String(stamp.getMonth() + 1).padStart(2, "0")
      + String(stamp.getDate()).padStart(2, "0")
      + "-"
      + String(stamp.getHours()).padStart(2, "0")
      + String(stamp.getMinutes()).padStart(2, "0")
      + String(stamp.getSeconds()).padStart(2, "0")
      + ".txt";
    var blob = new Blob([text], { type: "text/plain;charset=utf-8" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url;
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
    toast("已导出 " + name);
  }

  function schedule() {
    if (timer) clearInterval(timer);
    timer = null;
    if (autoEl.checked) timer = setInterval(function () { loadLog(false); }, 3000);
  }

  document.getElementById("tabs").addEventListener("click", function (event) {
    var btn = event.target.closest("button[data-source]");
    if (!btn) return;
    setSource(btn.getAttribute("data-source"));
  });
  document.getElementById("btn-refresh").addEventListener("click", function () { loadLog(true); });
  document.getElementById("btn-copy").addEventListener("click", copyLog);
  document.getElementById("btn-export").addEventListener("click", exportLog);
  limitEl.addEventListener("change", function () { loadLog(true); });
  autoEl.addEventListener("change", schedule);
  logEl.addEventListener("scroll", function () {
    stickBottom = logEl.scrollHeight - logEl.scrollTop - logEl.clientHeight < 80;
  });

  loadLog(true);
  schedule();
})();
</script>
</body>
</html>
"""


UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #fff; }
  body { box-sizing: border-box; padding: 10px 12px 12px; font: 13px/1.45 "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; color: #0f172a; overflow: auto; }
  .hint { min-height: 18px; margin-bottom: 8px; color: #64748b; }
  .hint.warn { color: #b45309; font-weight: 650; }
  .hint.ok { color: #047857; font-weight: 650; }
  .drop { border: 1px dashed #cbd5e1; border-radius: 8px; padding: 8px; }
  .drop.active { border-color: #2563eb; background: #eff6ff; }
  .toolbar { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
  button.pick, button.cancel-all {
    border: 1px solid #1e293b; background: #1e293b; color: #f8fafc; border-radius: 8px;
    font: inherit; font-weight: 600; padding: 7px 12px; cursor: pointer;
  }
  button.cancel-all { border-color: #b91c1c; background: #fff; color: #b91c1c; display: none; }
  button.pick:disabled, button.cancel-all:disabled { opacity: .45; cursor: not-allowed; }
  button.file-cancel {
    border: 1px solid #cbd5e1; background: #fff; color: #b91c1c; border-radius: 6px;
    font: inherit; font-size: 12px; padding: 2px 8px; cursor: pointer; flex: 0 0 auto;
  }
  button.file-cancel:disabled { opacity: .4; cursor: not-allowed; color: #94a3b8; }
  .overall { margin-top: 8px; color: #334155; font-variant-numeric: tabular-nums; }
  .overall .track { height: 8px; margin-top: 6px; background: #e2e8f0; border-radius: 99px; overflow: hidden; }
  .overall .fill { height: 100%; width: 0; background: #2563eb; border-radius: 99px; transition: width .15s linear; }
  .list-title { margin: 10px 0 6px; font-weight: 650; color: #0f172a; }
  #file-list { display: flex; flex-direction: column; gap: 8px; max-height: 280px; overflow: auto; padding-right: 2px; }
  .file-row { border: 1px solid #e2e8f0; border-radius: 8px; padding: 8px 10px; background: #f8fafc; }
  .file-row.active { border-color: #93c5fd; background: #eff6ff; }
  .file-row .top { display: flex; justify-content: space-between; gap: 8px; margin-bottom: 6px; align-items: center; }
  .file-row .name { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; min-width: 0; }
  .file-row .right { display: flex; gap: 8px; align-items: center; flex: 0 0 auto; }
  .file-row .status { color: #64748b; font-size: 12px; }
  .file-row.done .status { color: #047857; }
  .file-row.fail .status, .file-row.cancelled .status { color: #b45309; }
  .file-row .track { height: 8px; background: #e2e8f0; border-radius: 99px; overflow: hidden; }
  .file-row .fill { height: 100%; width: 0; background: #2563eb; border-radius: 99px; transition: width .15s linear; }
  .file-row .meta { margin-top: 4px; color: #64748b; font-size: 12px; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
  <div id="hint" class="hint warn">请输入模型名称</div>
  <div id="drop" class="drop">
    <div class="toolbar">
      <button id="pick" class="pick" type="button" disabled>选择音频，可同时选多个（并行上传）</button>
      <button id="cancel-all" class="cancel-all" type="button">全部取消</button>
    </div>
    <input id="files" type="file" multiple hidden accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.opus,.aiff,.aif,audio/*">
  </div>
  <div class="list-title">各文件上传进度（并行）</div>
  <div id="file-list"></div>
  <div class="overall">
    <div id="stats">等待上传</div>
    <div class="track" aria-hidden="true"><div id="fill" class="fill"></div></div>
  </div>
<script>
(function () {
  var hint = document.getElementById("hint");
  var pick = document.getElementById("pick");
  var cancelAllBtn = document.getElementById("cancel-all");
  var files = document.getElementById("files");
  var drop = document.getElementById("drop");
  var fill = document.getElementById("fill");
  var stats = document.getElementById("stats");
  var fileList = document.getElementById("file-list");
  var uploading = false;
  var jobs = [];
  var startedAt = 0;
  var totalBytes = 0;
  var modelName = "";
  var CHUNK_SIZE = 4 * 1024 * 1024;
  var PARALLEL = 2;
  var MAX_RETRY = 4;
  var unloadMsg = "音频正在上传，刷新或关闭页面会中断上传。确定要离开吗？";

  function clientLog(level, message, detail) {
    try {
      var body = new FormData();
      body.append("level", level || "info");
      body.append("message", message || "");
      body.append("detail", JSON.stringify(detail || {}));
      var xhr = new XMLHttpRequest();
      xhr.open("POST", "dataset-upload-log");
      xhr.send(body);
    } catch (err) {}
  }

  function formatSize(bytes) {
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
    if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
    return Math.round(bytes) + " B";
  }

  function formatRate(bytesPerSec) {
    if (!isFinite(bytesPerSec) || bytesPerSec < 0) bytesPerSec = 0;
    if (bytesPerSec >= 1048576) return (bytesPerSec / 1048576).toFixed(1) + " MB/s";
    if (bytesPerSec >= 1024) return (bytesPerSec / 1024).toFixed(1) + " KB/s";
    return Math.round(bytesPerSec) + " B/s";
  }

  function onBeforeUnload(event) {
    if (!uploading) return;
    event.preventDefault();
    event.returnValue = unloadMsg;
    return unloadMsg;
  }

  function setUnloadGuard(on) {
    uploading = !!on;
    cancelAllBtn.style.display = on ? "inline-block" : "none";
    try {
      parent.window.__yprvcUploading = !!on;
      if (!parent.window.__yprvcUnloadBound) {
        parent.window.__yprvcUnloadBound = function (event) {
          if (!parent.window.__yprvcUploading) return;
          event.preventDefault();
          event.returnValue = unloadMsg;
          return unloadMsg;
        };
        parent.window.addEventListener("beforeunload", parent.window.__yprvcUnloadBound);
      }
    } catch (err) {}
  }

  window.addEventListener("beforeunload", onBeforeUnload);

  function readName() {
    try {
      var root = parent.document.querySelector("#model-name");
      if (!root) return "";
      var field = root.querySelector("textarea, input");
      return field && field.value ? field.value.trim() : "";
    } catch (err) {
      return "";
    }
  }

  function syncLock() {
    if (uploading) return;
    var name = readName();
    pick.disabled = !name;
    if (!name) {
      hint.className = "hint warn";
      hint.textContent = "请输入模型名称";
    } else if (!hint.classList.contains("ok")) {
      hint.className = "hint";
      hint.textContent = "模型「" + name + "」。可一次选多个，最多同时并行上传 " + PARALLEL + " 个文件。";
    }
  }

  function refreshParent() {
    try {
      var root = parent.document.querySelector("#refresh-audio-dir");
      if (!root) return;
      var button = root.matches("button") ? root : root.querySelector("button");
      if (button) button.click();
    } catch (err) {}
  }

  function randomId() {
    var alphabet = "0123456789abcdef";
    var chars = [];
    for (var i = 0; i < 32; i++) chars.push(alphabet.charAt(Math.floor(Math.random() * alphabet.length)));
    return chars.join("");
  }

  function sumLoaded() {
    var n = 0;
    for (var i = 0; i < jobs.length; i++) n += jobs[i].loaded || 0;
    return n;
  }

  function countBy(state) {
    var n = 0;
    for (var i = 0; i < jobs.length; i++) if (jobs[i].state === state) n++;
    return n;
  }

  function refreshOverall() {
    var loaded = sumLoaded();
    var elapsed = Math.max((Date.now() - startedAt) / 1000, 0.001);
    var percent = totalBytes > 0 ? Math.min(100, Math.round(loaded / totalBytes * 100)) : 0;
    fill.style.width = percent + "%";
    stats.textContent = "总进度 " + percent + "% · 并行 " + countBy("uploading") + "/" + PARALLEL
      + " · 完成 " + countBy("done") + " · 取消 " + countBy("cancelled") + " · 失败 " + countBy("failed")
      + " · " + formatRate(loaded / elapsed) + " · " + formatSize(loaded) + " / " + formatSize(totalBytes);
  }

  function setJobUi(job) {
    var row = job.row;
    if (!row) return;
    var percent = job.file.size > 0 ? Math.min(100, Math.round((job.loaded || 0) / job.file.size * 100)) : 0;
    row.fill.style.width = percent + "%";
    if (job.state === "waiting") {
      row.el.className = "file-row";
      row.status.textContent = "排队中";
      row.cancel.disabled = false;
    } else if (job.state === "uploading") {
      row.el.className = "file-row active";
      row.status.textContent = "上传中 " + percent + "%";
      row.cancel.disabled = false;
    } else if (job.state === "done") {
      row.el.className = "file-row done";
      row.status.textContent = "已完成";
      row.cancel.disabled = true;
      row.fill.style.width = "100%";
    } else if (job.state === "cancelled") {
      row.el.className = "file-row cancelled";
      row.status.textContent = "已取消";
      row.cancel.disabled = true;
    } else {
      row.el.className = "file-row fail";
      row.status.textContent = "失败";
      row.cancel.disabled = true;
    }
    var elapsed = Math.max((Date.now() - (job.started || startedAt)) / 1000, 0.001);
    row.meta.textContent = formatSize(job.loaded || 0) + " / " + formatSize(job.file.size || 0)
      + (job.state === "uploading" ? (" · " + formatRate((job.loaded || 0) / elapsed)) : "")
      + (job.message ? (" · " + job.message) : "");
  }

  function cleanupPart(job) {
    if (!job || !job.uploadId || !modelName) return;
    var body = new FormData();
    body.append("name", modelName);
    body.append("filename", job.file.name || "");
    body.append("upload_id", job.uploadId);
    body.append("chunk_size", String(CHUNK_SIZE));
    try {
      var xhr = new XMLHttpRequest();
      xhr.open("POST", "dataset-upload-cancel");
      xhr.send(body);
    } catch (err) {}
  }

  function cancelJob(index) {
    var job = jobs[index];
    if (!job || job.state === "done" || job.state === "cancelled" || job.state === "failed") return;
    job.cancelled = true;
    if (job.xhr) {
      try { job.xhr.abort(); } catch (err) {}
      job.xhr = null;
    }
    if (job.state === "uploading" || job.state === "waiting") {
      job.state = "cancelled";
      job.message = "用户取消";
      clientLog("warning", "用户取消上传", {
        model: modelName,
        file: job.file.name,
        loaded: job.loaded,
        size: job.file.size,
        upload_id: job.uploadId
      });
      cleanupPart(job);
      setJobUi(job);
      refreshOverall();
      pump();
      maybeFinish();
    }
  }

  function cancelAll() {
    clientLog("warning", "用户点击全部取消", { model: modelName, jobs: jobs.length });
    for (var i = 0; i < jobs.length; i++) cancelJob(i);
  }

  function renderRows(batch) {
    fileList.innerHTML = "";
    jobs = [];
    for (var i = 0; i < batch.length; i++) {
      var row = document.createElement("div");
      row.className = "file-row";
      row.innerHTML = '<div class="top"><div class="name"></div><div class="right"><div class="status">排队中</div><button class="file-cancel" type="button">取消</button></div></div><div class="track"><div class="fill"></div></div><div class="meta">0 B / ' + formatSize(batch[i].size || 0) + '</div>';
      row.querySelector(".name").textContent = (i + 1) + ". " + (batch[i].name || ("文件 " + (i + 1)));
      var cancelBtn = row.querySelector(".file-cancel");
      (function (idx) {
        cancelBtn.addEventListener("click", function () { cancelJob(idx); });
      })(i);
      fileList.appendChild(row);
      jobs.push({
        file: batch[i],
        state: "waiting",
        loaded: 0,
        uploadId: randomId(),
        xhr: null,
        cancelled: false,
        message: "",
        started: 0,
        row: {
          el: row,
          fill: row.querySelector(".fill"),
          status: row.querySelector(".status"),
          meta: row.querySelector(".meta"),
          cancel: cancelBtn
        }
      });
    }
  }

  function postChunk(job, index, total, onProgress, onDone, onFail, retryLeft) {
    if (typeof retryLeft !== "number") retryLeft = MAX_RETRY;
    var file = job.file;
    var start = index * CHUNK_SIZE;
    var pieceSize = Math.min(CHUNK_SIZE, Math.max(0, file.size - start));
    var blob = file.slice(start, start + pieceSize);
    var body = new FormData();
    body.append("name", modelName);
    body.append("filename", file.name);
    body.append("upload_id", job.uploadId);
    body.append("chunk_index", String(index));
    body.append("chunk_total", String(total));
    body.append("chunk_size", String(CHUNK_SIZE));
    body.append("files", blob, file.name);
    var xhr = new XMLHttpRequest();
    job.xhr = xhr;
    xhr.open("POST", "dataset-upload");
    xhr.upload.onprogress = function (event) {
      if (job.cancelled) return;
      onProgress(start + Math.min(event.loaded || 0, pieceSize));
    };
    function retryOrFail(reason, status) {
      job.xhr = null;
      if (job.cancelled) {
        onFail("cancelled");
        return;
      }
      var retriable = status === 502 || status === 503 || status === 504 || status === 0 || status === 408;
      if (retriable && retryLeft > 0) {
        clientLog("warning", "分片上传失败，准备重试", {
          model: modelName,
          file: file.name,
          chunk: index + 1,
          total: total,
          status: status,
          reason: reason,
          retry_left: retryLeft - 1
        });
        setTimeout(function () {
          if (job.cancelled) {
            onFail("cancelled");
            return;
          }
          postChunk(job, index, total, onProgress, onDone, onFail, retryLeft - 1);
        }, 800 + (MAX_RETRY - retryLeft) * 700);
        return;
      }
      clientLog("error", "分片上传最终失败", {
        model: modelName,
        file: file.name,
        chunk: index + 1,
        total: total,
        status: status,
        reason: reason
      });
      onFail(reason);
    }
    xhr.onerror = function () {
      retryOrFail("上传被中断，请重试", 0);
    };
    xhr.onabort = function () {
      job.xhr = null;
      onFail("cancelled");
    };
    xhr.onload = function () {
      job.xhr = null;
      if (job.cancelled) {
        onFail("cancelled");
        return;
      }
      var payload = {};
      try { payload = JSON.parse(xhr.responseText || "{}"); } catch (err) {}
      if (xhr.status >= 200 && xhr.status < 300 && payload.ok) onDone(payload);
      else retryOrFail(payload.message || ("上传失败（" + (xhr.status || "网络错误") + "）"), xhr.status || 0);
    };
    xhr.send(body);
  }

  function uploadOne(job) {
    if (job.cancelled) {
      job.state = "cancelled";
      setJobUi(job);
      pump();
      maybeFinish();
      return;
    }
    job.state = "uploading";
    job.started = Date.now();
    setJobUi(job);
    var total = Math.max(1, Math.ceil((job.file.size || 0) / CHUNK_SIZE));
    function sendAt(index) {
      if (job.cancelled) {
        job.state = "cancelled";
        job.message = "用户取消";
        cleanupPart(job);
        setJobUi(job);
        pump();
        maybeFinish();
        return;
      }
      hint.textContent = "并行上传中：进行中 " + countBy("uploading") + "，排队 " + countBy("waiting") + "，完成 " + countBy("done") + "。可取消单个文件。";
      postChunk(job, index, total, function (loadedInFile) {
        job.loaded = loadedInFile;
        setJobUi(job);
        refreshOverall();
      }, function (payload) {
        if (job.cancelled) {
          job.state = "cancelled";
          job.message = "用户取消";
          cleanupPart(job);
          setJobUi(job);
          pump();
          maybeFinish();
          return;
        }
        if (payload.partial && index + 1 < total) {
          sendAt(index + 1);
          return;
        }
        job.loaded = job.file.size || job.loaded;
        job.state = "done";
        job.message = "";
        setJobUi(job);
        refreshOverall();
        pump();
        maybeFinish();
      }, function (message) {
        if (message === "cancelled" || job.cancelled) {
          job.state = "cancelled";
          job.message = "用户取消";
          cleanupPart(job);
        } else {
          job.state = "failed";
          job.message = message || "上传失败";
          cleanupPart(job);
        }
        setJobUi(job);
        refreshOverall();
        pump();
        maybeFinish();
      });
    }
    sendAt(0);
  }

  function pump() {
    var active = countBy("uploading");
    for (var i = 0; i < jobs.length && active < PARALLEL; i++) {
      if (jobs[i].state === "waiting" && !jobs[i].cancelled) {
        uploadOne(jobs[i]);
        active++;
      }
    }
  }

  function maybeFinish() {
    for (var i = 0; i < jobs.length; i++) {
      if (jobs[i].state === "waiting" || jobs[i].state === "uploading") return;
    }
    setUnloadGuard(false);
    pick.disabled = false;
    files.value = "";
    var done = countBy("done");
    var cancelled = countBy("cancelled");
    var failed = countBy("failed");
    refreshOverall();
    if (done > 0) refreshParent();
    if (done === jobs.length) {
      hint.className = "hint ok";
      hint.textContent = "已全部上传完成：" + done + " 个音频";
    } else if (done > 0) {
      hint.className = "hint warn";
      hint.textContent = "上传结束：成功 " + done + "，取消 " + cancelled + "，失败 " + failed;
    } else {
      hint.className = "hint warn";
      hint.textContent = "没有文件上传成功（取消 " + cancelled + "，失败 " + failed + "）";
    }
    clientLog("info", "上传批次结束", {
      model: modelName,
      done: done,
      cancelled: cancelled,
      failed: failed,
      files: jobs.map(function (j) {
        return { name: j.file.name, state: j.state, message: j.message || "", loaded: j.loaded, size: j.file.size };
      })
    });
    syncLock();
  }

  function startUpload(list) {
    modelName = readName();
    if (!modelName) {
      hint.className = "hint warn";
      hint.textContent = "请输入模型名称";
      return;
    }
    if (uploading) {
      hint.className = "hint warn";
      hint.textContent = "当前仍有上传任务，请先取消或等待完成";
      return;
    }
    var batch = [];
    for (var i = 0; i < list.length; i++) batch.push(list[i]);
    if (!batch.length) return;
    setUnloadGuard(true);
    pick.disabled = true;
    hint.className = "hint warn";
    hint.textContent = "正在并行上传 " + batch.length + " 个文件（同时最多 " + PARALLEL + " 个）。可点「取消」中止单个文件。";
    fill.style.width = "0%";
    startedAt = Date.now();
    totalBytes = 0;
    for (var j = 0; j < batch.length; j++) totalBytes += batch[j].size || 0;
    renderRows(batch);
    clientLog("info", "开始并行上传", {
      model: modelName,
      count: batch.length,
      total_bytes: totalBytes,
      parallel: PARALLEL,
      chunk_size: CHUNK_SIZE,
      files: batch.map(function (f) { return { name: f.name, size: f.size }; })
    });
    refreshOverall();
    pump();
  }

  pick.addEventListener("click", function () {
    if (!readName()) {
      hint.className = "hint warn";
      hint.textContent = "请输入模型名称";
      return;
    }
    files.click();
  });
  cancelAllBtn.addEventListener("click", cancelAll);
  files.addEventListener("change", function () {
    if (files.files && files.files.length) startUpload(files.files);
  });
  drop.addEventListener("dragover", function (event) {
    event.preventDefault();
    if (readName() && !uploading) drop.className = "drop active";
  });
  drop.addEventListener("dragleave", function () { drop.className = "drop"; });
  drop.addEventListener("drop", function (event) {
    event.preventDefault();
    drop.className = "drop";
    if (!readName()) {
      hint.className = "hint warn";
      hint.textContent = "请输入模型名称";
      return;
    }
    if (event.dataTransfer && event.dataTransfer.files) startUpload(event.dataTransfer.files);
  });
  setInterval(syncLock, 300);
  syncLock();
})();
</script>
</body>
</html>
"""


# 浏览器访问仙宫云时走 HTTP/2。单次请求体超过大约 16MB，前面的 nginx 会在一分钟后直接拆连接，
# 页面上就是没有正文的 502。下面按 8MB 收分片，整文件仍落在模型目录里。
UPLOAD_CHUNK_LIMIT = 8 * 1024 * 1024
_upload_locks = {}
_upload_locks_guard = threading.Lock()


def valid_upload_id(value):
    text = str(value or "")
    if len(text) < 8 or len(text) > 64:
        return False
    return all(char in "0123456789abcdef" for char in text)


def audio_part_path(folder, dest_name, upload_id, chunk_size):
    partial = folder / ("%s.%s.%s.part" % (dest_name, upload_id, int(chunk_size)))
    if partial.parent.resolve() != folder.resolve():
        raise ValueError("文件名无效")
    return partial


def upload_part_lock(key):
    with _upload_locks_guard:
        lock = _upload_locks.get(key)
        if lock is None:
            lock = threading.Lock()
            _upload_locks[key] = lock
        return lock


def release_upload_part_lock(key, lock):
    with _upload_locks_guard:
        if _upload_locks.get(key) is lock:
            _upload_locks.pop(key, None)


def store_audio_chunk(partial, index, chunk_size, data):
    offset = index * chunk_size
    if index == 0:
        if partial.exists():
            partial.unlink()
    elif not partial.exists() or partial.stat().st_size < offset:
        raise ValueError("分片顺序错误，请重新上传这个文件")
    with partial.open("r+b" if partial.exists() else "wb+") as handle:
        handle.seek(offset)
        handle.truncate()
        written = handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    if written != len(data):
        raise OSError("写入不完整")
    return offset + len(data)


def discard_audio_parts(folder, dest_name):
    prefix = dest_name + "."
    for old in folder.iterdir():
        if old.is_file() and old.name.startswith(prefix) and old.name.endswith(".part"):
            old.unlink()


def register_upload_routes(app):
    if getattr(app, "_yprvc_upload_routes", False):
        return
    app._yprvc_upload_routes = True
    from fastapi import File, Form, UploadFile
    from fastapi.responses import HTMLResponse, JSONResponse

    @app.get("/dataset-upload-ui", response_class=HTMLResponse)
    def dataset_upload_ui():
        return HTMLResponse(UPLOAD_PAGE, headers={"Cache-Control": "no-store"})

    @app.post("/dataset-upload-log")
    async def dataset_upload_log(
        level: str = Form("info"),
        message: str = Form(""),
        detail: str = Form(""),
    ):
        lvl = (level or "info").strip().lower()
        if lvl not in {"debug", "info", "warning", "error", "exception"}:
            lvl = "info"
        text = (message or "").strip() or "(empty)"
        extra = (detail or "").strip()
        if extra and len(extra) > 4000:
            extra = extra[:4000] + "..."
        if extra:
            log_upload("%s | %s", text, extra, level=lvl)
        else:
            log_upload("%s", text, level=lvl)
        return {"ok": True}

    @app.post("/dataset-upload-cancel")
    async def dataset_upload_cancel(
        name: str = Form(""),
        filename: str = Form(""),
        upload_id: str = Form(""),
        chunk_size: int = Form(0),
    ):
        log_upload(
            "取消上传请求 name=%s filename=%s upload_id=%s chunk_size=%s",
            name,
            filename,
            upload_id,
            chunk_size,
        )
        try:
            folder = dataset_folder(name, create=False)
        except ValueError as exc:
            log_upload("取消上传失败: %s", exc, level="warning")
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        if not folder.is_dir():
            return JSONResponse({"ok": True, "message": "目录不存在，无需清理"})
        if not valid_upload_id(upload_id):
            log_upload("取消上传失败: 上传编号无效 %s", upload_id, level="warning")
            return JSONResponse({"ok": False, "message": "上传编号无效"}, status_code=400)
        try:
            dest_name = safe_audio_filename((filename or "").strip() or "audio.bin")
        except ValueError:
            log_upload("取消上传失败: 文件名无效 %s", filename, level="warning")
            return JSONResponse({"ok": False, "message": "文件名无效"}, status_code=400)
        removed = 0
        try:
            if chunk_size > 0:
                partial = audio_part_path(folder, dest_name, upload_id, chunk_size)
                if partial.is_file():
                    partial.unlink()
                    removed += 1
            discard_audio_parts(folder, dest_name)
        except OSError:
            log_upload("取消上传时清理分片失败: %s", dest_name, level="error")
            logger.exception("取消上传时清理分片失败: %s", dest_name)
            return JSONResponse({"ok": False, "message": "清理分片失败"}, status_code=500)
        log_upload("已取消上传 filename=%s removed=%s", dest_name, removed)
        return {"ok": True, "removed": removed, "filename": dest_name}

    @app.get("/train-status-ui", response_class=HTMLResponse)
    def train_status_ui():
        return HTMLResponse(TRAIN_STATUS_UI_PAGE, headers={"Cache-Control": "no-store"})

    @app.get("/ui-session")
    def ui_session_get():
        return JSONResponse(build_ui_session_payload())

    @app.post("/ui-session")
    async def ui_session_post(name: str = Form("")):
        payload = write_ui_session(name, note="client")
        if not payload:
            return JSONResponse(
                {"ok": False, "message": "模型名称无效或为空"},
                status_code=400,
            )
        return JSONResponse({"ok": True, **payload})

    @app.get("/logs-ui", response_class=HTMLResponse)
    def logs_ui():
        return HTMLResponse(LOG_VIEWER_PAGE, headers={"Cache-Control": "no-store"})

    @app.get("/logs-api")
    def logs_api(source: str = "webui", limit: int = 500):
        return JSONResponse(read_log_payload(source, limit))

    @app.get("/webui-log")
    def webui_log(limit: int = 200, source: str = "webui"):
        return JSONResponse(read_log_payload(source or "webui", limit))

    @app.get("/logs-download")
    def logs_download(source: str = "webui", limit: int = 5000):
        from fastapi.responses import PlainTextResponse

        payload = read_log_payload(source, limit)
        key = payload.get("source") or "webui"
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = "yprvc-%s-%s.txt" % (key, stamp)
        return PlainTextResponse(
            payload.get("text") or "",
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="%s"' % filename,
                "Cache-Control": "no-store",
            },
        )

    @app.get("/train-status")
    def train_status():
        return JSONResponse(get_train_status_payload())

    @app.get("/job-status")
    def job_status():
        return JSONResponse(get_job_status_payload())

    @app.post("/dataset-upload")
    async def dataset_upload(
        name: str = Form(""),
        files: list[UploadFile] = File(...),
        filename: str = Form(""),
        upload_id: str = Form(""),
        chunk_index: int = Form(-1),
        chunk_total: int = Form(0),
        chunk_size: int = Form(0),
    ):
        log_upload(
            "收到上传 name=%s filename=%s upload_id=%s chunk=%s/%s chunk_size=%s files=%s",
            name,
            filename,
            upload_id,
            chunk_index,
            chunk_total,
            chunk_size,
            len(files or []),
        )
        if not (name or "").strip():
            log_upload("上传拒绝: 未填模型名称", level="warning")
            return JSONResponse({"ok": False, "message": "请输入模型名称"}, status_code=400)
        if not files:
            log_upload("上传拒绝: 无文件", level="warning")
            return JSONResponse({"ok": False, "message": "请选择音频文件"}, status_code=400)
        try:
            folder = dataset_folder(name, create=True)
        except ValueError as exc:
            log_upload("上传拒绝: %s", exc, level="warning")
            return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
        if chunk_total > 0:
            return await accept_audio_chunk(
                folder,
                files,
                filename,
                upload_id,
                chunk_index,
                chunk_total,
                chunk_size,
            )
        saved = []
        skipped = []
        seen = set()
        total_bytes = 0
        for upload in files:
            original = getattr(upload, "filename", "") or ""
            try:
                dest_name = safe_audio_filename(original)
            except ValueError:
                skipped.append(original or "未命名文件")
                await upload.close()
                continue
            dest = folder / dest_name
            partial = folder / (dest_name + ".uploading")
            try:
                with partial.open("wb") as handle:
                    while True:
                        chunk = await upload.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        total_bytes += len(chunk)
                partial.replace(dest)
            except Exception:
                logger.exception("保存音频失败: %s", original or dest_name)
                if partial.exists():
                    partial.unlink()
                return JSONResponse({"ok": False, "message": "保存 %s 失败" % (original or dest_name)}, status_code=500)
            finally:
                await upload.close()
            if dest_name not in seen:
                saved.append(dest_name)
                seen.add(dest_name)
        if saved:
            try:
                write_ui_session(folder.name, note="upload_done")
            except Exception:
                pass
        return {
            "ok": True,
            "saved": saved,
            "skipped": skipped,
            "bytes": total_bytes,
            "folder": str(folder),
            "message": summarize_saved_audio(folder, saved, skipped),
        }


async def accept_audio_chunk(folder, files, filename, upload_id, chunk_index, chunk_total, chunk_size):
    from fastapi.responses import JSONResponse

    if len(files) != 1:
        log_upload("分片拒绝: 一次只能上传一片 files=%s", len(files), level="warning")
        return JSONResponse({"ok": False, "message": "一次只能上传一片"}, status_code=400)
    if chunk_total > 8192 or chunk_index < 0 or chunk_index >= chunk_total:
        log_upload(
            "分片拒绝: 序号无效 chunk=%s/%s", chunk_index, chunk_total, level="warning"
        )
        return JSONResponse({"ok": False, "message": "分片序号无效"}, status_code=400)
    if chunk_size <= 0 or chunk_size > UPLOAD_CHUNK_LIMIT:
        log_upload(
            "分片拒绝: 大小无效 chunk_size=%s limit=%s",
            chunk_size,
            UPLOAD_CHUNK_LIMIT,
            level="warning",
        )
        return JSONResponse({"ok": False, "message": "分片大小无效"}, status_code=400)
    if not valid_upload_id(upload_id):
        log_upload("分片拒绝: 上传编号无效 %s", upload_id, level="warning")
        return JSONResponse({"ok": False, "message": "上传编号无效"}, status_code=400)
    upload = files[0]
    original = (filename or "").strip() or (getattr(upload, "filename", "") or "")
    try:
        dest_name = safe_audio_filename(original)
        partial = audio_part_path(folder, dest_name, upload_id, chunk_size)
    except ValueError:
        await upload.close()
        return JSONResponse({"ok": False, "message": "只能上传 wav、mp3、flac 等音频"}, status_code=400)
    data = bytearray()
    too_big = False
    try:
        while True:
            piece = await upload.read(1024 * 1024)
            if not piece:
                break
            if len(data) + len(piece) > UPLOAD_CHUNK_LIMIT:
                too_big = True
                break
            data.extend(piece)
    finally:
        await upload.close()
    if too_big or len(data) > chunk_size:
        return JSONResponse({"ok": False, "message": "分片过大"}, status_code=400)
    if chunk_index < chunk_total - 1 and len(data) != chunk_size:
        return JSONResponse({"ok": False, "message": "分片不完整"}, status_code=400)
    lock_key = str(partial)
    lock = upload_part_lock(lock_key)
    try:
        with lock:
            final_size = store_audio_chunk(partial, chunk_index, chunk_size, data)
            if chunk_index != chunk_total - 1:
                log_upload(
                    "分片已写入 %s chunk=%s/%s size=%s",
                    dest_name,
                    chunk_index + 1,
                    chunk_total,
                    len(data),
                )
                return {"ok": True, "partial": True, "chunk_index": chunk_index, "chunk_total": chunk_total}
            dest = folder / dest_name
            partial.replace(dest)
            try:
                discard_audio_parts(folder, dest_name)
            except OSError:
                logger.exception("清理残留分片失败: %s", dest_name)
        log_upload("音频已保存 %s (%s bytes)", dest, final_size)
        logger.info("音频已保存 %s (%s bytes)", dest, final_size)
        try:
            write_ui_session(folder.name, note="upload_done")
        except Exception:
            pass
        return {
            "ok": True,
            "saved": [dest_name],
            "skipped": [],
            "bytes": final_size,
            "folder": str(folder),
            "message": summarize_saved_audio(folder, [dest_name], []),
        }
    except ValueError as exc:
        log_upload("分片失败 ValueError %s: %s", original or dest_name, exc, level="warning")
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    except Exception:
        log_upload("分片失败 Exception %s", original or dest_name, level="error")
        logger.exception("保存音频分片失败: %s", original or dest_name)
        if partial.exists():
            partial.unlink()
        return JSONResponse({"ok": False, "message": "保存 %s 失败" % (original or dest_name)}, status_code=500)
    finally:
        release_upload_part_lock(lock_key, lock)


def main():
    parser = argparse.ArgumentParser(description="RVC training WebUI")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "7860")))
    args = parser.parse_args()
    port = pick_port(args.port)
    if port != args.port:
        logger.warning("端口 %s 已被占用，改用 %s", args.port, port)
    logger.info("训练页 http://127.0.0.1:%s", port)
    logger.info(asset_status())
    from gradio.routes import App

    original_create_app = App.create_app

    def create_app(blocks, app_kwargs=None):
        app = original_create_app(blocks, app_kwargs)
        register_upload_routes(app)
        return app

    App.create_app = create_app
    build_app().queue(concurrency_count=8, max_size=16).launch(
        server_name="0.0.0.0",
        server_port=port,
        inbrowser=True,
        show_error=True,
    )


if __name__ == "__main__":
    main()
