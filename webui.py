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
from time import sleep

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
        return str(folder), describe_audio_dir(folder)
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
    for port in range(start, min(start + 100, 65536)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise RuntimeError("7860 附近没有可用端口")


def spawn(args):
    logger.info("Execute: %s", " ".join(args))
    return Popen(args, cwd=str(ROOT), env=child_env())


PAUSE_FILE = "train.pause"
STOP_FILE = "train.stop"
TRAIN_LOCK = threading.Lock()
TRAIN_STATE = {
    "process": None,
    "folder": None,
    "stop_requested": False,
}


def train_process():
    process = TRAIN_STATE["process"]
    if process is None or process.poll() is not None:
        return None
    return process


def clear_train_flags(folder):
    if folder is None:
        return
    for name in (PAUSE_FILE, STOP_FILE):
        path = folder / name
        if path.is_file():
            path.unlink()


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


def release_train_process(process, folder):
    with TRAIN_LOCK:
        if TRAIN_STATE["process"] is process:
            TRAIN_STATE["process"] = None
            TRAIN_STATE["folder"] = None
    clear_train_flags(folder)


def wait_process(process):
    done = [False]

    def watch():
        process.wait()
        done[0] = True

    threading.Thread(target=watch, daemon=True).start()
    return done


def wait_processes(processes):
    done = [False]

    def watch():
        for process in processes:
            process.wait()
        done[0] = True

    threading.Thread(target=watch, daemon=True).start()
    return done


def read_log(path):
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


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
    yield from follow_log(log_path, wait_process(process))


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
            done = wait_processes(processes)
        yield from follow_log(log_path, done)
        log_path.write_text(read_log(log_path), encoding="utf-8")

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
    yield from follow_log(log_path, wait_processes(processes))


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
        TRAIN_STATE["stop_requested"] = False
    yield "训练进行中\n正在加载模型", "running"
    last_text = ""
    while process.poll() is None:
        sleep(1)
        text = ""
        if progress_path.is_file():
            text = "\n".join(
                progress_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
            ).strip()
        shown = text or "正在加载模型"
        if shown != last_text:
            last_text = shown
            yield "训练进行中\n" + shown, "running"
    code = process.wait()
    text = ""
    if progress_path.is_file():
        text = "\n".join(
            progress_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        ).strip()
    try:
        with TRAIN_LOCK:
            stopped = TRAIN_STATE["stop_requested"]
    finally:
        release_train_process(process, folder)
    if stopped:
        yield "训练已终止。再次训练会从最近一次保存的检查点继续\n" + text, "hidden"
        return
    if code not in (0, None):
        yield "训练异常退出，退出码 %s\n%s" % (code, text), "hidden"
        return
    yield "训练结束\n" + text, "hidden"


def control_updates(phase):
    return (
        {"__type__": "update", "visible": phase == "running"},
        {"__type__": "update", "visible": phase == "paused"},
        {"__type__": "update", "visible": phase in {"running", "paused"}},
    )


def pack_status(message, phase):
    return (message,) + control_updates(phase)


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
        return pack_status("训练已暂停。当前这一步跑完后会停住，点「继续训练」恢复", "paused")


def click_resume_train():
    with TRAIN_LOCK:
        process = train_process()
        folder = TRAIN_STATE["folder"]
        if process is None or folder is None:
            return pack_status("当前没有正在进行的训练", "hidden")
        pause_path = folder / PAUSE_FILE
        if pause_path.is_file():
            pause_path.unlink()
        return pack_status("训练已继续", "running")


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
    threading.Thread(
        target=force_stop_later, args=(process, pid, 60), daemon=True
    ).start()
    return pack_status(
        "已发送终止指令。当前这一步结束后会保存权重并退出；若一分钟内还没退出，会强制结束",
        "hidden",
    )


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
                        '<iframe class="yp-upload-frame" src="dataset-upload-ui?v=2" title="上传音频"></iframe>',
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
                    )
                    info1 = gr.Textbox(label=i18n("输出信息"), value="", lines=6, max_lines=12)
            exp_dir1.change(preview_dataset_dir, [exp_dir1], [trainset_dir4, upload_info])
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
                    info2 = gr.Textbox(label=i18n("输出信息"), value="", lines=7, max_lines=14)

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
                    value=i18n("否"),
                    info="在每次保存点将最终推理小模型同步保存到 weights",
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
                            elem_classes=["yp-ctrl-btn", "yp-btn-pause"],
                        )
                        but_resume = gr.Button(
                            "继续训练",
                            variant="secondary",
                            visible=False,
                            elem_classes=["yp-ctrl-btn", "yp-btn-resume"],
                        )
                        but_stop = gr.Button(
                            "终止训练",
                            variant="secondary",
                            visible=False,
                            elem_classes=["yp-ctrl-btn", "yp-btn-stop"],
                        )
                    with gr.Row():
                        but5 = gr.Button(i18n("一键训练"), variant="primary", elem_classes=["yp-primary-accent-btn", "yp-btn-train-all"])
                with gr.Column(scale=2):
                    info3 = gr.Textbox(label=i18n("输出信息"), value="", lines=7, max_lines=16)

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
    return app


UPLOAD_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  html, body { margin: 0; height: 100%; background: #fff; }
  body { box-sizing: border-box; padding: 10px 12px 12px; font: 13px/1.45 "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif; color: #0f172a; }
  .hint { min-height: 18px; margin-bottom: 8px; color: #64748b; }
  .hint.warn { color: #b45309; font-weight: 650; }
  .drop { border: 1px dashed #cbd5e1; border-radius: 8px; padding: 8px; }
  .drop.active { border-color: #2563eb; background: #eff6ff; }
  button.pick { border: 1px solid #1e293b; background: #1e293b; color: #f8fafc; border-radius: 8px; font: inherit; font-weight: 600; padding: 7px 12px; cursor: pointer; }
  button.pick:disabled { opacity: .45; cursor: not-allowed; }
  .track { height: 8px; margin-top: 10px; background: #e2e8f0; border-radius: 99px; overflow: hidden; }
  .fill { height: 100%; width: 0; background: #2563eb; border-radius: 99px; transition: width .15s linear; }
  .stats { margin-top: 6px; color: #334155; font-variant-numeric: tabular-nums; }
</style>
</head>
<body>
  <div id="hint" class="hint warn">请输入模型名称</div>
  <div id="drop" class="drop">
    <button id="pick" class="pick" type="button" disabled>选择音频，可同时选多个</button>
    <input id="files" type="file" multiple hidden accept=".wav,.mp3,.flac,.ogg,.m4a,.aac,.wma,.opus,.aiff,.aif,audio/*">
  </div>
  <div class="track" aria-hidden="true"><div id="fill" class="fill"></div></div>
  <div id="stats" class="stats">等待上传</div>
<script>
(function () {
  var hint = document.getElementById("hint");
  var pick = document.getElementById("pick");
  var files = document.getElementById("files");
  var drop = document.getElementById("drop");
  var fill = document.getElementById("fill");
  var stats = document.getElementById("stats");
  var uploading = false;

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
    } else {
      hint.className = "hint";
      hint.textContent = "模型「" + name + "」。可以一次选择多个音频。";
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

  function showProgress(loaded, total, started) {
    var elapsed = Math.max((Date.now() - started) / 1000, 0.001);
    var rate = loaded / elapsed;
    var percent = total > 0 ? Math.min(100, Math.round(loaded / total * 100)) : 0;
    fill.style.width = percent + "%";
    var line = "进度 " + percent + "% · 速率 " + formatRate(rate);
    if (total > 0) line += " · " + formatSize(loaded) + " / " + formatSize(total);
    else line += " · " + formatSize(loaded);
    stats.textContent = line;
  }

  var CHUNK_SIZE = 8 * 1024 * 1024;

  function randomId() {
    var alphabet = "0123456789abcdef";
    var chars = [];
    for (var i = 0; i < 32; i++) chars.push(alphabet.charAt(Math.floor(Math.random() * alphabet.length)));
    return chars.join("");
  }

  function failUpload(message) {
    uploading = false;
    hint.className = "hint warn";
    hint.textContent = message || "上传失败";
    fill.style.width = "0%";
    files.value = "";
    syncLock();
  }

  function finishUpload(savedCount, skipCount, started, sent) {
    uploading = false;
    fill.style.width = "100%";
    var elapsed = Math.max((Date.now() - started) / 1000, 0.001);
    stats.textContent = "进度 100% · 速率 " + formatRate(sent / elapsed);
    hint.className = "hint";
    hint.textContent = "已保存 " + savedCount + " 个文件" + (skipCount ? "，跳过 " + skipCount + " 个" : "");
    files.value = "";
    refreshParent();
    syncLock();
  }

  function postChunk(modelName, file, uploadId, index, total, onProgress, onDone, onFail) {
    var start = index * CHUNK_SIZE;
    var pieceSize = Math.min(CHUNK_SIZE, Math.max(0, file.size - start));
    var blob = file.slice(start, start + pieceSize);
    var body = new FormData();
    body.append("name", modelName);
    body.append("filename", file.name);
    body.append("upload_id", uploadId);
    body.append("chunk_index", String(index));
    body.append("chunk_total", String(total));
    body.append("chunk_size", String(CHUNK_SIZE));
    body.append("files", blob, file.name);
    var xhr = new XMLHttpRequest();
    xhr.open("POST", "dataset-upload");
    xhr.upload.onprogress = function (event) {
      onProgress(start + Math.min(event.loaded || 0, pieceSize));
    };
    xhr.onerror = function () {
      onFail("上传被中断，请重试");
    };
    xhr.onload = function () {
      var payload = {};
      try { payload = JSON.parse(xhr.responseText || "{}"); } catch (err) {}
      if (xhr.status >= 200 && xhr.status < 300 && payload.ok) onDone(payload);
      else onFail(payload.message || ("上传失败（" + (xhr.status || "网络错误") + "）"));
    };
    xhr.send(body);
  }

  function startUpload(list) {
    var name = readName();
    if (!name) {
      hint.className = "hint warn";
      hint.textContent = "请输入模型名称";
      return;
    }
    var batch = [];
    for (var i = 0; i < list.length; i++) batch.push(list[i]);
    if (!batch.length || uploading) return;
    uploading = true;
    pick.disabled = true;
    hint.className = "hint";
    hint.textContent = "正在上传 " + batch.length + " 个文件";
    fill.style.width = "0%";
    stats.textContent = "进度 0% · 速率 0 MB/s";
    var started = Date.now();
    var sent = 0;
    var savedCount = 0;
    var skipCount = 0;
    var previousBytes = 0;
    var totalBytes = 0;
    for (var j = 0; j < batch.length; j++) totalBytes += batch[j].size || 0;

    function uploadAt(fileIndex) {
      if (fileIndex >= batch.length) {
        finishUpload(savedCount, skipCount, started, sent);
        return;
      }
      var file = batch[fileIndex];
      var total = Math.max(1, Math.ceil((file.size || 0) / CHUNK_SIZE));
      var uploadId = randomId();
      function sendAt(index) {
        hint.textContent = "正在上传 " + (fileIndex + 1) + "/" + batch.length + "，第 " + (index + 1) + "/" + total + " 片";
        postChunk(name, file, uploadId, index, total, function (loadedInFile) {
          sent = previousBytes + loadedInFile;
          showProgress(sent, totalBytes, started);
        }, function (payload) {
          if (payload.partial && index + 1 < total) {
            sendAt(index + 1);
            return;
          }
          savedCount += (payload.saved || []).length;
          skipCount += (payload.skipped || []).length;
          previousBytes += file.size || 0;
          sent = previousBytes;
          showProgress(sent, totalBytes, started);
          uploadAt(fileIndex + 1);
        }, failUpload);
      }
      sendAt(0);
    }
    uploadAt(0);
  }

  pick.addEventListener("click", function () {
    if (!readName()) {
      hint.className = "hint warn";
      hint.textContent = "请输入模型名称";
      return;
    }
    files.click();
  });
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
        if not (name or "").strip():
            return JSONResponse({"ok": False, "message": "请输入模型名称"}, status_code=400)
        if not files:
            return JSONResponse({"ok": False, "message": "请选择音频文件"}, status_code=400)
        try:
            folder = dataset_folder(name, create=True)
        except ValueError as exc:
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
        return JSONResponse({"ok": False, "message": "一次只能上传一片"}, status_code=400)
    if chunk_total > 8192 or chunk_index < 0 or chunk_index >= chunk_total:
        return JSONResponse({"ok": False, "message": "分片序号无效"}, status_code=400)
    if chunk_size <= 0 or chunk_size > UPLOAD_CHUNK_LIMIT:
        return JSONResponse({"ok": False, "message": "分片大小无效"}, status_code=400)
    if not valid_upload_id(upload_id):
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
                return {"ok": True, "partial": True, "chunk_index": chunk_index, "chunk_total": chunk_total}
            dest = folder / dest_name
            partial.replace(dest)
            try:
                discard_audio_parts(folder, dest_name)
            except OSError:
                logger.exception("清理残留分片失败: %s", dest_name)
        logger.info("音频已保存 %s (%s bytes)", dest, final_size)
        return {
            "ok": True,
            "saved": [dest_name],
            "skipped": [],
            "bytes": final_size,
            "folder": str(folder),
            "message": summarize_saved_audio(folder, [dest_name], []),
        }
    except ValueError as exc:
        return JSONResponse({"ok": False, "message": str(exc)}, status_code=400)
    except Exception:
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
