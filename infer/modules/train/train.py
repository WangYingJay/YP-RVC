import os
import sys
import logging
import shutil
import warnings

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(os.path.join(now_dir))

import datetime

from infer.lib.train import utils

hps = utils.get_hparams()
os.environ["CUDA_VISIBLE_DEVICES"] = hps.gpus.replace("-", ",")
n_gpus = len(hps.gpus.split("-"))
from random import randint, shuffle

import torch

from infer.lib.torch_compat import allow_legacy_torch_load

allow_legacy_torch_load()
warnings.filterwarnings(
    "ignore",
    message="`torch.nn.utils.weight_norm` is deprecated",
    category=FutureWarning,
)

try:
    import intel_extension_for_pytorch as ipex  # pylint: disable=import-error, unused-import

    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        from infer.modules.ipex.gradscaler import gradscaler_init
        from torch.xpu.amp import autocast

        _GradScaler = gradscaler_init()

        def make_scaler(enabled):
            return _GradScaler(enabled=enabled)

        ipex_init()
    else:
        raise RuntimeError("cuda")
except Exception:
    from torch.amp import GradScaler as _GradScaler
    from torch.amp import autocast as _autocast

    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)

    def make_scaler(enabled):
        return _GradScaler("cuda", enabled=enabled)

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
from time import sleep
from time import time as ttime

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from infer.lib.infer_pack import commons
from infer.lib.train.data_utils import (
    DistributedBucketSampler,
    TextAudioCollate,
    TextAudioCollateMultiNSFsid,
    TextAudioLoader,
    TextAudioLoaderMultiNSFsid,
)

if hps.version == "v1":
    from infer.lib.infer_pack.models import MultiPeriodDiscriminator
    from infer.lib.infer_pack.models import SynthesizerTrnMs256NSFsid as RVC_Model_f0
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid_nono as RVC_Model_nof0,
    )
else:
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs768NSFsid as RVC_Model_f0,
        SynthesizerTrnMs768NSFsid_nono as RVC_Model_nof0,
        MultiPeriodDiscriminatorV2 as MultiPeriodDiscriminator,
    )

from infer.lib.train.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from infer.lib.train.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from infer.lib.train.process_ckpt import savee

global_step = 0
PAUSE_FILE = "train.pause"
STOP_FILE = "train.stop"
_pause_logged = False


class EpochRecorder:
    def __init__(self):
        self.last_time = ttime()

    def elapsed(self):
        now_time = ttime()
        elapsed_time = now_time - self.last_time
        self.last_time = now_time
        return elapsed_time


def control_code():
    model_dir = hps.model_dir
    if os.path.isfile(os.path.join(model_dir, STOP_FILE)):
        return 2
    if os.path.isfile(os.path.join(model_dir, PAUSE_FILE)):
        return 1
    return 0


def wait_control(rank, logger):
    """Rank 0 读取暂停/终止标记，再广播给全部进程，避免中途停下时互相卡死。"""
    global _pause_logged
    while True:
        code = control_code() if rank == 0 else 0
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([code], dtype=torch.long)
            dist.broadcast(flag, src=0)
            code = int(flag.item())
        if code == 0:
            if rank == 0 and _pause_logged and logger is not None:
                logger.info("训练已继续")
            _pause_logged = False
            return "run"
        if code == 2:
            if rank == 0 and logger is not None:
                logger.info("收到终止指令，正在保存权重并退出")
            _pause_logged = False
            return "stop"
        if rank == 0 and logger is not None and not _pause_logged:
            logger.info("训练已暂停")
            _pause_logged = True
        if rank == 0:
            sleep(0.5)


def save_training_state(rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger, export_name=None):
    if rank != 0:
        return
    step_tag = global_step if hps.if_latest == 0 else 2333333
    g_path = os.path.join(hps.model_dir, "G_{}.pth".format(step_tag))
    utils.save_checkpoint(
        net_g,
        optim_g,
        hps.train.learning_rate,
        epoch,
        g_path,
    )
    utils.save_checkpoint(
        net_d,
        optim_d,
        hps.train.learning_rate,
        epoch,
        os.path.join(hps.model_dir, "D_{}.pth".format(step_tag)),
    )
    epoch_dir = os.path.join(hps.model_dir, "Epoch")
    os.makedirs(epoch_dir, exist_ok=True)
    epoch_copy = os.path.join(epoch_dir, "%s-%s.pth" % (hps.name, epoch))
    try:
        shutil.copy2(g_path, epoch_copy)
        if logger is not None:
            logger.info("Epoch copy saved to %s", epoch_copy)
    except OSError as exc:
        if logger is not None:
            logger.warning("Failed to copy epoch checkpoint to %s: %s", epoch_copy, exc)
    infer_name = export_name
    if infer_name is None and str(hps.save_every_weights) == "1":
        infer_name = "%s_e%s_s%s" % (hps.name, epoch, global_step)
    if not infer_name:
        return
    if hasattr(net_g, "module"):
        ckpt = net_g.module.state_dict()
    else:
        ckpt = net_g.state_dict()
    result = savee(
        ckpt,
        hps.sample_rate,
        hps.if_f0,
        infer_name,
        epoch,
        hps.version,
        hps,
    )
    if logger is not None:
        logger.info("saving ckpt %s:%s" % (infer_name, result))


def report_progress(hps, logger, message):
    if logger is not None:
        logger.info(message)
    path = os.path.join(hps.model_dir, "train_progress.txt")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(message + "\n")
        handle.flush()


def format_duration(seconds):
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return "%d小时%d分%d秒" % (hours, minutes, secs)
    if minutes:
        return "%d分%d秒" % (minutes, secs)
    return "%d秒" % secs


# 训练镜像的系统时区通常是 UTC。预计结束时间按北京时间，从当前时刻加上剩余秒数。
_DISPLAY_TZ = datetime.timezone(datetime.timedelta(hours=8))


def display_now():
    return datetime.datetime.now(_DISPLAY_TZ)


_epoch_seconds = []


def report_outputs(hps, logger, weight_name):
    import glob

    lines = []
    weight_dir = os.environ.get("RVC_WEIGHT_DIR", "assets/weights")
    weight_path = os.path.abspath(os.path.join(weight_dir, "%s.pth" % weight_name))
    if os.path.isfile(weight_path):
        lines.append("推理权重  %s" % weight_path)
    checkpoints = []
    for pattern, label in (("G_*.pth", "生成器检查点"), ("D_*.pth", "判别器检查点")):
        matches = glob.glob(os.path.join(hps.model_dir, pattern))
        if not matches:
            continue
        latest = max(matches, key=os.path.getmtime)
        checkpoints.append("%s  %s" % (label, os.path.abspath(latest)))
    lines.extend(checkpoints)
    if not lines:
        return
    report_progress(hps, logger, "产物路径")
    for line in lines:
        report_progress(hps, logger, line)


def report_epoch(hps, logger, epoch, end_epoch, elapsed):
    _epoch_seconds.append(elapsed)
    average = sum(_epoch_seconds) / len(_epoch_seconds)
    remaining = average * max(0, end_epoch - epoch)
    now = display_now()
    finish = now + datetime.timedelta(seconds=remaining)
    if finish.date() == now.date():
        finish_text = finish.strftime("%H:%M")
    else:
        finish_text = finish.strftime("%m-%d %H:%M")
    current_text = now.strftime("%m-%d %H:%M")
    report_progress(
        hps,
        logger,
        "第 %s / %s 轮  本轮 %s  剩余 %s  预计 %s 结束  当前 %s"
        % (
            epoch,
            end_epoch,
            format_duration(elapsed),
            format_duration(remaining),
            finish_text,
            current_text,
        ),
    )


def stop_and_save(rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger):
    save_training_state(
        rank,
        epoch,
        hps,
        net_g,
        net_d,
        optim_g,
        optim_d,
        logger,
        export_name=hps.name,
    )
    if rank == 0:
        report_outputs(hps, logger, hps.name)
    return "stop"


def main():
    n_gpus = torch.cuda.device_count()

    if torch.cuda.is_available() == False and torch.backends.mps.is_available() == True:
        n_gpus = 1
    if n_gpus < 1:
        # patch to unblock people without gpus. there is probably a better way.
        print("NO GPU DETECTED: falling back to CPU - this may take a while")
        n_gpus = 1
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(randint(20000, 55555))
    children = []
    logger = utils.get_logger(hps.model_dir)
    for i in range(n_gpus):
        subproc = mp.Process(
            target=run,
            args=(i, n_gpus, hps, logger),
        )
        children.append(subproc)
        subproc.start()

    failed = 0
    for i in range(n_gpus):
        children[i].join()
        exitcode = children[i].exitcode
        if exitcode not in (0, None, 2333333):
            print("训练进程异常退出，退出码 %s" % exitcode, flush=True)
            failed = exitcode or 1
    if failed:
        sys.exit(failed)


def run(rank, n_gpus, hps, logger: logging.Logger):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logger.info(hps)
        # utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="gloo", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)

    if hps.if_f0 == 1:
        train_dataset = TextAudioLoaderMultiNSFsid(hps.data.training_files, hps.data)
    else:
        train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size * n_gpus,
        # [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1200,1400],  # 16s
        [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
    )
    # It is possible that dataloader's workers are out of shared memory. Please try to raise your shared memory limit.
    # num_workers=8 -> num_workers=4
    if hps.if_f0 == 1:
        collate_fn = TextAudioCollateMultiNSFsid()
    else:
        collate_fn = TextAudioCollate()
    # Windows 上多进程 DataLoader 会在首轮直接访问冲突退出，页面看起来像训练秒结束。
    worker_count = 0 if os.name == "nt" else 4
    loader_kwargs = {}
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 8
    train_loader = DataLoader(
        train_dataset,
        num_workers=worker_count,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        **loader_kwargs,
    )
    if hps.if_f0 == 1:
        net_g = RVC_Model_f0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
            sr=hps.sample_rate,
        )
    else:
        net_g = RVC_Model_nof0(
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            **hps.model,
            is_half=hps.train.fp16_run,
        )
    if torch.cuda.is_available():
        net_g = net_g.cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
    if torch.cuda.is_available():
        net_d = net_d.cuda(rank)
    optim_g = torch.optim.AdamW(
        net_g.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    optim_d = torch.optim.AdamW(
        net_d.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )
    # net_g = DDP(net_g, device_ids=[rank], find_unused_parameters=True)
    # net_d = DDP(net_d, device_ids=[rank], find_unused_parameters=True)
    if n_gpus > 1 and torch.cuda.is_available():
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])
    elif n_gpus > 1:
        net_g = DDP(net_g)
        net_d = DDP(net_d)

    try:  # 如果能加载自动resume
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
        )  # D多半加载没事
        if rank == 0:
            logger.info("loaded D")
        # _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g,load_opt=0)
        _, _, _, epoch_str = utils.load_checkpoint(
            utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
        )
        global_step = (epoch_str - 1) * len(train_loader)
        # epoch_str = 1
        # global_step = 0
    except:  # 如果首次不能加载，加载pretrain
        # traceback.print_exc()
        epoch_str = 1
        global_step = 0
        if hps.pretrainG != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainG))
            if hasattr(net_g, "module"):
                logger.info(
                    net_g.module.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
            else:
                logger.info(
                    net_g.load_state_dict(
                        torch.load(hps.pretrainG, map_location="cpu")["model"]
                    )
                )  ##测试不加载优化器
        if hps.pretrainD != "":
            if rank == 0:
                logger.info("loaded pretrained %s" % (hps.pretrainD))
            if hasattr(net_d, "module"):
                logger.info(
                    net_d.module.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )
            else:
                logger.info(
                    net_d.load_state_dict(
                        torch.load(hps.pretrainD, map_location="cpu")["model"]
                    )
                )

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
    )

    scaler = make_scaler(hps.train.fp16_run)

    cache = []
    end_epoch = int(getattr(hps, "total_epoch", None) or hps.train.epochs)
    if epoch_str > end_epoch:
        if rank == 0:
            report_progress(
                hps,
                logger,
                "已有检查点在第 %s 轮，本次总轮数是 %s，没有新的训练。要继续训练，请把总训练轮数调到大于 %s"
                % (epoch_str, end_epoch, epoch_str),
            )
            report_outputs(hps, logger, hps.name)
        return
    for epoch in range(epoch_str, end_epoch + 1):
        if rank == 0:
            status = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                logger,
                [writer, writer_eval],
                cache,
            )
        else:
            status = train_and_evaluate(
                rank,
                epoch,
                hps,
                [net_g, net_d],
                [optim_g, optim_d],
                [scheduler_g, scheduler_d],
                scaler,
                [train_loader, None],
                None,
                None,
                cache,
            )
        if status == "stop":
            break
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            if wait_control(rank, logger) == "stop":
                return stop_and_save(
                    rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger
                )
            for batch_idx, info in enumerate(train_loader):
                if wait_control(rank, logger) == "stop":
                    return stop_and_save(
                        rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger
                    )
                # Unpack
                if hps.if_f0 == 1:
                    (
                        phone,
                        phone_lengths,
                        pitch,
                        pitchf,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                else:
                    (
                        phone,
                        phone_lengths,
                        spec,
                        spec_lengths,
                        wave,
                        wave_lengths,
                        sid,
                    ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    if hps.if_f0 == 1:
                        pitch = pitch.cuda(rank, non_blocking=True)
                        pitchf = pitchf.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                if hps.if_f0 == 1:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                pitch,
                                pitchf,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
                else:
                    cache.append(
                        (
                            batch_idx,
                            (
                                phone,
                                phone_lengths,
                                spec,
                                spec_lengths,
                                wave,
                                wave_lengths,
                                sid,
                            ),
                        )
                    )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    # Run steps
    epoch_recorder = EpochRecorder()
    if wait_control(rank, logger) == "stop":
        return stop_and_save(
            rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger
        )
    for batch_idx, info in data_iterator:
        if wait_control(rank, logger) == "stop":
            return stop_and_save(
                rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger
            )
        # Data
        ## Unpack
        if hps.if_f0 == 1:
            (
                phone,
                phone_lengths,
                pitch,
                pitchf,
                spec,
                spec_lengths,
                wave,
                wave_lengths,
                sid,
            ) = info
        else:
            phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            if hps.if_f0 == 1:
                pitch = pitch.cuda(rank, non_blocking=True)
                pitchf = pitchf.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            if hps.if_f0 == 1:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, pitch, pitchf, spec, spec_lengths, sid)
            else:
                (
                    y_hat,
                    ids_slice,
                    x_mask,
                    z_mask,
                    (z, z_p, m_p, logs_p, m_q, logs_q),
                ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            step_count = max(1, len(train_loader))
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                logger.info(
                    "Train Epoch: {} [{:.0f}%]".format(
                        epoch, 100.0 * batch_idx / step_count
                    )
                )
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9

                logger.info([global_step, lr])
                logger.info(
                    f"loss_disc={loss_disc:.3f}, loss_gen={loss_gen:.3f}, loss_fm={loss_fm:.3f},loss_mel={loss_mel:.3f}, loss_kl={loss_kl:.3f}"
                )
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
                image_dict = {
                    "slice/mel_org": utils.plot_spectrogram_to_numpy(
                        y_mel[0].data.cpu().numpy()
                    ),
                    "slice/mel_gen": utils.plot_spectrogram_to_numpy(
                        y_hat_mel[0].data.cpu().numpy()
                    ),
                    "all/mel": utils.plot_spectrogram_to_numpy(
                        mel[0].data.cpu().numpy()
                    ),
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step,
                    images=image_dict,
                    scalars=scalar_dict,
                )
        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0:
        save_training_state(
            rank, epoch, hps, net_g, net_d, optim_g, optim_d, logger
        )

    if rank == 0:
        report_epoch(
            hps,
            logger,
            epoch,
            int(getattr(hps, "total_epoch", None) or hps.train.epochs),
            epoch_recorder.elapsed(),
        )
    if epoch >= hps.total_epoch and rank == 0:
        logger.info("Training is done. The program is closed.")

        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        savee(ckpt, hps.sample_rate, hps.if_f0, hps.name, epoch, hps.version, hps)
        report_outputs(hps, logger, hps.name)
        sleep(1)
        os._exit(2333333)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    main()
