# YP-RVC 训练镜像

只包含 RVC WebUI 的训练页：处理数据、提取音高和特征、训练模型、训练特征索引、一键训练。推理、伴奏分离、ONNX、ckpt 融合和有品接口都不在这个目录里。

代码、HuBERT、RMVPE、预训练 G/D 和静音样本都在本目录内。运行时不要再指向别的仓库、本机 Python 或盘符路径。云平台只提供 GPU 驱动；Python 和 CUDA 运行库在镜像里。

## 目录里有什么

- `webui.py`：训练页，监听 `0.0.0.0:7860`
- `assets/hubert/hubert_base.pt`：特征提取
- `assets/rmvpe/rmvpe.pt`：rmvpe 音高
- `assets/pretrained/`、`assets/pretrained_v2/`：各采样率、有无音高的底模
- `logs/mute/`：写训练集列表时用的静音样本，始终留在项目里的 `logs/mute`
- `configs/v2/40k.json`：与 v1 的 40k 超参相同。上游没有单独的 v2 40k 文件，40k 训练实际用的就是这组参数

这些权重在 `.gitignore` 里，避免把几个 GB 提交进 git。打镜像时用的是磁盘上的文件，不要删。

## 环境变量

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `PORT` | `7860` | WebUI 端口 |
| `RVC_DATASET_DIR` | `/app/datasets` | 训练音频目录，界面里的默认路径 |
| `RVC_LOG_DIR` | `/app/logs` | 实验、索引和小模型输出 |

静音样本固定在镜像内的 `/app/logs/mute`，不要把数据盘直接盖到 `/app/logs` 上。实验输出改到数据盘时，把 `RVC_LOG_DIR` 指到别的目录，例如 `/data/logs`。

小模型在 `$RVC_LOG_DIR/weights`，特征索引副本在 `$RVC_LOG_DIR/indices`。

正式训练前先把数据集和实验目录指到数据盘。之后若把实例存成镜像，系统盘上的训练结果不会被烤进去。

## 在云实例里构建镜像

优云智算、仙宫云都可以：开一台带 NVIDIA 驱动的 GPU 实例，把本目录上传进去，然后构建。

```bash
docker build -t yprvc-train .
docker run --gpus all -p 7860:7860 \
  -e RVC_DATASET_DIR=/data/datasets \
  -e RVC_LOG_DIR=/data/logs \
  -v /数据盘路径/datasets:/data/datasets \
  -v /数据盘路径/logs:/data/logs \
  yprvc-train
```

`/数据盘路径` 以控制台里看到的挂载点为准。

构建完成后，在控制台把实例保存为私有镜像。下次直接选这个镜像，并同样暴露 7860。

## 用公共 PyTorch 镜像直接跑

如果实例已经是 PyTorch 2.x + CUDA 12.1，可以不自己打 Docker，在本目录执行：

```bash
apt-get update && apt-get install -y ffmpeg libsndfile1 build-essential
bash tools/install_fairseq.sh
grep -v '^fairseq==' requirements.txt > /tmp/requirements.nofairseq.txt
pip install -r /tmp/requirements.nofairseq.txt
export RVC_DATASET_DIR=/数据盘路径/datasets
export RVC_LOG_DIR=/数据盘路径/logs
bash start.sh
```

装完并确认页面能打开后，再把实例保存成私有镜像。不要使用系统里另外一份 conda，也不要指向这个目录以外的模型。

## 打开页面

服务监听 `0.0.0.0:7860`。仙宫云和优云智算都要在实例里把 7860 配成可访问端口，否则外网链接无效。

浏览器打开控制台给出的地址，把干声放进数据集目录，按页面上的五步训练。默认是 v2、40k、带音高，底模为 `assets/pretrained_v2/f0G40k.pth` 和 `f0D40k.pth`。
