# dasac 测试

## 环境准备

### 1. 基础环境

原项目要求 Python >=3.6, PyTorch >=1.4, CUDA >= 10.0。

- 在中科曙光的机器上，用 Python == 3.8.19，PyTorch == 2.1.1，CUDA == 10.2 测试通过。
- 在实验室台式机上，用 Python == 3.8.19，PyTorch == 2.3.1，CUDA == 12.2 测试通过。

其中，PyTorch 的安装可以参考[官网](https://pytorch.org/get-started/locally/)。

- 中科曙光的机器要从服务器上下载 wheel 文件手动安装。

### 2. OpenCV

安装最新版本的 OpenCV：

```bash
conda install -c conda-forge opencv
```

### 3. 其他依赖

```bash
pip install -r requirements.txt
```

## 数据准备

### 1. 数据集

在项目根目录下创建 `input` 文件夹，将数据集放在其中的 `image` 文件夹下，形如：

```bash
input
├── image
│   ├── 000001.png
│   ├── 000002.png
│   ├── 000003.png
│   ├── 000004.png
│   ├── 000005.png
│   └── ...
└── test_input.png
```

- `test_input.png` 不是必需的。

### 2. 模型

使用提供的预训练模型或者训练好的模型用于推理。对于预训练模型，可以通过脚本下载：

```bash
bash snapshots/cityscapes/baselines/download_baselines.sh
```

- 下载模型的选择可以在 `download_baselines.sh` 中修改，当前程序适配的是 `resnet101_gta` 系列的模型。

## 运行测试

- 对于中科曙光的机器，脚本应使用 `ixsmi` 获取 GPU 信息。
- 对于有 NVIDIA 驱动的机器，脚本应修改为使用 `nvidia-smi` 获取 GPU 信息。

### 1. 单通道测试

```bash
bash test_single.sh
```

成功运行后，会在当前目录下保存 `test.log` 和 `gpu_info.log` 文件。前者保存图片处理的时间，后者保存 GPU 的信息。

### 2. 双通道测试

```bash
bash test_batch.sh
```

成功运行后，会在当前目录下保存 `test_1.log`、`test_2.log` 和 `gpu_info.log` 文件。

### 3. 评估

```bash
python test.py --mode parse_logs --gpu_info_log_path <gpu_info_log_path> --test_log_path <test_log_path>
```

- `<gpu_info_log_path>` 为 GPU 信息的日志文件路径。
- `<test_log_path>` 为测试日志文件路径。对于双通道测试，选择其中一个即可。

评估结果会输出到终端，包含 GPU 温度、功耗、内存使用、利用率和处理帧率等信息。
