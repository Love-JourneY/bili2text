<p align="center">
  <img src="assets/light_logo2.png" alt="bili2text logo" width="360" />
</p>

> ## ⚠️ 这是改写版（fork）：运行时已被整体替换
>
> 本仓库是 [lanbinleo/bili2text](https://github.com/lanbinleo/bili2text) 的 fork。
> **我们把底层的转写运行时整个换掉了** ——
>
> | | 上游原版 | **本 fork** |
> |---|---|---|
> | 本地引擎 | openai-whisper | **Qwen3-ASR（0.6B）** |
> | 推理运行时 | **PyTorch + CUDA**（torch 2.11 + cu130） | **ONNX Runtime**（经 sherpa-onnx） |
> | 实测 venv | **4.8 GB** | **约 100 MB**（不再需要 PyTorch） |
> | 中文输出 | 繁简不稳、标点常缺/半角 | 稳定简体、全角标点 |
> | 长音频 | Whisper 原生 30 秒窗口滚动 | **Silero VAD 自动切块** |
> | 分段 | 单行大段落 | **每段一行，自带起止时间** |
>
> **两个本地引擎（whisper / sensevoice）及其 extras 已从本 fork 中移除。**
> 所有依赖 PyTorch 的代码路径都不复存在 —— 这不是"多了一个选项"，而是**换掉了发动机**。
>
> 为什么换（都是实测数据，不是口味问题）：
> 1. **运行时太重**：`--extra whisper` 会拉进整条 torch+CUDA 栈，venv 实测 **4.8 GB**。
> 2. **中文输出不稳**：同一模型对某些视频会**整篇输出繁体且完全不打标点**
>    （GPU/CPU 跑出来一模一样，不是设备问题）。
> 3. **专名更容易错**：同一段音频，Whisper 把 `Anthropic` 听成 `Andropic`、
>    把「分析代码」听成「分析带吗」；Qwen3-ASR 两处都对。
> 4. **切块后反而更快**：675 秒音频整段一次喂 RTF 1.14 → **VAD 切块后 RTF 0.36**。
>
> 想用原版请回到上游仓库。

<p align="center">
  <img src="https://img.shields.io/badge/bilibili-视频转文字-fb7299?style=flat&logo=bilibili&color=white" />
  <img src="https://img.shields.io/badge/runtime-ONNX%20%2F%20sherpa--onnx-4b8bbe" />
  <img src="https://img.shields.io/badge/engine-Qwen3--ASR-5b8c5a" />
  <img src="https://img.shields.io/github/license/lanbinleo/bili2text?style=flat&color=green" />
</p>

# bili2text（Qwen3-ASR / ONNX 改写版）

**bili2text** 是一个把 Bilibili 视频转成文字的命令行工具。

贴一个 Bilibili 链接或 BV 号进去，它会自动下载视频、提取音频、跑语音识别，最后输出一份文字稿。

除了命令行，还附带了简单的 Web 界面和桌面窗口，方便不习惯终端的用户使用。

![截图](assets/new_v_sc.png)

*PS：这个是老的界面截图*

## 转写引擎

| 引擎 | 类型 | 说明 |
| --- | --- | --- |
| **Qwen3-ASR** | 本地模型（ONNX） | **默认**。走 sherpa-onnx + ONNX Runtime，不需要 PyTorch/CUDA。支持中英混说，长音频用 Silero VAD 自动切块。 |
| **火山引擎** | 云端 API | 字节跳动的商用语音识别，需要凭据。 |

> 上游的 Whisper / SenseVoice 两个本地引擎在本 fork 中**已移除**。

## 快速开始

### 前置依赖

- **Python 3.10–3.12** 和 [uv](https://docs.astral.sh/uv/)
- **ffmpeg**（提取音频）
- **sherpa-onnx**（推理运行时）—— 它是**系统级依赖**，不通过 pip 装：
  ```bash
  # Arch
  sudo pacman -S sherpa-onnx
  # 其他平台
  pip install sherpa-onnx
  ```
- **Qwen3-ASR 模型目录**（含 `conv_frontend.onnx` / `encoder*.onnx` / `decoder*.onnx` / `tokenizer/`）
- **Silero VAD 模型**（约 2 MB，长音频切块用）
  ```bash
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  ```

### 安装

```bash
git clone https://github.com/Love-JourneY/bili2text.git
cd bili2text
uv sync --extra web      # 注意：没有 --extra whisper 了，那个 extra 已不存在
```

### 初始化配置

```bash
uv run bili2text init
```

向导会让你填 **Qwen3-ASR 模型目录**、**Silero VAD 路径**、线程数与 onnxruntime provider。

### 转写视频

```bash
uv run bili2text tx "https://www.bilibili.com/video/BV1kfDTBXEfu"
uv run bili2text tx "BV1kfDTBXEfu"
uv run bili2text tx ./my-video.mp4
uv run bili2text tx "BV1kfDTBXEfu" --model /path/to/qwen3-asr-model
```

### 环境自检

```bash
uv run bili2text doctor
```

会逐项报告：`yt-dlp` / `ffmpeg` / **`sherpa-onnx`** / **`Qwen3-ASR 模型`** / **`Silero VAD`** / `requests`。

## 命令一览

| 命令 | 缩写 | 说明 |
| --- | --- | --- |
| `bili2text transcribe` | `tx` | 转写视频或音频 |
| `bili2text batch` | - | 批量转写多条输入 |
| `bili2text bootstrap` | `init` | 配置向导 |
| `bili2text web` | `ui` | 启动 Web 界面 |
| `bili2text server` | `srv` | 启动服务模式 |
| `bili2text window` | `win` | 启动桌面窗口 |
| `bili2text doctor` | `diag` | 检查运行环境 |
| `bili2text language` | `lang` | 切换界面语言 |

## 关于 ONNX 的 GPU 加速

`provider` 配置项会原样传给 onnxruntime，可选 **`cpu` / `cuda` / `coreml`**。
**能加速，但需要带对应 EP 的 build**：

- Arch：`onnxruntime-cpu`（默认）与 **`onnxruntime-cuda`** / `onnxruntime-opt-cuda` 是**不同包**，
  想要 CUDA 就要装带 cuda 的那个（并与 sherpa-onnx 的链接一致）。
- 其他平台：装 `onnxruntime-gpu`。
- 注意 Qwen3-ASR 的解码器是**自回归**的，GPU 对它的收益不如对纯卷积/注意力模型那么线性；
  长音频真正的提速来自 **VAD 切块**（实测 RTF 1.14 → 0.36）。

## 开发

- [开发文档](docs/DEVELOPMENT.md)
- [更新日志](CHANGELOG.md)

## 许可证

MIT License —— 继承自上游 [lanbinleo/bili2text](https://github.com/lanbinleo/bili2text)。

## 使用须知

使用本工具时，请遵守你所在地区的版权法律与平台规则。确保你有权下载和转写相关视频内容。

开发者不对任何非法使用行为负责。
