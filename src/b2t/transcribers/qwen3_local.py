"""Qwen3-ASR 本地转写(ONNX Runtime / sherpa-onnx)。

这是本 fork 的**默认与推荐引擎**,用来彻底替掉原先的 openai-whisper(PyTorch+CUDA)。

为什么换掉 Whisper:
  1. **运行时太重** —— whisper extra 会拉整条 PyTorch+CUDA 栈,venv 实测 4.8GB;
     而 Qwen3-ASR 走 onnxruntime,运行时只用几百 MB。
  2. **中文输出不稳** —— 实测同一模型对某些视频会整篇出**繁体**且**完全不打标点**
     (GPU/CPU 都一样,不是设备问题)。
  3. **专名更容易错** —— 同段音频 Whisper 把 Anthropic 听成 Andropic、
     把"分析代码"听成"分析带吗";Qwen3-ASR 两处都对。

实现方式:调用 sherpa-onnx 的 `sherpa-onnx-vad-with-offline-asr` 可执行文件。
选它而不是自己拼 VAD 的原因:它**原生带 Silero VAD 切块**,长音频不会撞上
Qwen3-ASR 的上下文上限(max_total_len),而且实测切块后**更快**
(675 秒音频:整段一次喂 RTF 1.14 → VAD 切块 RTF 0.36)。

它还会输出**每段起止时间**,所以我们的文稿天然带分段,比原来的单行大段落可读。
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from contextlib import suppress
from pathlib import Path
from typing import Any

from b2t.progress import ProgressReporter
from b2t.transcribers.base import Transcriber

# 二进制候选名(Arch 包 sherpa-onnx 装到 /usr/bin;pip 装的在 PATH 里)
BINARY_NAMES = (
    "sherpa-onnx-vad-with-offline-asr",
    "sherpa-onnx-offline",  # 退化路径:没有 VAD 版时仍可跑短音频
)

# 模型文件名(sherpa-onnx 的 qwen3_asr 四件套)
FRONTEND_FILE = "conv_frontend.onnx"
ENCODER_CANDIDATES = ("encoder.int8.onnx", "encoder.onnx")
DECODER_CANDIDATES = ("decoder.int8.onnx", "decoder.onnx")

# 输出形如:  293.054 -- 313.340: 这里是一段话
_SEGMENT_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*--\s*(-?\d+(?:\.\d+)?)\s*:\s*(.*)$")


class Qwen3LocalTranscriber(Transcriber):
    name = "qwen3"

    def __init__(
        self,
        *,
        model_dir: Path,
        vad_model: Path | None = None,
        num_threads: int = 4,
        provider: str = "cpu",
        max_new_tokens: int = 512,
        max_total_len: int = 2048,
        binary: str | None = None,
    ) -> None:
        self.model_dir = Path(model_dir).expanduser()
        self.vad_model = Path(vad_model).expanduser() if vad_model else None
        self.num_threads = int(num_threads or 4)
        self.provider = (provider or "cpu").strip().lower()
        self.max_new_tokens = int(max_new_tokens or 512)
        self.max_total_len = int(max_total_len or 2048)
        self._binary = binary
        self._segments: list[dict[str, Any]] = []

    # -- 对外接口 ----------------------------------------------------------
    def transcribe(
        self,
        audio_path: Path,
        *,
        prompt: str | None = None,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        # 说明:Qwen3-ASR(sherpa-onnx)不吃 initial_prompt,所以 prompt 参数在此引擎下无效。
        # 保留签名只为与 Transcriber 接口一致。
        _ = prompt

        binary = self._resolve_binary()
        model_dir = self._validate_model_dir()
        command = self._build_command(binary, model_dir, audio_path)

        if progress is not None:
            progress.running("transcribing", message="transcribing", indeterminate=True)

        completed = subprocess.run(
            command,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip().splitlines()
            tail = " | ".join(detail[-4:]) if detail else "unknown error"
            raise RuntimeError(f"Qwen3-ASR(sherpa-onnx)失败: {tail}")

        segments = parse_segments(completed.stdout) or parse_segments(completed.stderr)
        if not segments:
            raise RuntimeError(
                "Qwen3-ASR 没有产出任何文本。sherpa-onnx 输出格式可能变了,"
                f"原始尾部:{(completed.stdout or completed.stderr or '')[-300:]!r}"
            )

        self._segments = segments
        text = join_segments(segments)
        if not text.strip():
            raise RuntimeError("transcriber returned an empty transcript")

        return {
            "text": text,
            "segments": segments,
            "language": "auto",
            "model": str(self.model_dir),
            "device": self.provider,
            "engine_detail": "sherpa-onnx / qwen3_asr",
        }

    # -- 内部 --------------------------------------------------------------
    def _resolve_binary(self) -> str:
        if self._binary:
            found = shutil.which(self._binary) or self._binary
            if Path(found).exists():
                return found
            raise RuntimeError(f"指定的 sherpa-onnx 可执行文件不存在: {self._binary}")
        for name in BINARY_NAMES:
            found = shutil.which(name)
            if found:
                return found
        raise RuntimeError(build_sherpa_missing_message())

    def _validate_model_dir(self) -> Path:
        if not self.model_dir.exists():
            raise RuntimeError(f"Qwen3-ASR 模型目录不存在: {self.model_dir}")
        if not (self.model_dir / FRONTEND_FILE).exists():
            raise RuntimeError(
                f"{self.model_dir} 看起来不是 sherpa-onnx 的 qwen3_asr 模型"
                f"(缺 {FRONTEND_FILE})。请指向含 conv_frontend/encoder/decoder/tokenizer 的目录。"
            )
        return self.model_dir

    def _pick(self, candidates: tuple[str, ...]) -> str:
        for name in candidates:
            if (self.model_dir / name).exists():
                return str(self.model_dir / name)
        raise RuntimeError(f"模型目录 {self.model_dir} 里找不到 {candidates}")

    def _build_command(self, binary: str, model_dir: Path, audio_path: Path) -> list[str]:
        command = [binary]
        if "vad" in Path(binary).name and self.vad_model:
            command.append(f"--silero-vad-model={self.vad_model}")
        command += [
            f"--qwen3-asr-conv-frontend={model_dir / FRONTEND_FILE}",
            f"--qwen3-asr-encoder={self._pick(ENCODER_CANDIDATES)}",
            f"--qwen3-asr-decoder={self._pick(DECODER_CANDIDATES)}",
            f"--qwen3-asr-tokenizer={model_dir / 'tokenizer'}",
            f"--qwen3-asr-max-new-tokens={self.max_new_tokens}",
            f"--qwen3-asr-max-total-len={self.max_total_len}",
            f"--num-threads={self.num_threads}",
            f"--provider={self.provider}",
            str(audio_path),
        ]
        return command


# -- 输出解析(独立函数,便于单测)------------------------------------------
def parse_segments(raw: str | None) -> list[dict[str, Any]]:
    """从 sherpa-onnx 输出里抽出分段。

    新版(VAD 切块)每段一行:`起始 -- 结束: 文本`;
    旧版(整段一次)输出一行 JSON,里面带 "text"。
    两种都兼容。
    """
    if not raw:
        return []
    segments: list[dict[str, Any]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        matched = _SEGMENT_RE.match(line)
        if matched:
            start, end, text = matched.groups()
            if text.strip():
                segments.append(
                    {"start": float(start), "end": float(end), "text": text.strip()}
                )
            continue
        if line.startswith("{") and '"text"' in line:
            with suppress(json.JSONDecodeError):
                payload = json.loads(line)
                text = str(payload.get("text") or "").strip()
                if text:
                    segments.append({"start": None, "end": None, "text": text})
    return segments


def join_segments(segments: list[dict[str, Any]]) -> str:
    """每段一行 —— 天然分段,比原来的单行大段落可读得多。"""
    return "\n".join(segment["text"] for segment in segments if segment.get("text"))


def build_sherpa_missing_message() -> str:
    return (
        "Qwen3-ASR 需要 sherpa-onnx 运行时,但没找到可执行文件。\n"
        "  Arch:   sudo pacman -S sherpa-onnx\n"
        "  其他:   pip install sherpa-onnx\n"
        "还需要一个 Silero VAD 模型用于长音频切块(约 2MB):\n"
        "  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
    )
